// Speculative inference for semantic selects and projects (SET llm_speculation = 'greedy' | 'knn' | 'ahc').
//
// Per input chunk:
//   1. cluster the rows by embedding similarity (GroupByClusters; precomputed embedding column if the planner
//      attached one, otherwise an embeddings call);
//   2. infer each cluster's representative and propagate its output to every member;
//   3. infer a verification sample of each cluster (min(floor(sqrt(n - 1)), llm_speculation_verify_max) rows);
//      a sampled row always keeps its own output;
//   4. fall back (llm_speculation_fallback):
//        none     - nothing more (the behaviour of the original LLM_USE_CLUSTER code);
//        cluster  - re-infer the other members of every cluster whose sample mismatch rate exceeds
//                   llm_speculation_tolerance;
//        operator - re-infer every remaining row of the chunk if the chunk's overall sample mismatch rate
//                   exceeds llm_speculation_tolerance.
// Before step 2, a chunk whose planned saving 1 - (clusters + samples) / rows is below
// llm_speculation_min_reduction is inferred without speculation (every distinct input).
// All phases use the regular batching (ml_batch_size) and parallelism (llm_no_threads), and all of their calls
// and tokens are counted in the operator's statistics. One "Speculation:" line per chunk reports what happened.

#include "llm_api.hpp"
#include "llm_common.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <set>
#include <sstream>

namespace duckdb {

void LlmApiPredictor::RunKeyed(const map<string, vector<idx_t>> &keyed, DataChunk &output, const PredictInfo &info,
                               map<string, string> &outs, SpecCounters &counters) {
	vector<string> prompts;
	prompts.reserve(keyed.size());
	for (const auto &entry : keyed) {
		prompts.push_back(entry.first); // follows the map's iteration order, as extract_array_data expects
	}
	const idx_t n = prompts.size();
	if (n == 0) {
		return;
	}
	const idx_t threads = std::max<idx_t>(n_threads, 1);
	idx_t imp_batch_size = std::max<idx_t>(batch_size, 1);
	if (threads * imp_batch_size > n && n >= threads) {
		imp_batch_size = (n + threads - 1) / threads;
	}
	const idx_t rounds = (n + imp_batch_size - 1) / imp_batch_size;

	auto record = [&](const string &key, const string &llm_out) {
		outs[key] = llm_out;
		if (use_cache) {
			cache[key] = llm_out;
		}
	};
	auto account = [&](const BatchResult &r) {
		counters.calls += r.n_calls;
		counters.tokens += r.tokens;
		counters.in_tokens += r.in_tokens;
		counters.out_tokens += r.out_tokens;
	};

	vector<idx_t> failed; // prompt indexes whose batch failed
	std::vector<std::future<std::unique_ptr<BatchResult>>> futures;
	for (idx_t batch = 0; batch < rounds; batch += threads) {
		for (idx_t run = 0; run < threads && batch + run < rounds; run++) {
			futures.push_back(std::async(std::launch::async, &LlmApiPredictor::PredictBatch, this,
			                             std::ref(*api.get()), std::ref(prompts), n, batch + run, imp_batch_size));
		}
		for (auto &f : futures) {
			const auto result = f.get();
			account(*result);
			const idx_t frow = result->frow;
			if (result->is_concat) {
				if (!result->Success()) {
					for (idx_t i = frow; i < frow + result->n_rows; ++i) {
						failed.push_back(i);
					}
					continue;
				}
				PromptUtil::extract_array_data(result->outputs[0], output, keyed, frow, info, result->n_rows, record);
			} else if (result->outputs.size() == result->n_rows) {
				for (idx_t i = 0; i < result->n_rows; i++) {
					const auto &key = prompts[frow + i];
					record(key, result->outputs[i]);
					for (const idx_t row : keyed.at(key)) {
						PromptUtil::extract_row_data(result->outputs[i], row, output, info);
					}
				}
			} else {
				for (idx_t i = frow; i < frow + result->n_rows; ++i) {
					failed.push_back(i);
				}
			}
		}
		futures.clear();
	}

	// Failed batches: one call per row, in parallel.
	for (idx_t start = 0; start < failed.size(); start += threads) {
		for (idx_t run = 0; run < threads && start + run < failed.size(); run++) {
			const idx_t idx = failed[start + run];
			futures.push_back(std::async(std::launch::async, &LlmApiPredictor::PredictOne, this, std::ref(*api.get()),
			                             std::ref(prompts[idx]), idx));
		}
		for (auto &f : futures) {
			const auto result = f.get();
			account(*result);
			if (!result->Success() || result->outputs.empty()) {
				continue;
			}
			const auto &key = prompts[result->frow];
			record(key, result->outputs[0]);
			for (const idx_t row : keyed.at(key)) {
				PromptUtil::extract_row_data(result->outputs[0], row, output, info);
			}
		}
		futures.clear();
	}
}

void LlmApiPredictor::PredictChunkSpeculative(ClientContext &client, DataChunk &input, DataChunk &output,
                                              const idx_t rows, const PredictInfo &info,
                                              unique_ptr<PredictStats> &stats) {
	using std::chrono::steady_clock;
	const auto begin = steady_clock::now();
	std::cout << ("No. tuples: " + std::to_string(rows) + "\n");

	// --- 1. Cluster ---
	embed_tokens = 0;
	embed_calls = 0;
	const auto clusters = GroupByClusters(input, rows, info);
	const auto clustered = steady_clock::now();
	idx_t singletons = 0, max_cluster = 0;
	for (const auto &c : clusters) {
		singletons += c.rows.size() == 1;
		max_cluster = std::max<idx_t>(max_cluster, c.rows.size());
	}

	// "When to speculate", checked before any call: the calls speculation can save on this chunk. If too few,
	// infer every distinct input instead (plain deduplicated inference).
	idx_t planned_samples = 0;
	for (const auto &c : clusters) {
		planned_samples += c.sample.size();
	}
	const double reduction =
	    1.0 - static_cast<double>(clusters.size() + planned_samples) / static_cast<double>(std::max<idx_t>(rows, 1));
	if (reduction < spec.min_reduction) {
		map<string, vector<idx_t>> all_keyed;
		for (idx_t row = 0; row < rows; row++) {
			all_keyed[PromptUtil::embed_prompt(row, input, info, true)].push_back(row);
		}
		map<string, string> all_outs;
		SpecCounters all_c;
		RunKeyed(all_keyed, output, info, all_outs, all_c);
		const auto end = steady_clock::now();
		const int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		std::cout << "Speculation: rows=" << rows << " clusters=" << clusters.size() << " singletons=" << singletons
		          << " max_cluster=" << max_cluster << " samples=" << planned_samples << " reduction=" << reduction
		          << " skipped=1 full_calls=" << all_c.calls << " full_tokens=" << all_c.tokens
		          << " embed_calls=" << embed_calls << " embed_tokens=" << embed_tokens
		          << " cluster_ms=" << std::chrono::duration_cast<std::chrono::microseconds>(clustered - begin).count() / 1000.0
		          << "\n";
		std::cout << ("Total calls: " + std::to_string(all_c.calls) + "\n");
		std::cout << ("Total tokens: " + std::to_string(all_c.tokens) + "\n");
		std::cout << ("Total input tokens: " + std::to_string(all_c.in_tokens) + "\n");
		std::cout << ("Total output tokens: " + std::to_string(all_c.out_tokens) + "\n");
		std::cout << ("Total time (s): " + std::to_string(total_time * 1.0 / 1000000) + "\n");
		stats->predict += total_time;
		stats->llm_calls += all_c.calls;
		stats->inputs_used += all_c.in_tokens;
		stats->outputs_used += all_c.out_tokens;
		stats->tokens_used += all_c.tokens;
		return;
	}

	// --- 2. Representatives (output propagated to all members) ---
	SpecCounters rep_c, ver_c, fb_c;
	map<string, string> rep_outs; // representative prompt -> raw output
	map<string, vector<idx_t>> rep_keyed;
	idx_t cache_hits = 0;
	for (const auto &c : clusters) {
		if (use_cache) {
			auto hit = cache.find(c.key);
			if (hit != cache.end()) {
				rep_outs[c.key] = hit->second;
				for (const idx_t row : c.rows) {
					PromptUtil::extract_row_data(hit->second, row, output, info);
				}
				cache_hits++;
				continue;
			}
		}
		auto &dst = rep_keyed[c.key];
		dst.insert(dst.end(), c.rows.begin(), c.rows.end());
	}
	RunKeyed(rep_keyed, output, info, rep_outs, rep_c);
	const auto represented = steady_clock::now();

	// --- 3. Verification sample (each sampled row keeps its own output) ---
	map<string, vector<idx_t>> ver_keyed;
	map<string, string> ver_outs;
	idx_t n_samples = 0;
	for (const auto &c : clusters) {
		for (const idx_t row : c.sample) {
			ver_keyed[PromptUtil::embed_prompt(row, input, info, /*is_multi=*/true)].push_back(row);
			n_samples++;
		}
	}
	RunKeyed(ver_keyed, output, info, ver_outs, ver_c);
	const auto verified = steady_clock::now();

	// Mismatches per cluster (a missing output on either side counts as a mismatch).
	vector<idx_t> mismatches(clusters.size(), 0);
	idx_t total_mismatches = 0, mismatched_clusters = 0;
	for (idx_t ci = 0; ci < clusters.size(); ci++) {
		const auto &c = clusters[ci];
		if (c.sample.empty()) {
			continue;
		}
		const auto rep_it = rep_outs.find(c.key);
		for (const idx_t row : c.sample) {
			const auto ver_it = ver_outs.find(PromptUtil::embed_prompt(row, input, info, true));
			const bool ok = rep_it != rep_outs.end() && ver_it != ver_outs.end() &&
			                OutputsMatch(rep_it->second, ver_it->second, info);
			if (!ok) {
				mismatches[ci]++;
			}
		}
		total_mismatches += mismatches[ci];
		mismatched_clusters += mismatches[ci] > 0;
	}

	// --- 4. Fallback ---
	map<string, vector<idx_t>> fb_keyed;
	idx_t fallback_clusters = 0;
	auto add_cluster_rest = [&](const TupleCluster &c) {
		std::set<idx_t> skip(c.sample.begin(), c.sample.end());
		skip.insert(c.rows[0]);
		for (const idx_t row : c.rows) {
			if (!skip.count(row)) {
				fb_keyed[PromptUtil::embed_prompt(row, input, info, true)].push_back(row);
			}
		}
	};
	if (spec.fallback == 1) {
		for (idx_t ci = 0; ci < clusters.size(); ci++) {
			const auto &c = clusters[ci];
			if (!c.sample.empty() &&
			    static_cast<double>(mismatches[ci]) / static_cast<double>(c.sample.size()) > spec.tolerance) {
				add_cluster_rest(c);
				fallback_clusters++;
			}
		}
	} else if (spec.fallback == 2 && n_samples > 0 &&
	           static_cast<double>(total_mismatches) / static_cast<double>(n_samples) > spec.tolerance) {
		for (const auto &c : clusters) {
			add_cluster_rest(c);
		}
		fallback_clusters = clusters.size();
	}
	idx_t fallback_rows = 0;
	for (const auto &entry : fb_keyed) {
		fallback_rows += entry.second.size();
	}
	map<string, string> fb_outs;
	RunKeyed(fb_keyed, output, info, fb_outs, fb_c);
	const auto end = steady_clock::now();

	SpecCounters total;
	total.Add(rep_c);
	total.Add(ver_c);
	total.Add(fb_c);

	auto ms = [](steady_clock::time_point a, steady_clock::time_point b) {
		return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
	};
	std::stringstream ss;
	ss << "Speculation: rows=" << rows << " clusters=" << clusters.size() << " singletons=" << singletons
	   << " max_cluster=" << max_cluster << " reduction=" << reduction << " skipped=0 cache_hits=" << cache_hits << " rep_prompts=" << rep_keyed.size()
	   << " samples=" << n_samples << " mismatches=" << total_mismatches
	   << " mismatched_clusters=" << mismatched_clusters << " fallback_clusters=" << fallback_clusters
	   << " fallback_rows=" << fallback_rows << " rep_calls=" << rep_c.calls << " rep_tokens=" << rep_c.tokens
	   << " verify_calls=" << ver_c.calls << " verify_tokens=" << ver_c.tokens << " fallback_calls=" << fb_c.calls
	   << " fallback_tokens=" << fb_c.tokens << " embed_calls=" << embed_calls << " embed_tokens=" << embed_tokens
	   << " cluster_ms=" << ms(begin, clustered) << " rep_ms=" << ms(clustered, represented)
	   << " verify_ms=" << ms(represented, verified) << " fallback_ms=" << ms(verified, end) << "\n";
	std::cout << ss.str();
	std::cout << ("Total calls: " + std::to_string(total.calls) + "\n");
	std::cout << ("Total tokens: " + std::to_string(total.tokens) + "\n");
	std::cout << ("Total input tokens: " + std::to_string(total.in_tokens) + "\n");
	std::cout << ("Total output tokens: " + std::to_string(total.out_tokens) + "\n");

	const int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	stats->predict += total_time;
	std::cout << ("Total time (s): " + std::to_string(total_time * 1.0 / 1000000) + "\n");
	// Embedding calls are reported on the Speculation line, not as LLM calls/tokens.
	stats->llm_calls += total.calls;
	stats->inputs_used += total.in_tokens;
	stats->outputs_used += total.out_tokens;
	stats->tokens_used += total.tokens;
}

} // namespace duckdb
