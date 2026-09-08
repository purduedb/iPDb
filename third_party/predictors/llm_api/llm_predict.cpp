#include "llm_api.hpp"
#include "llm_common.hpp"

#include <chrono>
#include <cstring>
#include <future>
#include <sstream>
#include <thread>
#include "../common/indicators.hpp"

namespace duckdb {

void LlmApiPredictor::PredictChunk(ClientContext &client, DataChunk &input, DataChunk &output, const idx_t rows,
                                   const PredictInfo &info, unique_ptr<PredictStats> &stats) {
	std::cout << ("No. tuples: " + std::to_string(rows) + "\n");
#if LLM_USE_THREADS
	LLM_LOG("No. of threads: " + std::to_string(this->n_threads) + "\n");
	LLM_LOG("Requests per min: " + std::to_string(this->req_per_min) + "\n");
#endif

#if OPT_TIMING
	const steady_clock::time_point begin = steady_clock::now();
#endif

#if LLM_USE_THREADS
	map<string, vector<idx_t>> tuple_id_map {};
	vector<string> unprocessed {};
	map<string, string> rep_outputs {};  // cluster key → LLM result; used by VerifyClusters

#if LLM_USE_CLUSTER
	std::vector<TupleCluster> clusters;
	if (task == PREDICT_EMBED_TASK) {
		for (idx_t i = 0; i < rows; ++i) {
			unprocessed.push_back(input.GetValue(info.input_mask[0], i).ToSQLString());
		}
	} else {
		// Cluster rows by semantic similarity; only the representative per cluster is
		// sent to the LLM and its result is propagated to all cluster members.
		clusters = GroupByClusters(input, rows, info);
		LLM_LOG("No. of clusters: " + std::to_string(clusters.size()) + "\n");
		for (const auto &cluster : clusters) {
			if (auto hit = this->cache.find(cluster.key); this->use_cache && hit != this->cache.end()) {
				for (const idx_t row : cluster.rows) {
					prompt_util.extract_row_data(hit->second, row, output, info);
				}
				continue;
			}
			tuple_id_map[cluster.key] = cluster.rows;
		}
		// unprocessed must follow tuple_id_map iteration order (alphabetical by key) so
		// that frow-based indexing in extract_array_data stays aligned.
		for (const auto &[key, _] : tuple_id_map) {
			unprocessed.push_back(key);
		}
	}
#else // !LLM_USE_CLUSTER — original per-row logic, no grouping
	for (idx_t i = 0; i < rows; ++i) {
		if (task == PREDICT_EMBED_TASK) {
			unprocessed.push_back(input.GetValue(info.input_mask[0], i).ToSQLString());
			continue;
		}
		const auto embedded = prompt_util.embed_prompt(i, input, info, true);
		if (auto hit = this->cache.find(embedded); this->use_cache && hit != this->cache.end()) {
			prompt_util.extract_row_data(hit->second, i, output, info);
		} else if (this->use_cache) {
			tuple_id_map[embedded].push_back(i);
		} else {
			unprocessed.push_back(embedded);
		}
	}
	if (this->use_cache && task == PREDICT_LLM_TASK) {
		for (auto &[embedding, tuple_ids] : tuple_id_map) {
			unprocessed.push_back(embedding);
		}
	}
#endif // LLM_USE_CLUSTER

	idx_t unprocessed_rows = unprocessed.size();
	idx_t imp_batch_size = batch_size;
	if (n_threads * batch_size > unprocessed_rows && unprocessed_rows >= n_threads) {
		imp_batch_size = unprocessed_rows / n_threads;
		if (unprocessed_rows % n_threads > 0)
			imp_batch_size++;
	}

	idx_t rounds = unprocessed_rows / imp_batch_size;
	if (unprocessed_rows % imp_batch_size != 0) {
		rounds++;
	}

	if (use_batch) {
		LLM_LOG("Max Batch Size: " + std::to_string(imp_batch_size) + ", Unprocessed: " + std::to_string(unprocessed_rows) + ", Rounds: " + std::to_string(rounds) + "\n");
	}

	LLM_LOG("Rows: " + std::to_string(unprocessed_rows) + ", Rounds: " + std::to_string(rounds) + "\n");
	double progress_step = 100.0 / rounds;
	double progress = 0;
	int step = 1;
	indicators::ProgressBar bar{
		indicators::option::BarWidth{50},
		indicators::option::Start{"["},
		indicators::option::Fill{"■"},
		indicators::option::Lead{"■"},
		indicators::option::Remainder{"-"},
		indicators::option::End{" ]"},
		indicators::option::PostfixText{"LLM Calls (rounds=" + std::to_string(rounds) +",done=0)" ")"},
		indicators::option::ForegroundColor{indicators::Color::cyan},
		indicators::option::ShowPercentage{true},
		indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}  };
	bar.set_progress(0);

	// unprocessed = ApplyOrderStrat(unprocessed, orig_order);

	std::vector<std::future<std::unique_ptr<BatchResult>>> futures;
	std::vector<idx_t> batch_fails;
	size_t total_tokens = 0;
	size_t total_in_tokens = 0;
	size_t total_out_tokens = 0;
	size_t total_calls = 0;
	size_t sub_reqs = 0;
	size_t sub_secs = 0;
	for (size_t batch = 0; batch < rounds; batch = batch + n_threads) {
		steady_clock::time_point b_ts = steady_clock::now();
		for (size_t run = 0; run < n_threads && (batch + run) * imp_batch_size < unprocessed_rows; run++) {
			idx_t batch_i = batch + run;
			if (task == PREDICT_EMBED_TASK) {
				futures.push_back(std::async(std::launch::async, &LlmApiPredictor::PredictEmbedBatch, this, std::ref(*api.get()),
				                             std::ref(unprocessed), unprocessed_rows, batch_i, imp_batch_size));
				continue;
			}

			futures.push_back(std::async(std::launch::async, &LlmApiPredictor::PredictBatch, this, std::ref(*api.get()),
			                             std::ref(unprocessed), unprocessed_rows, batch_i, imp_batch_size));
		}
		size_t run = 0;
		for (auto &f : futures) {
			const auto result = f.get();
			progress += progress_step;
			bar.set_progress(progress);
			bar.set_option(indicators::option::PostfixText{"LLM Calls (rounds=" + std::to_string(rounds) +",step=" + std::to_string(step) +")" ")"});
			step++;
			const idx_t frow = result->frow;
			sub_reqs += result->n_rows;
			total_calls += result->n_calls;
			total_tokens += result->tokens;
			total_in_tokens += result->in_tokens;
			total_out_tokens += result->out_tokens;
			if (result->is_concat) {
				if (!result->Success()) {
					for (idx_t i = frow; i < frow + result->n_rows; ++i) {
						batch_fails.emplace_back(i);
					}
					continue;
				}
#if LLM_USE_CLUSTER
				prompt_util.extract_array_data(result->outputs[0], output, tuple_id_map, frow, info, result->n_rows,
										   [this, &rep_outputs](const string &embedded, const string &llm_out) {
											   if (this->use_cache) { cache[embedded] = llm_out; }
											   rep_outputs[embedded] = llm_out;
										   });
#else
				if (this->use_cache) {
					prompt_util.extract_array_data(result->outputs[0], output, tuple_id_map, frow, info, result->n_rows,
												   [this](const string &embedded, const string &llm_out) {
													   cache[embedded] = llm_out;
												   });
				} else {
					vector<idx_t> batch_orig_rows;
					batch_orig_rows.reserve(result->n_rows);
					for (idx_t k = frow; k < frow + result->n_rows; ++k) {
						batch_orig_rows.push_back(k);
					}
					prompt_util.extract_array_data(result->outputs[0], output, batch_orig_rows, info, result->n_rows);
				}
#endif // LLM_USE_CLUSTER
			} else {
				const auto n_rows = result->n_rows;
				if (task == PREDICT_EMBED_TASK) {
					idx_t col_id;
					for (size_t j = 0; j < info.result_set_names.size(); j++) {
						auto col_name = info.result_set_names[j];

						if (col_name == "vec") {
							col_id = j;
						}
					}

					Vector& array_vector = output.data[col_id];

					Vector& child_vector = ArrayVector::GetEntry(array_vector);

					// child_vector.(embeddings.size() * 384, FlatVector::Data(child_vector));

					auto* raw_data = FlatVector::GetData<float>(child_vector);

					size_t current_offset = 0;
					for (size_t i = 0; i < result->embeddings.size(); ++i) {
						if (result->embeddings[i].size() != 384) {
							throw std::runtime_error("Embedding dimension mismatch! Expected 384.");
						}

						// Copy the entire 384-float vector into the raw_data buffer
						std::memcpy(
						    raw_data + current_offset,
						    result->embeddings[i].data(),
						    384 * sizeof(float)
						);
						current_offset += 384;
					}

					// 6. Set the size of the chunk
					output.SetCardinality(result->embeddings.size());
				} else {
					for (size_t i = 0; i < n_rows; i++) {
						PropagateSingleResult(result->outputs[i], frow + i, tuple_id_map, output, info, &rep_outputs);
					}
				}
			}
			run++;
		}
		futures.clear();
		const steady_clock::time_point b_te = steady_clock::now();
		sub_secs += duration_cast<std::chrono::seconds>(b_te - b_ts).count();
	}

	unprocessed_rows = batch_fails.size();
	for (size_t frow = 0; frow < unprocessed_rows; frow += n_threads) {
		steady_clock::time_point b_ts = steady_clock::now();
		for (size_t run = 0; run < n_threads && (frow + run) < unprocessed_rows; run++) {
			idx_t row = frow + run;
			idx_t real_row = batch_fails[row];
			futures.push_back(std::async(std::launch::async, &LlmApiPredictor::PredictOne, this, std::ref(*api.get()),
			                             std::ref(unprocessed[real_row]), real_row));
		}
		for (auto &f : futures) {
			const auto result = f.get();
			const idx_t unprocessed_idx = result->frow;
			sub_reqs += result->n_rows;
			total_calls += result->n_calls;
			total_tokens += result->tokens;
			total_in_tokens += result->in_tokens;
			total_out_tokens += result->out_tokens;

			if (result->Success()) {
				PropagateSingleResult(result->outputs[0], unprocessed_idx, tuple_id_map, output, info, &rep_outputs);
			}/**/
		}
		futures.clear();
		const steady_clock::time_point b_te = steady_clock::now();
		sub_secs += duration_cast<std::chrono::seconds>(b_te - b_ts).count();
	}
#if LLM_USE_CLUSTER
	if (task != PREDICT_EMBED_TASK) {
		VerifyClusters(input, output, clusters, rep_outputs, info);
	}
#endif
#else
	for (size_t batch = 0; batch < rounds; batch++) {
		const int frow = batch * batch_size; // Offset of first row
		auto result = PredictBatch(client, api, input, rows, batch, batch_size, info);
		total_calls += result->calls;
		total_tokens += result->tokens;
		total_in_tokens += result->in_tokens;
		total_out_tokens += result->out_tokens;
		if (result->is_concat) {
			prompt_util.extract_array_data(result->outputs[0], output, frow, info, false, result->n_rows);
		} else {
			auto n_rows = result->outputs.size();
			for (size_t i = 0; i < n_rows; i++) {
				prompt_util.extract_row_data(result->outputs[i], frow + i, output, info);
			}
		}
	}
#endif
	std::cout << ("Total calls: " + std::to_string(total_calls) + "\n");
	std::cout << ("Total tokens: " + std::to_string(total_tokens) + "\n");
	std::cout << ("Total input tokens: " + std::to_string(total_in_tokens) + "\n");
	std::cout << ("Total output tokens: " + std::to_string(total_out_tokens) + "\n");

#if OPT_TIMING
	const steady_clock::time_point end = steady_clock::now();
	const int64_t total_time = duration_cast<std::chrono::microseconds>(end - begin).count();
	stats->predict += total_time;
#endif
	std::cout << ("Total time (s): " + std::to_string(total_time * 1.0 / 1000000) + "\n");
	stats->llm_calls += total_calls;
	stats->inputs_used += total_in_tokens;
	stats->outputs_used += total_out_tokens;
	stats->tokens_used += total_tokens;
}

vector<string> LlmApiPredictor::PredictString(ClientContext &client, vector<string> &input, const PredictInfo &info) {
	vector<string> output {};

	LLM_LOG("No. agg calls: " + std::to_string(input.size()) + "\n");
#if LLM_USE_THREADS
	LLM_LOG("No. of threads: " + std::to_string(this->n_threads) + "\n");
	LLM_LOG("Requests per min: " + std::to_string(this->req_per_min) + "\n");
#endif

#if OPT_TIMING
	const steady_clock::time_point begin = steady_clock::now();
#endif

#if LLM_USE_THREADS
	std::vector<std::future<std::unique_ptr<BatchResult>>> futures;
	size_t total_tokens = 0;
	size_t total_in_tokens = 0;
	size_t total_out_tokens = 0;
	size_t sub_reqs = 0;
	size_t sub_secs = 0;

	const steady_clock::time_point b_ts = steady_clock::now();
	futures.reserve(input.size());
	for (size_t call = 0; call < input.size(); call++) {
		futures.push_back(
		    std::async(std::launch::async, &LlmApiPredictor::PredictAgg, this, std::ref(*api.get()), input[call]));
	}
	for (auto &f : futures) {
		const auto result = f.get();
		sub_reqs += result->n_rows;
		total_tokens += result->tokens;
		total_in_tokens += result->in_tokens;
		total_out_tokens += result->out_tokens;
		output.push_back(result->outputs[0]);
	}
	futures.clear();
	const steady_clock::time_point b_te = steady_clock::now();
	sub_secs += duration_cast<std::chrono::seconds>(b_te - b_ts).count();

#else
	for (size_t call = 0; call < input.size(); call++) {
		auto input_i = input[call];
		auto result = PredictOne(client, api, input_i, info);
		total_tokens += result->tokens;
		total_in_tokens += result->in_tokens;
		total_out_tokens += result->out_tokens;
		output.push_back(result->outputs[0]);
	}
#endif
	LLM_LOG("Total tokens: " + std::to_string(total_tokens) + "\n");
	LLM_LOG("Total input tokens: " + std::to_string(total_in_tokens) + "\n");
	LLM_LOG("Total output tokens: " + std::to_string(total_out_tokens) + "\n");

#if OPT_TIMING
	const steady_clock::time_point end = steady_clock::now();
	const int64_t total_time = duration_cast<std::chrono::microseconds>(end - begin).count();
	LLM_LOG("Call time (s): " + std::to_string(total_time * 1.0 / 1000000) + "\n");
#endif
	return output;
}

void LlmApiPredictor::ScanChunk(ClientContext &client, DataChunk &output, const PredictInfo &info,
                                unique_ptr<PredictStats> &stats) {
#if OPT_TIMING
	stats->move = 0;
	const steady_clock::time_point begin = steady_clock::now();
#endif

	std::string rewritten = this->prompt;

	std::string llm_out {};
	nlohmann::json request;

	request["model"] = this->model_path;
	request["messages"] = {{{"content", GenerateSystemMessage(true)}, {"role", "system"}},
	                       {{"content", rewritten}, {"role", "user"}}};
	request["response_format"] = BuildArrayResponseFormat();

	auto completion = api->post("chat/completions", request);
	llm_out = ExtractContent(completion);

	const int tokens = completion["usage"]["total_tokens"].get<int>();
	const int in_tokens = completion["usage"]["prompt_tokens"].get<int>();
	const int out_tokens = completion["usage"]["completion_tokens"].get<int>();
	LLM_LOG(llm_out + "||\n");

	LLM_LOG("Total tokens: " + std::to_string(tokens) + "\n");

	prompt_util.extract_array_data(llm_out, output, 0, info, true);

#if OPT_TIMING
	const steady_clock::time_point end = steady_clock::now();
	const int64_t total_time = duration_cast<std::chrono::microseconds>(end - begin).count();
	stats->predict += total_time;
	LLM_LOG("Total time (s): " + std::to_string(total_time * 1.0 / 1000000) + "\n");

	stats->llm_calls += 1;
	stats->tokens_used += tokens;
	stats->inputs_used += in_tokens;
	stats->outputs_used += out_tokens;
#endif
}

// Builds the left/right column-value representation strings for one row of a join input.
static std::pair<std::string, std::string> BuildJoinRepr(const DataChunk &input, const idx_t row,
														  const idx_t n_left_cols,
														  const PredictInfo &info) {
	std::stringstream left_ss, right_ss;
	for (idx_t j = 0; j < n_left_cols; ++j) {
		left_ss << info.input_set_names[j] << " = `"
				<< input.GetValue(info.input_mask[j], row).ToSQLString() << "`, ";
	}
	for (idx_t j = n_left_cols; j < info.input_mask.size(); ++j) {
		right_ss << info.input_set_names[j] << " = `"
				 << input.GetValue(info.input_mask[j], row).ToSQLString() << "`, ";
	}
	return {left_ss.str(), right_ss.str()};
}

void LlmApiPredictor::PredictJoin(ClientContext &client, DataChunk &input, DataChunk &output, const idx_t rows,
                                   const idx_t n_left_cols, const PredictInfo &info,
                                   unique_ptr<PredictStats> &stats) {
#if OPT_TIMING
	const steady_clock::time_point begin = steady_clock::now();
#endif
	D_ASSERT(n_left_cols <= info.input_mask.size());

#if LLM_USE_CLUSTER
	// Step 1: Cluster rows by their full (left+right) input combination so that identical
	// pairs are processed only once and the LLM result is propagated to all duplicates.
	const auto clusters = GroupByClusters(input, rows, info);

	// Step 2: Build unique left/right representations from the cluster representatives only.
	std::map<std::string, idx_t> left_unique_map, right_unique_map;
	std::vector<std::string> left_unique, right_unique;
	std::vector<std::pair<idx_t, idx_t>> cluster_pair_ids(clusters.size());

	for (idx_t ci = 0; ci < clusters.size(); ++ci) {
		auto [left_repr, right_repr] = BuildJoinRepr(input, clusters[ci].rows[0], n_left_cols, info);
		if (!left_unique_map.count(left_repr)) {
			left_unique_map[left_repr] = left_unique.size();
			left_unique.push_back(left_repr);
		}
		if (!right_unique_map.count(right_repr)) {
			right_unique_map[right_repr] = right_unique.size();
			right_unique.push_back(right_repr);
		}
		cluster_pair_ids[ci] = {left_unique_map[left_repr], right_unique_map[right_repr]};
	}

	LLM_LOG("PredictJoin: " + std::to_string(left_unique.size()) + " unique left, " +
			std::to_string(right_unique.size()) + " unique right (" + std::to_string(clusters.size()) +
			" unique pairs, " + std::to_string(rows) + " total rows)\n");
#else // !LLM_USE_CLUSTER — original: iterate every row individually
	// Step 1: Build unique left/right representations and record each row's (left_id, right_id).
	std::map<std::string, idx_t> left_unique_map, right_unique_map;
	std::vector<std::string> left_unique, right_unique;
	std::vector<std::pair<idx_t, idx_t>> row_ids(rows);

	for (idx_t i = 0; i < rows; ++i) {
		auto [left_repr, right_repr] = BuildJoinRepr(input, i, n_left_cols, info);

		if (!left_unique_map.count(left_repr)) {
			left_unique_map[left_repr] = left_unique.size();
			left_unique.push_back(left_repr);
		}
		if (!right_unique_map.count(right_repr)) {
			right_unique_map[right_repr] = right_unique.size();
			right_unique.push_back(right_repr);
		}
		row_ids[i] = {left_unique_map[left_repr], right_unique_map[right_repr]};
	}

	LLM_LOG("PredictJoin: " + std::to_string(left_unique.size()) + " unique left, " +
				std::to_string(right_unique.size()) + " unique right (from " + std::to_string(rows) + " rows)\n");
#endif // LLM_USE_CLUSTER

	// Step 3: Build the prompt: user's original prompt + enumerated left/right items.
	std::stringstream prompt_ss;
	prompt_ss << this->prompt << "\n\nLeft items:\n";
	for (idx_t i = 0; i < left_unique.size(); ++i) {
		prompt_ss << "[" << i << "] " << left_unique[i] << "\n";
	}
	prompt_ss << "\nRight items:\n";
	for (idx_t i = 0; i < right_unique.size(); ++i) {
		prompt_ss << "[" << i << "] " << right_unique[i] << "\n";
	}
	prompt_ss << "\nReturn only the (left_id, right_id) index pairs that match according to the prompt above.";

	// Step 3: Fixed JSON schema for the join response.
	const std::string join_schema =
	    R"({"type":"object","additionalProperties":false,"required":["matching_pairs"],)"
	    R"("properties":{"matching_pairs":{"type":"array","items":{"type":"object",)"
	    R"("additionalProperties":false,"required":["left_id","right_id"],)"
	    R"("properties":{"left_id":{"type":"integer"},"right_id":{"type":"integer"}}}}}})";

	// Step 4: Make a single LLM call with the deduplicated items.
	nlohmann::json request;
	request["model"] = this->model_path;
	const std::string sys_msg =
	    R"(You are a data matching assistant. Given enumerated left items and right items, )"
	    R"(return only the index pairs that match according to the user's criteria. )"
	    R"(Respond with matching_pairs as an array of {"left_id": <int>, "right_id": <int>} objects. )"
	    R"(Omit non-matching pairs entirely.)";
	request["messages"] = {{{"content", sys_msg}, {"role", "system"}},
	                        {{"content", prompt_ss.str()}, {"role", "user"}}};

	std::stringstream sch;
	sch << R"({"type":"json_schema","json_schema":{"name":"join_response","strict":true,"schema":)"
	    << join_schema << "}}";
	request["response_format"] = PromptUtil::parse_json(sch.str());

	const auto req_ts = steady_clock::now();
	auto completion = api->post("chat/completions", request);
	const auto req_te = steady_clock::now();
	const auto req_time = duration_cast<std::chrono::seconds>(req_te - req_ts).count();
	LLM_LOG("PredictJoin request time (s): " + std::to_string(req_time) + "\n");

	// Step 5: Parse the returned matching pair IDs.
	std::set<std::pair<idx_t, idx_t>> matched_pairs;
	if (completion.contains("error")) {
		LLM_LOG("PredictJoin LLM error: " + completion["error"].get<std::string>() + "\n");
	} else {
		const auto llm_out = ExtractContent(completion);
		if (!llm_out.empty()) {
			LLM_LOG("PredictJoin response: " + llm_out + "\n");
			try {
				auto response = nlohmann::json::parse(PromptUtil::extract_json(llm_out));
				if (response.contains("matching_pairs") && response["matching_pairs"].is_array()) {
					for (auto &pair_entry : response["matching_pairs"]) {
						const idx_t left_id = pair_entry["left_id"].get<idx_t>();
						const idx_t right_id = pair_entry["right_id"].get<idx_t>();
						if (left_id < left_unique.size() && right_id < right_unique.size()) {
							matched_pairs.insert({left_id, right_id});
						}
					}
				}
			} catch (const std::exception &e) {
				LLM_LOG("PredictJoin parse error: " + std::string(e.what()) + "\n");
			}
		}
	}

	LLM_LOG("PredictJoin: " + std::to_string(matched_pairs.size()) + " matching pairs found\n");

	// Step 6: Populate output.
#if LLM_USE_CLUSTER
	// Propagate each cluster's match result to all its member rows.
	for (idx_t ci = 0; ci < clusters.size(); ++ci) {
		const bool is_match = matched_pairs.count(cluster_pair_ids[ci]) > 0;
		for (const idx_t row : clusters[ci].rows) {
			for (size_t j = 0; j < info.result_set_names.size(); ++j) {
				const auto &output_type = info.result_set_types[j];
				if (output_type == LogicalTypeId::BOOLEAN) {
					output.SetValue(j, row, Value(is_match));
				} else if (is_match) {
					output.SetValue(j, row, Value(output_type));
				} else {
					FlatVector::SetNull(output.data[j], row, true);
				}
			}
		}
	}
#else
	for (idx_t i = 0; i < rows; ++i) {
		const bool is_match = matched_pairs.count(row_ids[i]) > 0;
		for (size_t j = 0; j < info.result_set_names.size(); ++j) {
			const auto &output_type = info.result_set_types[j];
			if (output_type == LogicalTypeId::BOOLEAN) {
				output.SetValue(j, i, Value(is_match));
			} else if (is_match) {
				output.SetValue(j, i, Value(output_type));
			} else {
				FlatVector::SetNull(output.data[j], i, true);
			}
		}
	}
#endif // LLM_USE_CLUSTER

	// Update stats.
	int total_tokens = 0, total_in = 0, total_out = 0;
	if (completion.contains("usage")) {
		total_tokens = completion["usage"]["total_tokens"].get<int>();
		total_in = completion["usage"]["prompt_tokens"].get<int>();
		total_out = completion["usage"]["completion_tokens"].get<int>();
	}
	std::cout << "PredictJoin - tokens: " + std::to_string(total_tokens) + "\n";

#if OPT_TIMING
	const steady_clock::time_point end = steady_clock::now();
	stats->predict += duration_cast<std::chrono::microseconds>(end - begin).count();
#endif
	stats->llm_calls += 1;
	stats->inputs_used += total_in;
	stats->outputs_used += total_out;
	stats->tokens_used += total_tokens;
}

// Returns true when the parsed output columns of 'a' and 'b' are identical.
// On any parse error returns false so the sample row is treated as a mismatch.
bool LlmApiPredictor::OutputsMatch(const std::string &a, const std::string &b, const PredictInfo &info) {
	try {
		const auto ja = nlohmann::json::parse(PromptUtil::extract_json(a));
		const auto jb = nlohmann::json::parse(PromptUtil::extract_json(b));
		for (const auto &col : info.result_set_names) {
			const auto va = ja.contains(col) ? ja[col] : nlohmann::json{};
			const auto vb = jb.contains(col) ? jb[col] : nlohmann::json{};
			if (va != vb) {
				return false;
			}
		}
		return true;
	} catch (...) {
		return false;
	}
}

void LlmApiPredictor::VerifyClusters(const DataChunk &input, DataChunk &output,
                                      const std::vector<TupleCluster> &clusters,
                                      const map<string, string> &rep_outputs,
                                      const PredictInfo &info) {
	LLM_LOG("VerifyClusters: checking " + std::to_string(clusters.size()) + " clusters\n");
	idx_t corrections = 0;
	for (const auto &cluster : clusters) {
		if (cluster.sample.empty()) {
			continue;
		}
		// Locate the representative's LLM output from this pass or the cache.
		std::string rep_result;
		auto it = rep_outputs.find(cluster.key);
		if (it != rep_outputs.end()) {
			rep_result = it->second;
		} else if (use_cache) {
			auto cache_it = cache.find(cluster.key);
			if (cache_it == cache.end()) {
				continue;
			}
			rep_result = cache_it->second;
		} else {
			continue;
		}
		for (const idx_t sample_row : cluster.sample) {
			const auto sample_prompt =
			    PromptUtil::embed_prompt(sample_row, input, info, /*is_multi=*/true);
			auto result = PredictOne(*api, sample_prompt, sample_row);
			if (!result->Success() || result->outputs.empty()) {
				continue;
			}
			const auto &sample_result = result->outputs[0];
			if (!OutputsMatch(rep_result, sample_result, info)) {
				LLM_LOG("VerifyClusters: mismatch at row " + std::to_string(sample_row) +
				        " — applying individual result\n");
				prompt_util.extract_row_data(sample_result, sample_row, output, info);
				++corrections;
			}
		}
	}
	LLM_LOG("VerifyClusters: " + std::to_string(corrections) + " correction(s) applied\n");
}

} // namespace duckdb
