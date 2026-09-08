#include "llm_api.hpp"
#include "llm_common.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <string>

namespace duckdb {

std::unique_ptr<BatchResult> LlmApiPredictor::PredictBatch(OpenAI &api, const vector<string> &input,
                                                           const idx_t rows, idx_t batch, idx_t batch_size) {
	auto result = make_uniq<BatchResult>();

	int tokens = 0;
	int in_tokens = 0;
	int out_tokens = 0;

	idx_t frow = batch * batch_size;                // Offset of first row
	idx_t lrow = std::min(frow + batch_size, rows); // Offset of last row
	idx_t num_rows = lrow - frow;                   // Number of rows in the batch
	LLM_LOG("------------------\nBatch size: " + std::to_string(num_rows) + "\n");

	result->n_rows = num_rows;
	result->frow = frow;

	if (use_batch) {
		std::string llm_out {};

		std::stringstream ss;
		ss << this->prompt + "; \nRespond with a JSON object for each the following " << num_rows << " inputs:\n[";

		auto it_start = std::next(input.begin(), frow);
		auto it_end = lrow < rows ? std::next(input.begin(), lrow) : input.end();
		for (auto it = it_start; it != it_end; ++it) {
			ss << "{" << *it << "},\n";
		}
		ss << "]";

		std::string rewritten = ss.str();
		LLM_LOG("Prompt len: " + std::to_string(rewritten.size()) + "\n");

		nlohmann::json request;

		request["model"] = this->model_path;
		// request["temperature"] = 0.5;
		request["messages"] = {{{"content", GenerateSystemMessage(true)}, {"role", "system"}},
		                       {{"content", rewritten}, {"role", "user"}}};
		request["response_format"] = BuildArrayResponseFormat(num_rows);

		auto req_ts = steady_clock::now();
		auto completion = api.post("chat/completions", request);
		auto req_te = steady_clock::now();
		auto req_time = duration_cast<std::chrono::seconds>(req_te - req_ts).count();
		LLM_LOG("Batch request time (s):" + std::to_string(req_time) + "\n");
		if (completion.empty()) {
			LLM_LOG( "Batch call failed! Falling back to row wise calls. Error: JSON parsing failed!\n");
		} else if (completion.contains("error")) {
			CheckApiError(completion, "Batch call");
			LLM_LOG( "Falling back to row wise calls.\n");
			LLM_LOG(request.dump());
		} else {
			tokens += completion["usage"]["total_tokens"].get<int>();
			in_tokens += completion["usage"]["prompt_tokens"].get<int>();
			out_tokens += completion["usage"]["completion_tokens"].get<int>();

			llm_out = ExtractContent(completion);
			if (!llm_out.empty()) {
				LLM_LOG("Batch No: " + std::to_string(batch) + "\n" + llm_out + "||" + "\n");

				result->outputs.push_back(llm_out);
				result->tokens = tokens;
				result->in_tokens = in_tokens;
				result->out_tokens = out_tokens;
				result->time = req_time;
				result->is_concat = true;
				result->n_calls = 1;
				return std::move(result);
			}

			LLM_LOG("LLM call failed! Error: Empty message content. Falling back to non-batch\n");
			LLM_LOG(rewritten + "\n");
		}
		result->is_concat = true;
		result->n_calls = 1;
		return std::move(result);
	}

	for (int row = frow; row < lrow; ++row) {
		auto single_result = PredictOne(api, input[row], row);
		result->time += single_result->time;
		result->tokens += single_result->tokens;
		result->in_tokens += single_result->in_tokens;
		result->out_tokens += single_result->out_tokens;
		result->n_calls += single_result->n_calls;
		if (single_result->Success()) {
			result->outputs.emplace_back(std::move(single_result->outputs[0]));
		}
	}
	result->is_concat = false;
	return std::move(result);
}

std::unique_ptr<BatchResult> LlmApiPredictor::PredictEmbedBatch(OpenAI &api, const vector<string> &input,
                                                                 const idx_t rows, idx_t batch, idx_t batch_size) {
	auto result = make_uniq<BatchResult>();

	int tokens = 0;
	int in_tokens = 0;
	int out_tokens = 0;

	idx_t frow = batch * batch_size;                // Offset of first row
	idx_t lrow = std::min(frow + batch_size, rows); // Offset of last row
	idx_t num_rows = lrow - frow;                   // Number of rows in the batch
	LLM_LOG("------------------\nBatch size: " + std::to_string(num_rows) + "\n");

	result->n_rows = num_rows;
	result->frow = frow;

	nlohmann::json request;

	request["model"] = this->model_path;
	request["input"] = input;
	request["dimensions"] = 384;

	auto req_ts = steady_clock::now();
	auto embeddings = api.post("embeddings", request);
	auto req_te = steady_clock::now();

	auto req_time = duration_cast<std::chrono::seconds>(req_te - req_ts).count();
	LLM_LOG("Batch request time (s):" + std::to_string(req_time) + "\n");
	if (embeddings.contains("error")) {
		CheckApiError(embeddings, "Embedding batch call");
	} else {
		tokens += embeddings["usage"]["total_tokens"].get<int>();
		in_tokens += embeddings["usage"]["prompt_tokens"].get<int>();

		result->embeddings.reserve(embeddings["data"].size());
		for (auto &msg : embeddings["data"]) {
			if (msg["embedding"].is_array()) {
				result->embeddings.push_back(msg["embedding"].get<std::vector<float>>());
			}
		}

		LLM_LOG("Batch No: " + std::to_string(batch) + "\n");
		result->tokens = tokens;
		result->in_tokens = in_tokens;
		result->out_tokens = out_tokens;
		result->time = req_time;
		result->is_concat = false;
		result->n_calls = 1;
		return std::move(result);
	}
	result->is_concat = true;
	result->n_calls = 1;
	return std::move(result);
}

std::unique_ptr<BatchResult> LlmApiPredictor::PredictOne(OpenAI &api, const string &input, idx_t row) {
	auto result = make_uniq<BatchResult>();

	result->n_rows = 1;
	result->frow = row;

	idx_t tokens = 0;
	idx_t in_tokens = 0;
	idx_t out_tokens = 0;
	int64_t total_time = 0;

	const std::string rewritten = this->prompt + ";\n" + input;

	nlohmann::json request;
	request["model"] = this->model_path;
	request["messages"] = {{{"content", GenerateSystemMessage(false)}, {"role", "system"}},
	                       {{"content", rewritten}, {"role", "user"}}};
	request["response_format"] = BuildSingleResponseFormat();

	auto req_ts = steady_clock::now();
	auto completion = api.post("chat/completions", request);
	auto req_te = steady_clock::now();
	auto req_time = duration_cast<std::chrono::seconds>(req_te - req_ts).count();
	LLM_LOG("Request time (s):" + std::to_string(req_time) + "\n");
	if (completion.contains("error")) {
		CheckApiError(completion, "LLM call");
		result->outputs.emplace_back("");
	}
	total_time += req_time;
	tokens += completion["usage"]["total_tokens"].get<int>();
	in_tokens += completion["usage"]["prompt_tokens"].get<int>();
	out_tokens += completion["usage"]["completion_tokens"].get<int>();

	const auto llm_out = ExtractContent(completion);
	if (!llm_out.empty()) {
		LLM_LOG("Row No: " + std::to_string(row) + "\n" + llm_out + "||\n");
		result->outputs.push_back(llm_out);
		result->tokens = tokens;
		result->in_tokens = in_tokens;
		result->out_tokens = out_tokens;
		result->time = total_time;
		result->is_concat = false;
		result->n_calls = 1;
		return std::move(result);
	}

	LLM_LOG("Single call failed, row: " << row << "\n");
	result->is_concat = false;
	result->n_calls = 1;
	return std::move(result);
}

std::unique_ptr<BatchResult> LlmApiPredictor::PredictAgg(OpenAI &api, const string &input) {
	auto result = make_uniq<BatchResult>();

	std::string llm_out {};

	std::string rewritten =
	    this->prompt + "; Consider all of the following inputs and produce a single output: \n" + input;
	LLM_LOG("prompt: \n" + rewritten + "\n");

	nlohmann::json request;

	request["model"] = this->model_path;
	auto sys_msg =
	    R"(You are a helpful assistant. Always respond with a plain text. Do not include any explanations, given inputs or language specifiers.)";
	request["messages"] = {{{"content", sys_msg}, {"role", "system"}}, {{"content", rewritten}, {"role", "user"}}};

	const auto req_ts = steady_clock::now();
	auto completion = api.post("chat/completions", request);
	const auto req_te = steady_clock::now();
	const auto req_time = duration_cast<std::chrono::seconds>(req_te - req_ts).count();

	llm_out = ExtractContent(completion);
	LLM_LOG(llm_out + "||\n");
	result->outputs.push_back(llm_out);

	result->tokens = completion["usage"]["total_tokens"].get<int>();
	result->in_tokens = completion["usage"]["prompt_tokens"].get<int>();
	result->out_tokens = completion["usage"]["completion_tokens"].get<int>();
	result->time = req_time;
	result->is_concat = false;
	result->n_calls = 1;
	result->n_rows = 1;
	return std::move(result);
}

vector<string> LlmApiPredictor::ApplyOrderStrat(vector<string> &rows, vector<idx_t> &orig_order) const {
	// Build index array sorted by descending string length
	vector<idx_t> indices(rows.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&rows](idx_t a, idx_t b) {
		return rows[a].size() > rows[b].size();
	});

	// Apply same rotation pattern to indices
	vector<idx_t> reordered_indices;
	reordered_indices.reserve(indices.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		reordered_indices.push_back(indices[i]);
		// Rotate by batch_size to distribute to next "batch"
		std::rotate(reordered_indices.rbegin(), reordered_indices.rbegin() + 1, reordered_indices.rend());
	}

	// Build reordered strings and populate orig_order mapping: reordered_pos -> original_row
	orig_order.resize(rows.size());
	vector<string> reordered;
	reordered.reserve(rows.size());
	for (size_t i = 0; i < reordered_indices.size(); ++i) {
		reordered.push_back(rows[reordered_indices[i]]);
		orig_order[i] = reordered_indices[i];
	}
	return reordered;
}

} // namespace duckdb
