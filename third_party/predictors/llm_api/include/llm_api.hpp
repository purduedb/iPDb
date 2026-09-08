#pragma once

#include <string>
#include <memory>

#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/operator/projection/physical_predict.hpp"
#include "../../common/common.hpp"

#include "openai.hpp"

#define OPT_TIMING 1

namespace duckdb {

// A group of semantically similar rows (by cosine similarity of their input-column embeddings).
// rows[0] is the representative sent to the LLM; its result is propagated to all other members.
struct TupleCluster {
	std::string key;      // embed_prompt string of the representative row (used for LLM prompt + cache)
	vector<idx_t> rows;   // rows[0] is the representative
	vector<idx_t> sample; // subset of rows[1:] chosen for post-hoc verification
};

struct BatchResult {
	size_t tokens;
	size_t in_tokens;
	size_t out_tokens;
	int64_t time;
	idx_t frow;
	idx_t n_rows;
	idx_t n_calls;
	bool is_concat;
	std::vector<std::string> outputs;
	std::vector<std::vector<float>> embeddings;

	bool Success() const {
		return n_calls == outputs.size() || n_rows == embeddings.size();
	}
};

class LlmApiPredictor : public Predictor {
public:
	int n_predict;
	std::string prompt;
	std::string base_api;
	std::string secret;
	std::string grammar;

	idx_t n_threads;
	idx_t req_per_min;

private:
	PromptUtil prompt_util;
	unique_ptr<OpenAI> api;

public:
	LlmApiPredictor(std::string prompt, std::string base_api, std::string secret);

public:
	void Config(const ClientContext &context, const case_insensitive_map_t<Value> &options) override;
	void Load(ClientContext &client, const std::string &path, unique_ptr<PredictStats> &stats) override;
	void PredictChunk(ClientContext &client, DataChunk &input, DataChunk &output, const idx_t rows,
	                  const PredictInfo &info, unique_ptr<PredictStats> &stats) override;
	vector<string> PredictString(ClientContext &client, vector<string> &input, const PredictInfo &info) override;
	void PredictJoin(ClientContext &client, DataChunk &input, DataChunk &output, const idx_t rows,
					 const idx_t n_left_cols, const PredictInfo &info, unique_ptr<PredictStats> &stats) override;
	void ScanChunk(ClientContext &client, DataChunk &output, const PredictInfo &info,
	               unique_ptr<PredictStats> &stats) override;

	std::unique_ptr<BatchResult> PredictBatch(OpenAI &api, const vector<string> &input, const idx_t rows, idx_t batch,
	                                          idx_t batch_size);
	std::unique_ptr<BatchResult> PredictEmbedBatch(OpenAI &api, const vector<string> &input, const idx_t rows,
	                                               idx_t batch, idx_t batch_size);
	std::unique_ptr<BatchResult> PredictOne(OpenAI &api, const string &input, idx_t row);
	std::unique_ptr<BatchResult> PredictAgg(OpenAI &api, const string &input);

	vector<string> ApplyOrderStrat(vector<string> &rows, vector<idx_t> &orig_order) const;
	// Groups rows in 'input' by semantic similarity of their info.input_set_names column values.
	// Uses the embeddings API (cluster_embed_model) when configured; falls back to exact match.
	// clusters[i].rows[0] is the representative whose LLM result is propagated to all members.
	std::vector<TupleCluster> GroupByClusters(const DataChunk &input, idx_t rows, const PredictInfo &info) const;

private:
	void GenerateGrammar();
	std::string GenerateSystemMessage(bool is_array) const;

	static void CheckApiError(const nlohmann::json &completion, const string &context);
	nlohmann::json BuildSingleResponseFormat() const;
	nlohmann::json BuildArrayResponseFormat(idx_t n_rows = 0) const;

	std::vector<std::vector<float>> EmbedTexts(const std::vector<std::string> &texts) const;
	static std::string ExtractContent(const nlohmann::json &completion);
	void PropagateSingleResult(const std::string &llm_out, idx_t unprocessed_idx,
							   map<string, vector<idx_t>> &tuple_id_map,
							   DataChunk &output, const PredictInfo &info,
							   map<string, string> *rep_outputs = nullptr);

	void VerifyClusters(const DataChunk &input, DataChunk &output,
	                    const std::vector<TupleCluster> &clusters,
	                    const map<string, string> &rep_outputs,
	                    const PredictInfo &info);
	static bool OutputsMatch(const std::string &a, const std::string &b,
	                         const PredictInfo &info);
};
} // namespace duckdb
