#include "llm_api.hpp"
#include "llm_common.hpp"

#include "duckdb/main/extension_helper.hpp"

#include <string>
#include <utility>

namespace duckdb {

LlmApiPredictor::LlmApiPredictor(std::string prompt, std::string base_api, std::string secret)
    : n_predict(0), prompt(std::move(prompt)), base_api(std::move(base_api)), secret(std::move(secret)), n_threads(0),
      req_per_min(0) {
}

void LlmApiPredictor::Config(const ClientContext &context, const case_insensitive_map_t<Value> &options) {
	const auto &config = ClientConfig::GetConfig(context);

	this->batch_size = options.find("batch_size") != options.end() ? IntegerValue::Get(options.at("batch_size"))
	                                                               : config.ml_batch_size;
	this->llm_max_tokens = options.find("llm_max_tokens") != options.end()
	                           ? IntegerValue::Get(options.at("llm_max_tokens"))
	                           : config.llm_max_tokens;
	this->use_cache =
	    options.find("use_cache") != options.end() ? BooleanValue::Get(options.at("use_cache")) : config.llm_use_cache;
	this->use_batch =
	    options.find("use_batch") != options.end() ? BooleanValue::Get(options.at("use_batch")) : config.llm_use_batch;
	this->n_threads =
	    options.find("n_threads") != options.end() ? IntegerValue::Get(options.at("n_threads")) : config.llm_no_threads;
	this->req_per_min =
	    options.find("req_per_min") != options.end() ? IntegerValue::Get(options.at("req_per_min")) : 500;
	this->n_predict = 64;
}

void LlmApiPredictor::Load(ClientContext &client, const std::string &path, unique_ptr<PredictStats> &stats) {
#if OPT_TIMING
	const steady_clock::time_point begin = steady_clock::now();
#endif
	D_ASSERT(this->task == PredictorTask::PREDICT_LLM_TASK || this->task == PredictorTask::PREDICT_EMBED_TASK);

	this->model_path = path;
	LLM_LOG("Model Path: " + model_path + "\n");
	LLM_LOG("Base API: " + this->base_api + "\n");

	auto &db = DatabaseInstance::GetDatabase(client);
	this->api = OpenAI::createInstance(db, base_api, secret);

	if (this->task == PredictorTask::PREDICT_LLM_TASK) {
		GenerateGrammar();
	}

#if OPT_TIMING
	const steady_clock::time_point end = steady_clock::now();
	stats->load = duration_cast<std::chrono::microseconds>(end - begin).count();
#endif
}

// Returns the text content of the first choice in a chat/completions response,
// or an empty string when no valid content is found.
std::string LlmApiPredictor::ExtractContent(const nlohmann::json &completion) {
	if (!completion.contains("choices")) {
		return {};
	}
	for (const auto &msg : completion["choices"]) {
		if (msg.contains("message") && msg["message"]["content"].is_string()) {
			return msg["message"]["content"].get<std::string>();
		}
	}
	return {};
}

void LlmApiPredictor::CheckApiError(const nlohmann::json &completion, const string &context) {
	const int code = completion.contains("code") ? completion["code"].get<int>() : -1;
	const string reason =
	    completion.contains("error") && completion["error"].is_string() ? completion["error"].get<string>() : "unknown error";

	if (code == static_cast<int>(HTTPStatusCode::Unauthorized_401)) {
		throw HTTPException(context + " failed with 401: " + reason +
		                    ". Check that the configured API key/secret is valid.");
	}

	if (code == -1) {
		LLM_LOG( context + " failed! Connection error: " + reason + "\n");
	} else {
		LLM_LOG( context + " failed! HTTP " + std::to_string(code) + " (" +
		        HTTPUtil::GetStatusMessage(HTTPUtil::ToStatusCode(code)) + "): " + reason + "\n");
		if (code == static_cast<int>(HTTPStatusCode::TooManyRequests_429)) {
			LLM_LOG( "Rate limited by the API - consider lowering req_per_min or batch_size.\n");
		}
	}
}

} // namespace duckdb
