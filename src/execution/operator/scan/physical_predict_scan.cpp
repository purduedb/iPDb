#include "duckdb/execution/operator/projection/physical_predict.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/main/secret/secret_manager.hpp"

#include <iostream>

#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 3
#include "duckdb_llama_cpp.hpp"
#endif

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
#include "llm_api.hpp"
#endif

#define CHUNK_PRED 1
// #define VEC_PRED 0

namespace duckdb {
class PredictScanGlobalState final : public GlobalSourceState {
public:
	explicit PredictScanGlobalState(ClientContext & /*context*/, unique_ptr<Predictor> p, unique_ptr<PredictStats> s)
	    : predictor(std::move(p)), stats(std::move(s)) {
	}

	unique_ptr<Predictor> predictor;
	unique_ptr<PredictStats> stats;
};

class PredictScanLocalState final : public LocalSourceState {
public:
	explicit PredictScanLocalState(ExecutionContext & /*context*/) {
	}
};

PhysicalPredictScan::PhysicalPredictScan(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<BoundPredictInfo> bound_predict_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::PREDICT, std::move(types), 0) {
	predict_info = PredictInfo(std::move(*bound_predict_p));
}

unique_ptr<Predictor> PhysicalPredictScan::InitPredictor(const PredictInfo &info, const std::string &api_key) {
#if defined(ENABLE_PREDICT) && (PREDICTOR_IMPL == 3 || defined(ENABLE_LLM_API))
	const bool is_api = info.model_path.find(".gguf") == std::string::npos;
#ifdef DEBUG
	std::cout << "Is API Model: " << is_api << std::endl;
#endif
	if (is_api) {
#if defined(ENABLE_LLM_API)
		return make_uniq<LlmApiPredictor>(info.prompt, info.base_api, api_key);
#else
		throw InternalException("Unable to infer LLM API model without `ENABLE_LLM_API` build option.");
#endif
	}
#if (PREDICTOR_IMPL == 3)
	auto pred = make_uniq<LlamaCppPredictor>(info.prompt);
	pred->is_source = true;
	return std::move(pred);
#else
	throw InternalException("Unable to infer local LLMs without `PREDICTOR_IMPL='llama_cpp'` build option.");
#endif
#else
	return nullptr;
#endif
}

unique_ptr<LocalSourceState> PhysicalPredictScan::GetLocalSourceState(ExecutionContext &context,
                                                                      GlobalSourceState & /*gstate*/) const {
	return make_uniq<PredictScanLocalState>(context);
}

unique_ptr<GlobalSourceState> PhysicalPredictScan::GetGlobalSourceState(ClientContext &context) const {
	std::string api_key;
	if (!predict_info.secret.empty()) {
		auto &secret_manager = SecretManager::Get(context);
		const auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);

		if (const auto secret_entry = secret_manager.GetSecretByName(transaction, predict_info.secret)) {
			const auto &kv_secret = dynamic_cast<const KeyValueSecret &>(*secret_entry->secret);
			api_key = kv_secret.TryGetValue("bearer_token").ToString();
		} else {
			throw CatalogException("Secret " + predict_info.secret + " for the API is not found in the catalogs!");
		}
	}

	auto stats = make_uniq<PredictStats>();
	auto p = InitPredictor(predict_info, api_key);
	p->task = static_cast<PredictorTask>(predict_info.model_type);
	p->Config(context, predict_info.options);
	p->Load(context, predict_info.model_path, stats);

	return make_uniq<PredictScanGlobalState>(context, std::move(p), std::move(stats));
}

SourceResultType PhysicalPredictScan::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	auto &g_state = input.global_state.Cast<PredictScanGlobalState>();

	auto &predictor = *g_state.predictor.get();
	predictor.ScanChunk(context.client, chunk, predict_info, g_state.stats);

	QueryProfiler::Get(context.client).Flush(*this, *g_state.stats);

	return SourceResultType::FINISHED;
}

string PhysicalPredictScan::GetName() const {
	return "LLM_SCAN";
}

InsertionOrderPreservingMap<string> PhysicalPredictScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Type"] = EnumUtil::ToString<ModelType>(predict_info.model_type);
	result["Model Path"] = predict_info.model_path;

	for (const auto &[key, value] : predict_info.options) {
		stringstream ss;
		ss << value;
		result[key] = ss.str();
	}

	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

ProgressData PhysicalPredictScan::GetProgress(ClientContext &context, GlobalSourceState & /*gstate*/) const {
	ProgressData res;
	res.SetInvalid();
	return res;
}

} // namespace duckdb
