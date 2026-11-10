#include "duckdb/execution/operator/projection/physical_predict.hpp"

#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/common/exception/catalog_exception.hpp"
#include "duckdb/parallel/thread_context.hpp"

#include <iostream>
#include <map>

#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 2
#include "duckdb_onnx.hpp"
#elif defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 3
#include "duckdb_llama_cpp.hpp"
#endif

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
#include "llm_api.hpp"
#endif

namespace duckdb {
class PredictState final : public OperatorState {
public:
	explicit PredictState(ExecutionContext &/*context*/, unique_ptr<Predictor> p, unique_ptr<PredictStats> s)
	    : predictor(std::move(p)), stats(std::move(s)) {
	}

	unique_ptr<Predictor> predictor;
	unique_ptr<PredictStats> stats;

	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
#ifdef DEBUG
		std::cout << "Load @run: " << stats->load << std::endl;
		std::cout << "Move @run: " << stats->move << std::endl;
		std::cout << "Predict @run: " << stats->predict << std::endl;
		std::cout << "Move Rev @run: " << stats->move_rev << std::endl;
		std::cout << "Batch Size: " << predictor->batch_size << std::endl;
#endif
		// std::map<std::string, long> stats_map {{"load", stats->load},       {"move", stats->move},
		//                                        {"predict", stats->predict}, {"move_rev", stats->move_rev},
		//                                        {"correct", stats->correct}, {"total", stats->total}};

		QueryProfiler::Get(context.client).Flush(op, *stats.get());
	}
};

PredictInfo::PredictInfo(BoundPredictInfo bound)
    : model_type(bound.model_type), model_name(std::move(bound.model_name)),
      model_path(std::move(bound.model_path)), prompt(std::move(bound.prompt)),
      base_api(std::move(bound.base_api)), secret(std::move(bound.secret)),
      input_mask(std::move(bound.input_mask)), input_set_names(std::move(bound.input_set_names)),
      result_set_names(std::move(bound.result_set_names)),
      result_set_types(std::move(bound.result_set_types)), options(std::move(bound.options)),
      embedding_column_map(std::move(bound.embedding_column_map)), is_join(bound.is_join),
      join_left_inputs(std::move(bound.join_left_inputs)) {
}

PhysicalPredict::PhysicalPredict(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperator &child, unique_ptr<BoundPredictInfo> bound_predict_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::PREDICT, std::move(types), child.estimated_cardinality) {
	children.push_back(child);
	predict_info = PredictInfo(std::move(*bound_predict_p));
}

unique_ptr<Predictor> PhysicalPredict::InitPredictor(const PredictInfo &info, const std::string &api_key) {
#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 1
	return make_uniq<TorchPredictor>();
#elif defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 2
	return make_uniq<ONNXPredictor>();
#elif defined(ENABLE_PREDICT) && (PREDICTOR_IMPL == 3 || defined(ENABLE_LLM_API))
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
	return make_uniq<LlamaCppPredictor>(info.prompt);
#else
	throw InternalException("Unable to infer local LLMs without `PREDICTOR_IMPL='llama_cpp'` build option.");
#endif
#else
	return nullptr;
#endif
}

unique_ptr<OperatorState> PhysicalPredict::GetOperatorState(ExecutionContext &context) const {
	std::string api_key;
	if (!predict_info.secret.empty()) {
		auto &secret_manager = SecretManager::Get(context.client);
		const auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context.client);

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
	p->Config(context.client, predict_info.options);
	p->Load(context.client, predict_info.model_path, stats);
	return make_uniq<PredictState>(context, std::move(p), std::move(stats));
}

OperatorResultType PhysicalPredict::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                            GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<PredictState>();

	DataChunk predictions;
	chunk.Split(predictions, input.ColumnCount());
	input.Copy(chunk);

	predictions.SetCardinality(input);

	auto &predictor = *state.predictor.get();
	predictor.PredictChunk(context.client, input, predictions, input.size(), predict_info, state.stats);

	chunk.Fuse(predictions);
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalPredict::GetName() const {
	return "PREDICT";
}

InsertionOrderPreservingMap<string> PhysicalPredict::ParamsToString() const {
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

} // namespace duckdb
