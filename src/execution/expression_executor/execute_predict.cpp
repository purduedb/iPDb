#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_predict_expression.hpp"
#include "duckdb/execution/operator/projection/physical_predict.hpp"

#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 3
#include "duckdb_llama_cpp.hpp"
#endif

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
#include "llm_api.hpp"
#endif

namespace duckdb {

class ExecutePredictState final : public ExpressionState {
public:
	explicit ExecutePredictState(const Expression &expr, ExpressionExecutorState &root)
	    : ExpressionState(expr, root), predict_info() {
	}

	~ExecutePredictState() override {
	}

	PredictInfo predict_info;
	unique_ptr<Predictor> predictor;
	unique_ptr<PredictStats> stats;
	bool is_loaded = false;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundPredictExpression &expr,
                                                                ExpressionExecutorState &state) {
	auto result = make_uniq<ExecutePredictState>(expr, state);
	result->predict_info = PredictInfo(*expr.bound_predict);
	idx_t mask_i = 0;
	for (auto &child : expr.children) {
		result->AddChild(*child);
		result->predict_info.input_mask.push_back(mask_i);
		mask_i++;
	}

	// Join predicates are evaluated with PredictJoin, which takes the first `n_left_cols` inputs as the left
	// side. Reorder the (executor-private) input mask/names so that left-side inputs come first. Embedding
	// columns are never on the left, so they stay at the tail. The expression's children keep their
	// placeholder order (other passes rely on it).
	auto &info = result->predict_info;
	if (info.is_join && info.join_left_inputs.size() == info.input_mask.size() &&
	    info.input_set_names.size() == info.input_mask.size()) {
		vector<idx_t> order;
		for (idx_t i = 0; i < info.input_mask.size(); i++) {
			if (info.join_left_inputs[i]) {
				order.push_back(i);
			}
		}
		info.join_n_left_cols = order.size();
		for (idx_t i = 0; i < info.input_mask.size(); i++) {
			if (!info.join_left_inputs[i]) {
				order.push_back(i);
			}
		}
		vector<idx_t> new_mask;
		vector<string> new_names;
		for (auto i : order) {
			new_mask.push_back(info.input_mask[i]);
			new_names.push_back(info.input_set_names[i]);
		}
		info.input_mask = std::move(new_mask);
		info.input_set_names = std::move(new_names);
	} else {
		info.is_join = false;
	}

	auto predictor = PhysicalPredict::InitPredictor(result->predict_info, result->predict_info.secret);
	predictor->task = PREDICT_LLM_TASK;
	result->predictor = std::move(predictor);

	auto stats = make_uniq<PredictStats>();
	result->stats = std::move(stats);

	result->Finalize();
	return std::move(result);
}

static void VerifyNullHandling(const BoundPredictExpression & /*expr*/, DataChunk & /*args*/, Vector & /*result*/) {
}

void ExpressionExecutor::Execute(const BoundPredictExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 const idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &pstate = state->Cast<ExecutePredictState>();
	auto &arguments = state->intermediate_chunk;

	DataChunk input;
	input.InitializeEmpty(state->types); // schema
	input.SetCardinality(count);

	// Join-ness is decided at bind time from the inputs' table bindings (see DeterminePredictJoinInputs);
	// a multi-input BOOLEAN predicate over a single relation is evaluated per row.
	const bool is_join = pstate.predict_info.is_join &&
	                     pstate.predict_info.result_set_types[0].id() == LogicalTypeId::BOOLEAN;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->return_type);
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
			input.data[i].Reference(arguments.data[i]);
		}
	}
	arguments.SetCardinality(count);
	arguments.Verify();

	if (!pstate.is_loaded) {
		pstate.predictor->Config(*context, pstate.predict_info.options);
		pstate.predictor->Load(*context, pstate.predict_info.model_path, pstate.stats);
		pstate.is_loaded = true;
	}

	DataChunk predictions;
	predictions.InitializeEmpty({expr.return_type}); // schema
	predictions.data[0].Reference(result);
	predictions.SetCardinality(count);

	if (is_join) {
		pstate.predictor->PredictJoin(*context, input, predictions, input.size(), pstate.predict_info.join_n_left_cols,
		                               pstate.predict_info, pstate.stats);
	} else {
		pstate.predictor->PredictChunk(*context, input, predictions, input.size(), pstate.predict_info, pstate.stats);
	}

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.return_type);

	result.Reference(predictions.data[0]);
	// predictions.data[0].Reference(nullptr);

	if (callback) {
		callback(pstate.stats->llm_calls, pstate.stats->inputs_used, pstate.stats->outputs_used, pstate.stats->tokens_used);
		pstate.stats->llm_calls = 0;
		pstate.stats->inputs_used = 0;
		pstate.stats->outputs_used = 0;
		pstate.stats->tokens_used = 0;
	}
	Verify(expr, result, count);
}

} // namespace duckdb
