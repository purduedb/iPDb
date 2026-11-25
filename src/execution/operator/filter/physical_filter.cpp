#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include <atomic>
namespace duckdb {

PhysicalFilter::PhysicalFilter(PhysicalPlan &physical_plan, vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality, const int limit)
    : CachingPhysicalOperator(physical_plan, PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality), limit(limit) {
	if (limit != -1) {
		caching_supported = false;
	}

	D_ASSERT(!select_list.empty());
	if (select_list.size() == 1) {
		expression = std::move(select_list[0]);
		return;
	}

	// Create a conjunction from the select list.
	auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
	for (auto &expr : select_list) {
		conjunction->children.push_back(std::move(expr));
	}
	expression = std::move(conjunction);
}

class FilterGlobalState : public GlobalOperatorState {
public:
	FilterGlobalState() : limit_reached(false) {
	}

	std::atomic<bool> limit_reached;
};

class FilterState : public CachingOperatorState {
public:
	explicit FilterState(ExecutionContext &context, Expression &expr)
	    : executor(context.client, expr,
	               [&](idx_t c, idx_t i, idx_t o, idx_t t) {
		               this->counters.llm_calls += c;
		               this->counters.inputs_used += i;
		               this->counters.outputs_used += o;
		               this->counters.tokens_used += t;
	               }),
	      sel(STANDARD_VECTOR_SIZE) {
	}

	ExpressionExecutor executor;
	SelectionVector sel;
	idx_t tuples_processed = 0;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		PredictStats predict_stats;
		predict_stats.llm_calls = counters.llm_calls;
		predict_stats.inputs_used = counters.inputs_used;
		predict_stats.outputs_used = counters.outputs_used;
		predict_stats.tokens_used = counters.tokens_used;

		QueryProfiler::Get(context.client).Flush(op, predict_stats);
	}

private:
	ExpressionProfileCounters counters;
};

unique_ptr<OperatorState> PhysicalFilter::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<FilterState>(context, *expression);
}

unique_ptr<GlobalOperatorState> PhysicalFilter::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<FilterGlobalState>();
}

OperatorResultType PhysicalFilter::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<FilterGlobalState>();
	auto &state = state_p.Cast<FilterState>();

	if (limit != -1 && gstate.limit_reached.load()) {
		return OperatorResultType::FINISHED;
	}

	if (limit != -1 && static_cast<idx_t>(limit) < input.size()) {
		const idx_t LIMIT = static_cast<idx_t>(limit);
		idx_t BATCH_SIZE = LIMIT * 2;
		SelectionVector batch_sel{STANDARD_VECTOR_SIZE};
		idx_t offset = 0;

		while (offset < input.size() && chunk.size() < LIMIT && state.tuples_processed < LIMIT) {
			const idx_t count = MinValue<idx_t>(BATCH_SIZE, input.size() - offset);
			for (idx_t i = 0; i < count; i++) {
				batch_sel.set_index(i, offset + i);
			}
			idx_t this_count = state.executor.SelectExpression(input, state.sel, batch_sel, count);

			idx_t batch_sel_count = 0;
			for (idx_t i = 0; i < this_count; i++) {
				if (state.sel.get_index(i) >= offset && state.sel.get_index(i) < offset + count) {
					batch_sel.set_index(batch_sel_count++, state.sel.get_index(i));
				}
			}

			if (batch_sel_count > 0) {
				const idx_t to_append = MinValue<idx_t>(batch_sel_count, STANDARD_VECTOR_SIZE - chunk.size());
				chunk.Append(input, true, &batch_sel, to_append);
			}

			offset += count;
			if (BATCH_SIZE < input.size() - offset) {
				BATCH_SIZE *= 2;
			}
			// state.tuples_processed += chunk.size();
		}
		if (chunk.size() >= static_cast<idx_t>(limit)) {
			gstate.limit_reached.store(true);
		}
		return OperatorResultType::NEED_MORE_INPUT;
	}

	const idx_t result_count = state.executor.SelectExpression(input, state.sel);
	if (result_count == input.size()) {
		// nothing was filtered: skip adding any selection vectors
		chunk.Reference(input);
	} else {
		chunk.Slice(input, state.sel, result_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

InsertionOrderPreservingMap<string> PhysicalFilter::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["__expression__"] = expression->GetName();
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
