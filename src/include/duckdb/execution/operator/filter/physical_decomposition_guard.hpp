//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/filter/physical_decomposition_guard.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/filter/decomposition_guard.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

//! Filter on top of a decomposed LLM join (T2). It evaluates either the decomposed predicate θ over the two
//! extracted columns, or -- if the runtime check fails -- the original pairwise LLM predicate. The decision is
//! made once, when the probe sample is large enough; chunks seen before that are buffered, so no row is emitted
//! before the decision and the result never mixes the two plans.
class PhysicalDecompositionGuard : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::FILTER;
	//! Maximum chunks a thread buffers while waiting for the probe sample before deciding anyway
	static constexpr idx_t MAX_BUFFERED_CHUNKS = 16;

public:
	PhysicalDecompositionGuard(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                           vector<unique_ptr<Expression>> other_predicates, unique_ptr<Expression> condition,
	                           unique_ptr<Expression> fallback, shared_ptr<DecompositionGuardState> guard,
	                           idx_t estimated_cardinality);

	//! Other filter predicates AND θ
	unique_ptr<Expression> decomposed_predicate;
	//! Other filter predicates AND the original LLM join predicate
	unique_ptr<Expression> fallback_predicate;
	shared_ptr<DecompositionGuardState> guard;

public:
	string GetName() const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state) const override;

	bool ParallelOperator() const override {
		return true;
	}
	bool RequiresFinalExecute() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

private:
	string condition_name;
	string fallback_name;
};

} // namespace duckdb
