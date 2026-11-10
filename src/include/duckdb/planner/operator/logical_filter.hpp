//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/enums/join_decomposition_mode.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! LogicalFilter represents a filter operation (e.g. WHERE or HAVING clause)
class LogicalFilter : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_FILTER;

public:
	explicit LogicalFilter(unique_ptr<Expression> expression);
	LogicalFilter();

	vector<idx_t> projection_map;
	int limit = -1;

	//! Runtime-checked join decomposition (T2): the decomposed predicate θ and the original LLM join predicate.
	//! When set, the filter evaluates `expressions AND (θ or fallback)`, choosing at runtime. They are kept out
	//! of `expressions` so that filter reordering and pruning passes leave them alone; the expression
	//! enumeration (binding resolution, column lifetime) still visits them.
	unique_ptr<Expression> guard_condition;
	unique_ptr<Expression> guard_fallback;
	shared_ptr<DecompositionGuardState> decomposition_guard;

public:
	vector<ColumnBinding> GetColumnBindings() override;

	bool HasProjectionMap() const override {
		return !projection_map.empty();
	}

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

	bool SplitPredicates() {
		return SplitPredicates(expressions);
	}
	//! Splits up the predicates of the LogicalFilter into a set of predicates
	//! separated by AND Returns whether or not any splits were made
	static bool SplitPredicates(vector<unique_ptr<Expression>> &expressions);

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
