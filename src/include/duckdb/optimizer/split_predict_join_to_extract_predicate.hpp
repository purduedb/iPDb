//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/split_predict_join_to_extract_predicate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
class Optimizer;

//! Join decomposition. If a LogicalFilter sits on top of a join (directly or through a projection) and one of its
//! conjuncts is a two-input PREDICT whose inputs come from the two sides of the join, an LLM planning call splits
//! it into one single-input extraction per side plus a comparison θ, and the plan becomes:
//!
//!   FILTER / LLM_JOIN_GUARD  [other conjuncts AND θ(task1, task2)]
//!     └─ JOIN
//!          ├─ PROJECTION [side-0 cols + PREDICT(task on side 0)]
//!          └─ PROJECTION [side-1 cols + PREDICT(task on side 1)]
//!
//! Whether to decompose is decided at plan time from estimated costs (T1) and user settings (T3). In AUTO mode
//! the filter becomes an LLM_JOIN_GUARD that checks the extracted values at runtime and falls back to the
//! original pairwise PREDICT if they look wrong (T2). Without the check, a cross product becomes a comparison
//! join on θ. Column bindings above each inserted projection are updated via ColumnBindingReplacer.
class SplitPredictJoinToExtractPredicate {
public:
	explicit SplitPredictJoinToExtractPredicate(Optimizer &optimizer);

	//! Entry point: transform the full plan and fix parent column bindings.
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> plan);

	//! Returns true when op is a LogicalFilter whose direct child is a supported
	//! join type and at least one filter expression contains a PREDICT sub-expression
	//! whose input columns originate from *both* sides of that join.
	static bool CanOptimize(LogicalOperator &op);

private:
	// One pending binding-fix entry per transformed filter node.
	struct BindingFix {
		//! The inserted extraction projection (used as stop_operator for ColumnBindingReplacer).
		LogicalOperator *stop_operator;
		//! Parallel vectors: old_bindings[i] was a join-side binding that is
		//! now represented by new_bindings[i] in the extraction projection.
		vector<ColumnBinding> old_bindings;
		vector<ColumnBinding> new_bindings;
	};

	Optimizer &optimizer;
	vector<BindingFix> pending_fixes;
	//! Whether any filter+join was rewritten
	bool transformed = false;

	//! Recursive top-down pass that transforms matching filter+join nodes.
	unique_ptr<LogicalOperator> Transform(unique_ptr<LogicalOperator> op);

	//! Perform the actual rewrite for a single matching filter+join node.
	unique_ptr<LogicalOperator> TransformFilterJoin(unique_ptr<LogicalOperator> op);
};

} // namespace duckdb