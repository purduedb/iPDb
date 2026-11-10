#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/filter/physical_decomposition_guard.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalFilter &op) {
	D_ASSERT(op.children.size() == 1);
	reference<PhysicalOperator> plan = CreatePlan(*op.children[0]);
	if (op.decomposition_guard) {
		// runtime-checked join decomposition: choose between θ and the original LLM join predicate at runtime
		D_ASSERT(op.guard_condition && op.guard_fallback);
		auto &guard = Make<PhysicalDecompositionGuard>(plan.get().GetTypes(), std::move(op.expressions),
		                                               std::move(op.guard_condition), std::move(op.guard_fallback),
		                                               op.decomposition_guard, op.estimated_cardinality);
		guard.children.push_back(plan);
		plan = guard;
	} else if (!op.expressions.empty()) {
		// create a filter if there is anything to filter
		auto &filter = Make<PhysicalFilter>(plan.get().GetTypes(), std::move(op.expressions), op.estimated_cardinality, op.limit);
		filter.children.push_back(plan);
		plan = filter;
	}
	if (op.HasProjectionMap()) {
		// there is a projection map, generate a physical projection
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < op.projection_map.size(); i++) {
			select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], op.projection_map[i]));
		}
		auto &proj = Make<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
		proj.children.push_back(plan);
		plan = proj;
	}
	return plan;
}

} // namespace duckdb
