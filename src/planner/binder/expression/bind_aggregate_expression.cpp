#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/function/aggregate/llm_agg_helpers.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/predict_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_binder/aggregate_binder.hpp"
#include "duckdb/planner/expression_binder/base_select_binder.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/query_node/bound_select_node.hpp"
#include "duckdb/planner/expression/bound_predict_expression.hpp"
#include "duckdb/catalog/catalog_entry/model_catalog_entry.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

static bool IsFunctionallyDependent(const unique_ptr<Expression> &expr, const vector<unique_ptr<Expression>> &deps) {
	//	Volatile expressions can't depend on anything else
	if (expr->IsVolatile()) {
		return false;
	}
	//	Constant expressions are always FD
	if (expr->IsFoldable()) {
		return true;
	}
	// If the expression matches ANY of the dependencies, then it is FD on them
	for (const auto &dep : deps) {
		// We don't need to check volatility of the dependencies because we checked it for the expression.
		if (expr->Equals(*dep)) {
			return true;
		}
	}

	// The expression doesn't match any dependency, so check ALL children.
	bool has_children = false;
	bool are_dependent = true;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		has_children = true;
		are_dependent &= IsFunctionallyDependent(child, deps);
	});
	return has_children && are_dependent;
}

static Value NegatePercentileValue(const Value &v, const bool desc) {
	if (v.IsNull()) {
		return v;
	}

	const auto frac = v.GetValue<double>();
	if (frac < 0 || frac > 1) {
		throw BinderException("PERCENTILEs can only take parameters in the range [0, 1]");
	}

	if (!desc) {
		return v;
	}

	const auto &type = v.type();
	switch (type.id()) {
	case LogicalTypeId::DECIMAL: {
		// Negate DECIMALs as DECIMAL.
		const auto integral = IntegralValue::Get(v);
		const auto width = DecimalType::GetWidth(type);
		const auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int16_t>(-integral), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int32_t>(-integral), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int64_t>(-integral), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(-integral, width, scale);
		default:
			throw InternalException("Unknown DECIMAL type");
		}
	}
	default:
		// Everything else can just be a DOUBLE
		return Value::DOUBLE(-v.GetValue<double>());
	}
}

static void NegatePercentileFractions(ClientContext &context, unique_ptr<ParsedExpression> &fractions, bool desc) {
	D_ASSERT(fractions.get());
	D_ASSERT(fractions->GetExpressionClass() == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = BoundExpression::GetExpression(*fractions);

	if (!bound->IsFoldable()) {
		return;
	}

	Value value = ExpressionExecutor::EvaluateScalar(context, *bound);
	if (value.type().id() == LogicalTypeId::LIST) {
		vector<Value> values;
		for (const auto &element_val : ListValue::GetChildren(value)) {
			values.push_back(NegatePercentileValue(element_val, desc));
		}
		if (values.empty()) {
			throw BinderException("Empty list in percentile not allowed");
		}
		bound = make_uniq<BoundConstantExpression>(Value::LIST(values));
	} else {
		bound = make_uniq<BoundConstantExpression>(NegatePercentileValue(value, desc));
	}
}

BindResult BaseSelectBinder::BindAggregate(FunctionExpression &aggr, AggregateFunctionCatalogEntry &func, idx_t depth) {
	// first bind the child of the aggregate expression (if any)
	this->bound_aggregate = true;
	unique_ptr<Expression> bound_filter;
	AggregateBinder aggregate_binder(binder, context);
	ErrorData error;

	// Now we bind the filter (if any)
	if (aggr.filter) {
		aggregate_binder.BindChild(aggr.filter, 0, error);
	}

	// Handle ordered-set aggregates by moving the single ORDER BY expression to the front of the children.
	//	https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-ORDEREDSET-TABLE
	// We also have to handle ORDER BY in the argument list, so note how many arguments we should have
	// and only inject the ordering expression if there are too few.
	idx_t ordered_set_agg = 0;
	bool negate_fractions = false;
	if (aggr.order_bys && aggr.order_bys->orders.size() == 1) {
		const auto &func_name = aggr.function_name;
		if (func_name == "mode") {
			ordered_set_agg = 1;
		} else if (func_name == "quantile_cont" || func_name == "quantile_disc") {
			ordered_set_agg = 2;

			auto &config = DBConfig::GetConfig(context);
			const auto &order = aggr.order_bys->orders[0];
			const auto sense = config.ResolveOrder(context, order.type);
			negate_fractions = (sense == OrderType::DESCENDING);
		}
	}

	for (idx_t i = 0; i < aggr.children.size(); ++i) {
		auto &child = aggr.children[i];
		aggregate_binder.BindChild(child, 0, error);
		// We have to negate the fractions for PERCENTILE_XXXX DESC
		if (!error.HasError() && ordered_set_agg && i == aggr.children.size() - 1) {
			NegatePercentileFractions(context, child, negate_fractions);
		}
	}

	// Bind the ORDER BYs, if any
	if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
		for (auto &order : aggr.order_bys->orders) {
			if (order.expression->GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
				auto &const_expr = order.expression->Cast<ConstantExpression>();
				if (!const_expr.value.type().IsIntegral()) {
					auto order_by_non_integer_literal = Settings::Get<OrderByNonIntegerLiteralSetting>(context);
					if (!order_by_non_integer_literal) {
						throw BinderException(
						    *order.expression,
						    "ORDER BY non-integer literal has no effect.\n* SET order_by_non_integer_literal=true to "
						    "allow this behavior.\n\nPerhaps you misplaced ORDER BY; ORDER BY must appear "
						    "after all regular arguments of the aggregate.");
					}
				}
			}
			aggregate_binder.BindChild(order.expression, 0, error);
		}
	}

	if (error.HasError()) {
		// failed to bind child
		if (aggregate_binder.HasBoundColumns()) {
			for (idx_t i = 0; i < aggr.children.size(); i++) {
				// however, we bound columns!
				// that means this aggregation belongs to this node
				// check if we have to resolve any errors by binding with parent binders
				auto result = aggregate_binder.BindCorrelatedColumns(aggr.children[i], error);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (result.HasError()) {
					result.error.Throw();
				}
				auto &bound_expr = BoundExpression::GetExpression(*aggr.children[i]);
				ExtractCorrelatedExpressions(binder, *bound_expr);
			}
			if (aggr.filter) {
				auto result = aggregate_binder.BindCorrelatedColumns(aggr.filter, error);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (result.HasError()) {
					result.error.Throw();
				}
				auto &bound_expr = BoundExpression::GetExpression(*aggr.filter);
				ExtractCorrelatedExpressions(binder, *bound_expr);
			}
			if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
				for (auto &order : aggr.order_bys->orders) {
					auto result = aggregate_binder.BindCorrelatedColumns(order.expression, error);
					if (result.HasError()) {
						result.error.Throw();
					}
					auto &bound_expr = BoundExpression::GetExpression(*order.expression);
					ExtractCorrelatedExpressions(binder, *bound_expr);
				}
			}
		} else {
			// we didn't bind columns, try again in children
			return BindResult(std::move(error));
		}
	} else if (depth > 0 && !aggregate_binder.HasBoundColumns()) {
		return BindResult("Aggregate with only constant parameters has to be bound in the root subquery");
	}

	if (aggr.filter) {
		auto &child = BoundExpression::GetExpression(*aggr.filter);
		bound_filter = BoundCastExpression::AddCastToType(context, std::move(child), LogicalType::BOOLEAN);
	}

	// all children bound successfully
	// extract the children and types
	vector<LogicalType> types;
	vector<LogicalType> arguments;
	vector<unique_ptr<Expression>> children;

	if (ordered_set_agg) {
		const bool order_sensitive = (aggr.function_name == "mode");
		// Inject missing ordering arguments
		if (aggr.children.size() < ordered_set_agg) {
			for (auto &order : aggr.order_bys->orders) {
				auto &child = BoundExpression::GetExpression(*order.expression);
				types.push_back(child->return_type);
				arguments.push_back(child->return_type);
				if (order_sensitive) {
					children.push_back(child->Copy());
				} else {
					children.push_back(std::move(child));
				}
			}
		}
		if (!order_sensitive) {
			aggr.order_bys->orders.clear();
		}
	}

	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = BoundExpression::GetExpression(*aggr.children[i]);
		types.push_back(child->return_type);
		arguments.push_back(child->return_type);
		children.push_back(std::move(child));
	}

	// bind the aggregate
	FunctionBinder function_binder(binder);
	auto best_function = function_binder.BindFunction(func.name, func.functions, types, error);
	if (!best_function.IsValid()) {
		error.AddQueryLocation(aggr);
		error.Throw();
	}
	// found a matching function!
	auto bound_function = func.functions.GetFunctionByOffset(best_function.GetIndex());

	if (!bound_function.CanAggregate() && bound_function.CanWindow()) {
		auto msg = StringUtil::Format("Function '%s' can only be used as a window function", bound_function.name);
		error = BinderException(msg);
		error.AddQueryLocation(aggr);
		error.Throw();
	}

	// Bind any sort columns, unless the aggregate is order-insensitive
	unique_ptr<BoundOrderModifier> order_bys;
	if (!aggr.order_bys->orders.empty()) {
		order_bys = make_uniq<BoundOrderModifier>();
		auto &config = DBConfig::GetConfig(context);
		for (auto &order : aggr.order_bys->orders) {
			auto &order_expr = BoundExpression::GetExpression(*order.expression);
			PushCollation(context, order_expr, order_expr->return_type);
			const auto sense = config.ResolveOrder(context, order.type);
			const auto null_order = config.ResolveNullOrder(context, sense, order.null_order);
			order_bys->orders.emplace_back(sense, null_order, std::move(order_expr));
		}
	}

	// If the aggregate is DISTINCT then the ORDER BYs need to be functional dependencies of the arguments.
	if (aggr.distinct && order_bys) {
		bool in_args = true;
		for (const auto &order_by : order_bys->orders) {
			in_args &= IsFunctionallyDependent(order_by.expression, children);
		}

		if (!in_args) {
			throw BinderException("In a DISTINCT aggregate, ORDER BY expressions must appear in the argument list");
		}
	}

	auto aggregate =
	    function_binder.BindAggregateFunction(bound_function, std::move(children), std::move(bound_filter),
	                                          aggr.distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
	if (aggr.export_state) {
		aggregate = ExportAggregateFunction::Bind(std::move(aggregate));
	}
	aggregate->order_bys = std::move(order_bys);

	// check for all the aggregates if this aggregate already exists
	idx_t aggr_index;
	auto entry = node.aggregate_map.find(*aggregate);
	if (entry == node.aggregate_map.end()) {
		// new aggregate: insert into aggregate list
		aggr_index = node.aggregates.size();
		node.aggregate_map[*aggregate] = aggr_index;
		node.aggregates.push_back(std::move(aggregate));
	} else {
		// duplicate aggregate: simplify refer to this aggregate
		aggr_index = entry->second;
	}

	// now create a column reference referring to the aggregate
	auto colref = make_uniq<BoundColumnRefExpression>(
	    aggr.GetAlias().empty() ? node.aggregates[aggr_index]->ToString() : aggr.GetAlias(),
	    node.aggregates[aggr_index]->return_type, ColumnBinding(node.aggregate_index, aggr_index), depth);
	// move the aggregate expression into the set of bound aggregates
	return BindResult(std::move(colref));
}

BindResult BaseSelectBinder::BindAggregate(PredictExpression &predict, idx_t depth) {
	// first bind the child of the aggregate expression (if any)
	this->bound_aggregate = true;
	AggregateBinder aggregate_binder(binder, context);
	ErrorData error;

	// // bind of each child
	for (idx_t i = 0; i < predict.children.size(); i++) {
		aggregate_binder.BindChild(predict.children[i], 0, error);
	}

	if (error.HasError()) {
		return BindResult(std::move(error));
	}
	
	// all children bound successfully
	// extract the children and types
	vector<LogicalType> child_types;
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < predict.children.size(); i++) {
		auto &child = BoundExpression::GetExpression(*predict.children[i]);
		child_types.push_back(child->return_type);
		children.push_back(std::move(child));
	}

	auto bound_predict = make_uniq<BoundPredictInfo>();
	bound_predict->model_name = predict.model_name;
	bound_predict->prompt = predict.prompt;
	bound_predict->input_set_names = std::move(predict.input_col_names);
	bound_predict->input_set_types = std::move(child_types);
	bound_predict->result_set_names.push_back(predict.out_col_name);
	bound_predict->result_set_types.push_back(predict.out_col_type);

	vector<reference<ModelCatalogEntry>> entries;
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::MODEL_ENTRY, [&](CatalogEntry &entry) {
			auto &item = entry.Cast<ModelCatalogEntry>();
			if (item.name == bound_predict->model_name) {
				entries.push_back(item);
			}
		});
	};

	if (entries.empty()) {
		throw BinderException("Model with name \"%s\" does not exist in calatog!",
				    		   bound_predict->model_name.c_str());
	}

	auto &stored_model = entries[0].get();
	auto stored_model_data = stored_model.GetData();
	bound_predict->model_type = stored_model_data.model_type;
	bound_predict->model_path = stored_model_data.model_path;
	bound_predict->base_api = stored_model_data.base_api;
	bound_predict->secret = stored_model_data.secret;

	QueryErrorContext error_context(predict.GetQueryLocation());
	// EntryLookupInfo function_lookup(CatalogType::SCALAR_FUNCTION_ENTRY, function.function_name, error_context);
	// auto func = GetCatalogEntry(function.catalog, function.schema, function_lookup, OnEntryNotFound::RETURN_NULL);
	EntryLookupInfo function_lookup(CatalogType::AGGREGATE_FUNCTION_ENTRY, predict.agg_func, error_context);
	auto cat_entry = binder.GetCatalogEntry("", "", function_lookup, OnEntryNotFound::RETURN_NULL);
	if (!cat_entry) {
		error.AddQueryLocation(predict);
		error.Throw();
	}
	auto &func = cat_entry->Cast<AggregateFunctionCatalogEntry>();

	// bind the aggregate
	FunctionBinder function_binder(binder);
	auto best_function = function_binder.BindFunction(func.name, func.functions, bound_predict->result_set_types, error);
	if (!best_function.IsValid()) {
		error.AddQueryLocation(predict);
		error.Throw();
	}
	// found a matching function!
	auto bound_function = func.functions.GetFunctionByOffset(best_function.GetIndex());

	// FunctionBinder function_binder(binder);

	// FunctionBinder function_binder(binder);
	// auto bound_function = ListFun::GetFunction();
	auto aggregate = function_binder.BindAggregateFunction(bound_function, std::move(children), nullptr, AggregateType::NON_DISTINCT);
	if (!aggregate) {
		error.AddQueryLocation(predict);
		error.Throw();
	}	

	std::string api_key;
	if (!bound_predict->secret.empty()) {
		auto &secret_manager = SecretManager::Get(context);
		const auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);

		if (const auto secret_entry = secret_manager.GetSecretByName(transaction, bound_predict->secret)) {
			const auto &kv_secret = dynamic_cast<const KeyValueSecret &>(*secret_entry->secret);
			api_key = kv_secret.TryGetValue("bearer_token").ToString();
		} else {
			throw CatalogException("LLM_AGG: Secret " + bound_predict->secret + " for the API is not found in the catalogs!");
		}
	}
	auto predict_bind = make_uniq<LlmAggBindData>(context, bound_predict, api_key);
	aggregate->bind_info = std::move(predict_bind);

	// check for all the aggregates if this aggregate already exists
	idx_t aggr_index;
	auto entry = node.aggregate_map.find(*aggregate);
	if (entry == node.aggregate_map.end()) {
		// new aggregate: insert into aggregate list
		aggr_index = node.aggregates.size();
		node.aggregate_map[*aggregate] = aggr_index;
		node.aggregates.push_back(std::move(aggregate));
	} else {
		// duplicate aggregate: simplify refer to this aggregate
		aggr_index = entry->second;
	}

	// now create a column reference referring to the aggregate
	auto colref = make_uniq<BoundColumnRefExpression>(
	    predict.GetAlias().empty() ? node.aggregates[aggr_index]->ToString() : predict.GetAlias(),
	    node.aggregates[aggr_index]->return_type, ColumnBinding(node.aggregate_index, aggr_index), depth);
	// move the aggregate expression into the set of bound aggregates
	return BindResult(std::move(colref));
}

} // namespace duckdb
