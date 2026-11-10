#include "duckdb/planner/expression/bound_predict_expression.hpp"

#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {

BoundPredictExpression::BoundPredictExpression(LogicalType return_type, vector<unique_ptr<Expression>> arguments)
    : Expression(ExpressionType::PREDICT, ExpressionClass::PREDICT, std::move(return_type)), children(std::move(arguments)) {
}


bool BoundPredictExpression::IsVolatile() const {
	return true;
}

bool BoundPredictExpression::IsConsistent() const {
	return false;
}

bool BoundPredictExpression::IsFoldable() const {
	return false;
}

bool BoundPredictExpression::CanThrow() const {
	return true;
}

string BoundPredictExpression::ToString() const {
	string result;
	if (!bound_predict->is_join) {
		result = bound_predict->result_set_names[0] + " (pred.)";
	} else {
		// Join-mode predicates (evaluated with PredictJoin) are marked so the routing is visible in EXPLAIN.
		idx_t n_left = 0;
		for (auto is_left : bound_predict->join_left_inputs) {
			n_left += is_left ? 1 : 0;
		}
		result = bound_predict->result_set_names[0] + " (pred. [join, " + std::to_string(n_left) + " left])";
	}
	if (!bound_predict->decomposition_note.empty()) {
		result += " [decomp: " + bound_predict->decomposition_note + "]";
	}
	return result;
}

hash_t BoundPredictExpression::Hash() const {
	const hash_t result = Expression::Hash();
	return result;
}

bool BoundPredictExpression::Equals(const BaseExpression &other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<BoundPredictExpression>();
	if (!ListEquals(children, other.children)) {
		return false;
	}
	if (!bound_predict->Equals(*other.bound_predict)) {
		return false;
	}
	return true;
}

bool BoundPredictExpression::PropagatesNullValues() const {
	return Expression::PropagatesNullValues();
}

unique_ptr<Expression> BoundPredictExpression::Copy() const {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	auto copy_info = bound_predict ? make_uniq<BoundPredictInfo>() : nullptr;
	if (bound_predict) {
		copy_info->model_type = bound_predict->model_type;
		copy_info->model_name = bound_predict->model_name;
		copy_info->model_path = bound_predict->model_path;
		copy_info->base_api = bound_predict->base_api;
		copy_info->secret = bound_predict->secret;
		copy_info->prompt = bound_predict->prompt;
		copy_info->input_set_names = bound_predict->input_set_names;
		copy_info->result_set_names = bound_predict->result_set_names;
		copy_info->input_set_types = bound_predict->input_set_types;
		copy_info->result_set_types = bound_predict->result_set_types;
		copy_info->options = bound_predict->options;
		copy_info->embedding_column_map = bound_predict->embedding_column_map;
		copy_info->is_join = bound_predict->is_join;
		copy_info->join_left_inputs = bound_predict->join_left_inputs;
		copy_info->decomposition_note = bound_predict->decomposition_note;
		copy_info->decomposition_guard = bound_predict->decomposition_guard;
		copy_info->guard_role = bound_predict->guard_role;
	}

	auto copy = make_uniq<BoundPredictExpression>(return_type, std::move(new_children));
	copy->bound_predict = std::move(copy_info);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundPredictExpression::Serialize(Serializer &serializer) const {
	Expression::Serialize(serializer);
	serializer.WriteProperty(200, "return_type", return_type);
	serializer.WriteProperty(201, "children", children);
}

unique_ptr<Expression> BoundPredictExpression::Deserialize(Deserializer &deserializer) {
	auto return_type = deserializer.ReadProperty<LogicalType>(200, "return_type");
	auto children = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(201, "children");

	auto result = make_uniq<BoundPredictExpression>(return_type, std::move(children));
	return std::move(result);
}

} // namespace duckdb
