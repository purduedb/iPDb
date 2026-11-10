//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/aggregate/llm_agg_helpers.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/planner/tableref/bound_predictref.hpp"

#include <iostream>

namespace duckdb {

struct LlmAggBindData final : FunctionData {
	LlmAggBindData(ClientContext &context_p, unique_ptr<BoundPredictInfo> &info, const string &api_key) : context(context_p), info(std::move(info)), api_key(api_key) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto copy_info = make_uniq<BoundPredictInfo>();
		copy_info->model_type = info->model_type;
		copy_info->model_name = info->model_name;
		copy_info->model_path = info->model_path;
		copy_info->base_api = info->base_api;
		copy_info->secret = info->secret;
		copy_info->prompt = info->prompt;
		copy_info->input_set_names = info->input_set_names;
		copy_info->result_set_names = info->result_set_names;
		copy_info->input_set_types = info->input_set_types;
		copy_info->result_set_types = info->result_set_types;
		copy_info->is_join = info->is_join;
		copy_info->join_left_inputs = info->join_left_inputs;

		auto copy = make_uniq<LlmAggBindData>(context, copy_info, this->api_key);
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<LlmAggBindData>();
		return info->Equals(*other.info.get());
	}

	ClientContext &context;
	unique_ptr<BoundPredictInfo> info;
	string api_key;
};


} // namespace duckdb
