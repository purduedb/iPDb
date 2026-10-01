//
// Created by Udesh Kumarasinghe on 11/3/25.
//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/projection/physical_predict.hpp"
#include "duckdb/common/enums/model_type.hpp"
#include "duckdb/planner/tableref/bound_predictref.hpp"

namespace duckdb {
//! PhysicalGNNPredict implements the source/sink physical PREDICT operation
class PhysicalGNNPredict final : public PhysicalOperator {
public:
	static constexpr auto TYPE = PhysicalOperatorType::PREDICT;

public:
	PhysicalGNNPredict(vector<LogicalType> types, PhysicalOperator &node_child, PhysicalOperator &edge_child,
	                   idx_t node_cardinality, idx_t edge_cardinality, BoundPredictInfo bound_predict_p);

	ModelType model_type;
	string model_path;

	idx_t num_nodes;
	idx_t num_edges;
	vector<idx_t> node_mask;
	vector<idx_t> edge_mask;
	vector<LogicalType> node_types;
	vector<LogicalType> result_set_types;
	case_insensitive_map_t<Value> options;

public:
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;

public:
	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

private:
	unique_ptr<Predictor> InitPredictor() const;

public:
	// Sink interface
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;
};
} // namespace duckdb
