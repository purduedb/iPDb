#include "duckdb/execution/operator/projection/physical_predict.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/common/types/row/tuple_data_iterator.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <cstddef>
#include <iostream>
#include <map>

#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 2
#include "duckdb_onnx.hpp"
#endif

namespace duckdb {
class PredictGNNGlobalState : public GlobalSinkState {
public:
	PredictGNNGlobalState(ClientContext &context, BufferManager &buffer_manager_p, unique_ptr<Predictor> p,
	                      const vector<LogicalType> &node_types, unique_ptr<PredictStats> s)
	    : buffer_manager(buffer_manager_p), predictor(std::move(p)), stats(std::move(s)) {
		layout->Initialize(node_types, false);
		data_collection = make_uniq<TupleDataCollection>(buffer_manager, layout);
	}

	mutex lock;
	bool phase1_done = false;

	unique_ptr<PredictStats> stats;
	unique_ptr<Predictor> predictor;

	BufferManager &buffer_manager;
	shared_ptr<TupleDataLayout> layout;
	unique_ptr<TupleDataCollection> data_collection;

	vector<float> node_data;
	vector<int64_t> edge_data;
	vector<float> output;
};

class PredictGNNLocalState : public LocalSinkState {
public:
	PredictGNNLocalState(ExecutionContext &context, BufferManager &buffer_manager_p,
	                     const vector<LogicalType> &node_types)
	    : buffer_manager(buffer_manager_p) {
		layout->Initialize(node_types, false);
		data_collection = make_uniq<TupleDataCollection>(buffer_manager, layout);
	}

	BufferManager &buffer_manager;
	shared_ptr<TupleDataLayout> layout;
	unique_ptr<TupleDataCollection> data_collection;

	vector<float> node_data;
	vector<int64_t> edge_data;
};

class PredictGNNSourceState : public GlobalSourceState {
public:
	bool initialized = false;
	int frow = 0;
	TupleDataScanState state;
};

PhysicalGNNPredict::PhysicalGNNPredict(vector<LogicalType> types_p, PhysicalOperator &node_child, PhysicalOperator &edge_child,
									   idx_t node_cardinality, idx_t edge_cardinality, BoundPredictInfo bound_predict_p)
    : PhysicalOperator(PhysicalOperatorType::PREDICT, std::move(types_p), node_cardinality) {
	children.push_back(node_child);
	children.push_back(edge_child);
	
	model_type = bound_predict_p.model_type;
	model_path = std::move(bound_predict_p.model_path);
	num_nodes = node_cardinality;
	num_edges = edge_cardinality;
	node_mask = std::move(bound_predict_p.input_mask);
	edge_mask = std::move(bound_predict_p.opt_mask);
	node_types = std::move(bound_predict_p.input_set_types);
	result_set_types = std::move(bound_predict_p.result_set_types);
	options = std::move(bound_predict_p.options);
}

unique_ptr<Predictor> PhysicalGNNPredict::InitPredictor() const {
#if defined(ENABLE_PREDICT) && PREDICTOR_IMPL == 2
	return make_uniq<ONNXPredictor>();
#else
	return nullptr;
#endif
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
unique_ptr<GlobalSourceState> PhysicalGNNPredict::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PredictGNNSourceState>();
}

SourceResultType PhysicalGNNPredict::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSourceInput &input) const {
#ifdef DEBUG
	std::cout << "-------- GNNPredict GetData()" << std::endl;
#endif
	auto &source = input.global_state.Cast<PredictGNNSourceState>();
	auto &sink = sink_state->Cast<PredictGNNGlobalState>();

	DataChunk predictions;
	chunk.Split(predictions, chunk.ColumnCount() - (int)result_set_types.size());

	if (!source.initialized) {
		sink.data_collection->InitializeScan(source.state);
		source.initialized = true;
	}

	sink.data_collection->Scan(source.state, chunk);
	predictions.SetCardinality(chunk);

	int num_rows = static_cast<int>(chunk.size());
	int output_size = static_cast<int>(result_set_types.size());
	auto out_data_ptr = sink.output.data();
	for (idx_t idx = 0; idx < output_size; idx++) {
		float *float_pointer = out_data_ptr + idx;
		data_ptr_t start = data_ptr_cast(float_pointer);

		auto dest = predictions.data[idx].GetData();
		for (int i = 0; i < num_rows; ++i) {
			*((float *)dest + i + source.frow) = *((float *)start + static_cast<ptrdiff_t>(i * output_size));
		}
	}

	source.frow += num_rows;

	chunk.Fuse(predictions);

	return source.frow < num_nodes ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
}

void PhysicalGNNPredict::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	this->op_state.reset();
	this->sink_state.reset();

	// 'current' is the probe pipeline: add this operator
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	// on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(this->children[1]);

	auto &child_meta_pipeline2 = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline2.Build(this->children[0]);
}

vector<const_reference<PhysicalOperator>> PhysicalGNNPredict::GetSources() const {
	auto result = children[0].get().GetSources();
	if (IsSource()) {
		result.push_back(*this);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
unique_ptr<LocalSinkState> PhysicalGNNPredict::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PredictGNNLocalState>(context, BufferManager::GetBufferManager(context.client), node_types);
}

unique_ptr<GlobalSinkState> PhysicalGNNPredict::GetGlobalSinkState(ClientContext &context) const {
	auto &client_config = ClientConfig::GetConfig(context);

	auto stats = make_uniq<PredictStats>();
	auto p = InitPredictor();
	p->task = static_cast<PredictorTask>(model_type);
	p->Config(client_config, options);
	p->Load(model_path, stats);
	return make_uniq<PredictGNNGlobalState>(context, BufferManager::GetBufferManager(context), std::move(p), node_types,
	                                        std::move(stats));
}

SinkResultType PhysicalGNNPredict::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
#ifdef DEBUG
	std::cout << "-------- GNNPredict Sink() -> " << context.pipeline->GetSource()->estimated_cardinality << std::endl;
#endif
	// append to the local sink state
	auto &sink = input.local_state.Cast<PredictGNNLocalState>();
	idx_t ec = context.pipeline->GetSource()->estimated_cardinality;
	if (ec == num_nodes) {
		// Node state
		sink.data_collection->Append(chunk);

		int cols = node_mask.size();
		size_t num_rows = chunk.size();
		size_t existing_nodes = sink.node_data.size();
		sink.node_data.resize(existing_nodes + num_rows * cols);
		int input_idx = 0;
		for (auto &idx : node_mask) {
			float *start = (float *)chunk.data[idx].GetData();

			float *dest = sink.node_data.data() + existing_nodes + num_rows * input_idx;
			for (int i = 0; i < num_rows; ++i) {
				*(dest + i) = *(start + i);
			}
			input_idx++;
		}
	} else {
		// Edge state
		int cols = edge_mask.size();
		size_t num_rows = chunk.size();
		size_t existing_edges = sink.edge_data.size();
		sink.edge_data.resize(existing_edges + num_rows * cols);
		int input_idx = 0;
		for (auto &idx : edge_mask) {
			int64_t *start = (int64_t *)chunk.data[idx].GetData();

			int64_t *dest = sink.edge_data.data() + existing_edges + input_idx;
			for (int i = 0; i < num_rows; ++i) {
				*(dest + static_cast<ptrdiff_t>(i * cols)) = *(start + i);
			}
			input_idx++;
		}
	}
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
SinkCombineResultType PhysicalGNNPredict::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<PredictGNNGlobalState>();
	auto &lstate = input.local_state.Cast<PredictGNNLocalState>();

	lock_guard<mutex> glock(gstate.lock);

	// concatinate node data
	gstate.data_collection->Combine(*lstate.data_collection);

	// append local edges to global edge list
	gstate.node_data.insert(gstate.node_data.end(), std::make_move_iterator(lstate.node_data.begin()),
	                        std::make_move_iterator(lstate.node_data.end()));
	gstate.edge_data.insert(gstate.edge_data.end(), std::make_move_iterator(lstate.edge_data.begin()),
	                        std::make_move_iterator(lstate.edge_data.end()));

	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalGNNPredict::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              OperatorSinkFinalizeInput &input) const {
	// Call the ONNX GNN model
	auto &gstate = input.global_state.Cast<PredictGNNGlobalState>();

	if (!gstate.phase1_done) {
		gstate.phase1_done = true;
		return SinkFinalizeType::READY;
	}

	int64_t feature_size = static_cast<int64_t>(node_mask.size());
	int64_t edge_size = static_cast<int64_t>(edge_mask.size());
	int out_size = static_cast<int>(result_set_types.size());

	gstate.output.resize(out_size * num_nodes);
	auto &predictor = *gstate.predictor.get();
	predictor.PredictGNN(gstate.node_data, gstate.edge_data, gstate.output, num_nodes, num_edges, feature_size,
	                     edge_size, out_size, gstate.stats);

#ifdef DEBUG
	std::cout << "Load @run: " << gstate.stats->load << std::endl;
	std::cout << "Move @run: " << gstate.stats->move << std::endl;
	std::cout << "Predict @run: " << gstate.stats->predict << std::endl;
	std::cout << "Move Rev @run: " << gstate.stats->move_rev << std::endl;
#endif

	return SinkFinalizeType::READY;
}

InsertionOrderPreservingMap<string> PhysicalGNNPredict::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Type"] = EnumUtil::ToString<ModelType>(model_type);
	result["Model Path"] = model_path;

	for (const auto &item : options) {
		stringstream ss;
		ss << item.second;
		result[item.first] = ss.str();
	}

	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
