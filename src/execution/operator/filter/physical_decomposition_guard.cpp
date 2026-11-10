#include "duckdb/execution/operator/filter/physical_decomposition_guard.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"

#include <iostream>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Statistics and signals
//===--------------------------------------------------------------------===//
bool DecompositionSideStats::IsUnknown(const Value &output) {
	if (output.IsNull()) {
		return true;
	}
	if (output.type().id() != LogicalTypeId::VARCHAR) {
		return false;
	}
	auto str = StringUtil::Lower(output.ToString());
	StringUtil::Trim(str);
	return str.empty() || str == "unknown" || str == "n/a" || str == "na" || str == "none" || str == "null" ||
	       str == "not available" || str == "not specified" || str == "unspecified";
}

void DecompositionSideStats::Add(const Value &input, const Value &output) {
	rows++;
	inputs.insert(input.IsNull() ? string("\x01NULL") : input.ToString());
	if (IsUnknown(output)) {
		unknown++;
		return;
	}
	auto key = output.ToString();
	auto entry = outputs.find(key);
	if (entry == outputs.end()) {
		outputs.emplace(std::move(key), std::make_pair(output, idx_t(1)));
	} else {
		entry->second.second++;
	}
}

double DecompositionSideStats::UnknownRate() const {
	return rows == 0 ? 0.0 : static_cast<double>(unknown) / static_cast<double>(rows);
}

double DecompositionSideStats::DistinctRatio() const {
	return inputs.empty() ? 0.0 : static_cast<double>(outputs.size()) / static_cast<double>(inputs.size());
}

static bool CompareValues(const Value &left, const Value &right, ExpressionType comparison) {
	switch (comparison) {
	case ExpressionType::COMPARE_EQUAL:
		return ValueOperations::Equals(left, right);
	case ExpressionType::COMPARE_NOTEQUAL:
		return ValueOperations::NotEquals(left, right);
	case ExpressionType::COMPARE_LESSTHAN:
		return ValueOperations::LessThan(left, right);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return ValueOperations::LessThanEquals(left, right);
	case ExpressionType::COMPARE_GREATERTHAN:
		return ValueOperations::GreaterThan(left, right);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ValueOperations::GreaterThanEquals(left, right);
	default:
		return false;
	}
}

static string FormatRate(double value) {
	return StringUtil::Format("%.2f", value);
}

DecompositionSignals ComputeDecompositionSignals(const DecompositionSideStats &build,
                                                 const DecompositionSideStats &probe, ExpressionType comparison,
                                                 bool build_is_task1, bool is_varchar,
                                                 const DecompositionGuardConfig &config) {
	DecompositionSignals s;
	s.build_rows = build.rows;
	s.probe_rows = probe.rows;
	s.build_unknown_rate = build.UnknownRate();
	s.probe_unknown_rate = probe.UnknownRate();
	s.build_distinct_ratio = build.DistinctRatio();
	if (build.rows == 0 || probe.rows == 0) {
		s.decision = DecompositionDecision::DECOMPOSED;
		s.reason = "nothing to check";
		return s;
	}

	const bool is_equality = comparison == ExpressionType::COMPARE_EQUAL;
	// Estimated selectivity of θ over (probe sample x build side), and (for =) the overlap.
	double matches = 0;
	if (is_equality) {
		double hits = 0;
		for (auto &probe_entry : probe.outputs) {
			auto build_entry = build.outputs.find(probe_entry.first);
			if (build_entry != build.outputs.end()) {
				hits += static_cast<double>(probe_entry.second.second);
				matches += static_cast<double>(probe_entry.second.second) *
				           static_cast<double>(build_entry->second.second);
			}
		}
		s.overlap = hits / static_cast<double>(probe.rows);
	} else {
		// Pairwise over distinct values; cap the probe side to keep this cheap.
		static constexpr idx_t MAX_PROBE_VALUES = 256;
		idx_t probe_values = 0;
		double probe_rows_used = 0;
		for (auto &probe_entry : probe.outputs) {
			if (probe_values++ >= MAX_PROBE_VALUES) {
				break;
			}
			probe_rows_used += static_cast<double>(probe_entry.second.second);
			for (auto &build_entry : build.outputs) {
				auto &task1 = build_is_task1 ? build_entry.second.first : probe_entry.second.first;
				auto &task2 = build_is_task1 ? probe_entry.second.first : build_entry.second.first;
				if (CompareValues(task1, task2, comparison)) {
					matches += static_cast<double>(probe_entry.second.second) *
					           static_cast<double>(build_entry.second.second);
				}
			}
		}
		// Scale the capped sample back to the probe rows seen (unknown rows never match).
		auto known_rows = static_cast<double>(probe.rows - probe.unknown);
		if (probe_rows_used > 0 && probe_rows_used < known_rows) {
			matches *= known_rows / probe_rows_used;
		}
	}
	s.selectivity = matches / (static_cast<double>(probe.rows) * static_cast<double>(build.rows));

	s.decision = DecompositionDecision::FALLBACK;
	if (s.build_unknown_rate > config.max_unknown_rate) {
		s.reason =
		    "build unknown rate " + FormatRate(s.build_unknown_rate) + " > " + FormatRate(config.max_unknown_rate);
	} else if (s.probe_unknown_rate > config.max_unknown_rate) {
		s.reason =
		    "probe unknown rate " + FormatRate(s.probe_unknown_rate) + " > " + FormatRate(config.max_unknown_rate);
	} else if (is_equality && is_varchar && build.inputs.size() >= DecompositionGuardConfig::MIN_INPUTS_FOR_IDENTITY &&
	           s.build_distinct_ratio >= config.max_identity_ratio) {
		s.reason =
		    "build extraction looks like an identity (distinct ratio " + FormatRate(s.build_distinct_ratio) + ")";
	} else if (is_equality && s.overlap < config.min_overlap) {
		s.reason = "overlap " + FormatRate(s.overlap) + " < " + FormatRate(config.min_overlap);
	} else if (s.selectivity > config.max_selectivity) {
		s.reason = "selectivity " + FormatRate(s.selectivity) + " > " + FormatRate(config.max_selectivity);
	} else {
		s.decision = DecompositionDecision::DECOMPOSED;
		s.reason = "checks passed";
	}
	return s;
}

string DecompositionSignals::ToString() const {
	string result = decision == DecompositionDecision::FALLBACK ? "fallback" : "decomposed";
	result += " (" + reason + "); unknown build " + FormatRate(build_unknown_rate) + " / probe " +
	          FormatRate(probe_unknown_rate);
	if (overlap >= 0) {
		result += ", overlap " + FormatRate(overlap);
	}
	if (selectivity >= 0) {
		result += ", selectivity " + StringUtil::Format("%.4f", selectivity);
	}
	result += ", build " + std::to_string(build_rows) + " rows, probe sample " + std::to_string(probe_rows) + " rows";
	return result;
}

//===--------------------------------------------------------------------===//
// Shared state
//===--------------------------------------------------------------------===//
DecompositionGuardState::DecompositionGuardState(DecompositionGuardConfig config_p, ExpressionType comparison_p,
                                                 bool build_is_task1_p, bool is_varchar_p)
    : config(config_p), comparison(comparison_p), build_is_task1(build_is_task1_p), is_varchar(is_varchar_p),
      decision(static_cast<uint8_t>(DecompositionDecision::UNDECIDED)) {
}

void DecompositionGuardState::RecordExtraction(DecompositionGuardRole role, Vector &input, Vector &output,
                                               idx_t count) {
	if (role == DecompositionGuardRole::NONE) {
		return;
	}
	if (role == DecompositionGuardRole::PROBE_EXTRACTION && GetDecision() != DecompositionDecision::UNDECIDED) {
		return;
	}
	lock_guard<mutex> guard(lock);
	auto &stats = role == DecompositionGuardRole::BUILD_EXTRACTION ? build : probe;
	for (idx_t i = 0; i < count; i++) {
		stats.Add(input.GetValue(i), output.GetValue(i));
	}
}

bool DecompositionGuardState::ProbeSampleReady() {
	lock_guard<mutex> guard(lock);
	return probe.inputs.size() >= config.sample_size;
}

DecompositionDecision DecompositionGuardState::Decide() {
	lock_guard<mutex> guard(lock);
	auto current = GetDecision();
	if (current != DecompositionDecision::UNDECIDED) {
		return current;
	}
	signals = ComputeDecompositionSignals(build, probe, comparison, build_is_task1, is_varchar, config);
	decision.store(static_cast<uint8_t>(signals.decision));
	std::cout << "\n[SplitPredictJoin] Runtime check: " << signals.ToString() << "\n" << std::endl;
	return signals.decision;
}

string DecompositionGuardState::DecisionString() {
	lock_guard<mutex> guard(lock);
	if (GetDecision() == DecompositionDecision::UNDECIDED) {
		return string();
	}
	return signals.ToString();
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
static unique_ptr<Expression> MakeConjunction(vector<unique_ptr<Expression>> expressions) {
	D_ASSERT(!expressions.empty());
	if (expressions.size() == 1) {
		return std::move(expressions[0]);
	}
	auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
	for (auto &expr : expressions) {
		conjunction->children.push_back(std::move(expr));
	}
	return std::move(conjunction);
}

PhysicalDecompositionGuard::PhysicalDecompositionGuard(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> other_predicates,
                                                       unique_ptr<Expression> condition,
                                                       unique_ptr<Expression> fallback,
                                                       shared_ptr<DecompositionGuardState> guard_p,
                                                       idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality),
      guard(std::move(guard_p)) {
	vector<unique_ptr<Expression>> decomposed;
	vector<unique_ptr<Expression>> pairwise;
	for (auto &expr : other_predicates) {
		decomposed.push_back(expr->Copy());
		pairwise.push_back(std::move(expr));
	}
	decomposed.push_back(std::move(condition));
	pairwise.push_back(std::move(fallback));
	decomposed_predicate = MakeConjunction(std::move(decomposed));
	fallback_predicate = MakeConjunction(std::move(pairwise));
	condition_name = decomposed_predicate->GetName();
	fallback_name = fallback_predicate->GetName();
}

string PhysicalDecompositionGuard::GetName() const {
	return "LLM_JOIN_GUARD";
}

class DecompositionGuardOperatorState : public OperatorState {
public:
	DecompositionGuardOperatorState(ExecutionContext &context, Expression &decomposed, Expression &fallback)
	    : decomposed_executor(context.client, decomposed), fallback_executor(context.client, fallback,
	                                                                         [&](idx_t c, idx_t i, idx_t o, idx_t t) {
		                                                                         this->counters.llm_calls += c;
		                                                                         this->counters.inputs_used += i;
		                                                                         this->counters.outputs_used += o;
		                                                                         this->counters.tokens_used += t;
	                                                                         }),
	      sel(STANDARD_VECTOR_SIZE) {
	}

	ExpressionExecutor decomposed_executor;
	ExpressionExecutor fallback_executor;
	SelectionVector sel;
	//! Chunks received before the decision
	vector<unique_ptr<DataChunk>> buffer;
	idx_t flush_position = 0;
	bool flushing = false;

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

unique_ptr<OperatorState> PhysicalDecompositionGuard::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<DecompositionGuardOperatorState>(context, *decomposed_predicate, *fallback_predicate);
}

static void FilterChunk(DecompositionGuardOperatorState &state, DecompositionDecision decision, DataChunk &input,
                        DataChunk &chunk) {
	auto &executor =
	    decision == DecompositionDecision::FALLBACK ? state.fallback_executor : state.decomposed_executor;
	const idx_t result_count = executor.SelectExpression(input, state.sel);
	if (result_count == input.size()) {
		chunk.Reference(input);
	} else {
		chunk.Slice(input, state.sel, result_count);
	}
}

static void BufferChunk(ExecutionContext &context, DecompositionGuardOperatorState &state, DataChunk &input) {
	auto copy = make_uniq<DataChunk>();
	copy->Initialize(Allocator::Get(context.client), input.GetTypes());
	input.Copy(*copy);
	state.buffer.push_back(std::move(copy));
}

OperatorResultType PhysicalDecompositionGuard::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<DecompositionGuardOperatorState>();
	auto decision = guard->GetDecision();

	if (decision != DecompositionDecision::UNDECIDED && state.buffer.empty()) {
		FilterChunk(state, decision, input, chunk);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	if (!state.flushing) {
		// Undecided, or decided by another thread while this thread still holds buffered chunks: keep the order.
		BufferChunk(context, state, input);
		if (decision == DecompositionDecision::UNDECIDED) {
			if (!guard->ProbeSampleReady() && state.buffer.size() < MAX_BUFFERED_CHUNKS) {
				return OperatorResultType::NEED_MORE_INPUT;
			}
			decision = guard->Decide();
		}
		state.flushing = true;
		state.flush_position = 0;
	}

	// Emit the buffered chunks one at a time (the current input is the last of them).
	FilterChunk(state, decision, *state.buffer[state.flush_position++], chunk);
	if (state.flush_position < state.buffer.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	state.buffer.clear();
	state.flushing = false;
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalizeResultType PhysicalDecompositionGuard::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                    GlobalOperatorState &gstate,
                                                                    OperatorState &state_p) const {
	auto &state = state_p.Cast<DecompositionGuardOperatorState>();
	if (state.buffer.empty()) {
		return OperatorFinalizeResultType::FINISHED;
	}
	// The input ended before the probe sample was large enough: decide with what was seen.
	auto decision = guard->Decide();
	if (!state.flushing) {
		state.flushing = true;
		state.flush_position = 0;
	}
	FilterChunk(state, decision, *state.buffer[state.flush_position++], chunk);
	if (state.flush_position < state.buffer.size()) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}
	state.buffer.clear();
	state.flushing = false;
	return OperatorFinalizeResultType::FINISHED;
}

InsertionOrderPreservingMap<string> PhysicalDecompositionGuard::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["__expression__"] = condition_name;
	result["Fallback"] = fallback_name;
	auto &config = guard->config;
	result["Check"] = "sample " + std::to_string(config.sample_size) + ", max unknown " +
	                  FormatRate(config.max_unknown_rate) + ", min overlap " + FormatRate(config.min_overlap) +
	                  ", max selectivity " + FormatRate(config.max_selectivity);
	auto decision = guard->DecisionString();
	if (!decision.empty()) {
		result["Decision"] = decision;
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
