//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/filter/decomposition_guard.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/join_decomposition_mode.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/unordered_set.hpp"

namespace duckdb {

class Vector;

//! Thresholds of the runtime check of a decomposed LLM join (T2).
struct DecompositionGuardConfig {
	//! Distinct probe-side inputs to sample before deciding
	idx_t sample_size = 64;
	//! Fall back if either side's unknown (NULL / empty / "unknown" / "n/a") rate exceeds this
	double max_unknown_rate = 0.5;
	//! (= only) fall back if fewer than this share of sampled probe rows find their value on the build side
	double min_overlap = 0.2;
	//! Fall back if the estimated selectivity of the decomposed predicate exceeds this
	double max_selectivity = 0.5;
	//! (= on VARCHAR only) fall back if distinct outputs / distinct inputs on the build side reaches this, i.e. the
	//! extraction acts as an identity, given at least MIN_INPUTS_FOR_IDENTITY distinct build inputs
	double max_identity_ratio = 0.95;
	static constexpr idx_t MIN_INPUTS_FOR_IDENTITY = 20;
};

//! Statistics of one extraction's (input, output) pairs.
struct DecompositionSideStats {
	idx_t rows = 0;
	idx_t unknown = 0;
	unordered_set<string> inputs;
	//! Distinct (non-unknown) outputs, keyed by their string form, with one representative value and a count
	map<string, std::pair<Value, idx_t>> outputs;

	void Add(const Value &input, const Value &output);
	double UnknownRate() const;
	//! Distinct outputs / distinct inputs (0 if no inputs)
	double DistinctRatio() const;
	static bool IsUnknown(const Value &output);
};

enum class DecompositionDecision : uint8_t { UNDECIDED, DECOMPOSED, FALLBACK };

//! The signals computed at decision time, and the decision they lead to.
struct DecompositionSignals {
	double build_unknown_rate = 0;
	double probe_unknown_rate = 0;
	double build_distinct_ratio = 0;
	idx_t build_rows = 0;
	idx_t probe_rows = 0;
	//! Share of sampled probe rows whose output appears on the build side; negative if not applicable (θ is not =)
	double overlap = -1;
	//! Estimated selectivity of θ over (probe sample x build side); negative if there was nothing to estimate
	double selectivity = -1;
	DecompositionDecision decision = DecompositionDecision::DECOMPOSED;
	//! Why the decision was made (the first failing check, or "checks passed")
	string reason;

	string ToString() const;
};

//! Computes the signals from the build side's full statistics and the probe side's sample, and decides.
//! `comparison` is θ written as `task1 <op> task2`; `build_is_task1` says which side task1 extracts.
DecompositionSignals ComputeDecompositionSignals(const DecompositionSideStats &build,
                                                 const DecompositionSideStats &probe, ExpressionType comparison,
                                                 bool build_is_task1, bool is_varchar,
                                                 const DecompositionGuardConfig &config);

//! State shared by the two extraction PREDICTs of a decomposed join and the guard above the join.
//! The build-side extraction completes before the probe side starts (the join materializes its build side),
//! so by the time the guard sees its first chunk, the build statistics are complete and the probe statistics
//! contain at least the first probe chunk.
class DecompositionGuardState {
public:
	DecompositionGuardState(DecompositionGuardConfig config, ExpressionType comparison, bool build_is_task1,
	                        bool is_varchar);

	const DecompositionGuardConfig config;
	const ExpressionType comparison;
	const bool build_is_task1;
	const bool is_varchar;

	//! Records the outputs of an extraction (called by the extraction PREDICT after each chunk).
	void RecordExtraction(DecompositionGuardRole role, Vector &input, Vector &output, idx_t count);
	//! Whether enough of the probe side has been seen to decide.
	bool ProbeSampleReady();
	//! Decides once (later calls return the first decision).
	DecompositionDecision Decide();

	DecompositionDecision GetDecision() const {
		return static_cast<DecompositionDecision>(decision.load());
	}
	bool IsFallback() const {
		return GetDecision() == DecompositionDecision::FALLBACK;
	}
	//! Human-readable decision, signals and cost of the check; empty if undecided.
	string DecisionString();

private:
	mutex lock;
	atomic<uint8_t> decision;
	DecompositionSideStats build;
	DecompositionSideStats probe;
	DecompositionSignals signals;
};

} // namespace duckdb
