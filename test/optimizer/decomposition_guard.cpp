#include "catch.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/operator/filter/decomposition_guard.hpp"

using namespace duckdb;

// Runtime check of a decomposed LLM join (T2): the decision function, without any LLM.

static DecompositionSideStats MakeStats(const vector<std::pair<string, string>> &rows) {
	DecompositionSideStats stats;
	for (auto &row : rows) {
		stats.Add(Value(row.first), row.second == "<NULL>" ? Value(LogicalType::VARCHAR) : Value(row.second));
	}
	return stats;
}

// n rows "<prefix> i" whose extracted value is format(i % groups)
static DecompositionSideStats MakeSide(const string &prefix, idx_t n, idx_t groups, const string &value_prefix) {
	vector<std::pair<string, string>> rows;
	for (idx_t i = 0; i < n; i++) {
		rows.emplace_back(prefix + " " + std::to_string(i), value_prefix + std::to_string(i % groups));
	}
	return MakeStats(rows);
}

static DecompositionSignals Decide(const DecompositionSideStats &build, const DecompositionSideStats &probe,
                                   ExpressionType cmp = ExpressionType::COMPARE_EQUAL, bool is_varchar = true) {
	DecompositionGuardConfig config;
	return ComputeDecompositionSignals(build, probe, cmp, true, is_varchar, config);
}

TEST_CASE("Decomposition guard: unknown outputs", "[predict]") {
	REQUIRE(DecompositionSideStats::IsUnknown(Value(LogicalType::VARCHAR)));
	REQUIRE(DecompositionSideStats::IsUnknown(Value("  Unknown ")));
	REQUIRE(DecompositionSideStats::IsUnknown(Value("N/A")));
	REQUIRE(DecompositionSideStats::IsUnknown(Value("")));
	REQUIRE(!DecompositionSideStats::IsUnknown(Value("LGA1700")));
	REQUIRE(!DecompositionSideStats::IsUnknown(Value::INTEGER(0)));

	auto stats = MakeStats({{"a", "x"}, {"b", "unknown"}, {"c", "<NULL>"}, {"a", "x"}});
	REQUIRE(stats.rows == 4);
	REQUIRE(stats.unknown == 2);
	REQUIRE(stats.inputs.size() == 3);
	REQUIRE(stats.outputs.size() == 1);
	REQUIRE(stats.UnknownRate() == Approx(0.5));
}

TEST_CASE("Decomposition guard: a good equality decomposition passes", "[predict]") {
	auto build = MakeSide("board", 256, 8, "S");
	auto probe = MakeSide("cpu", 64, 8, "S");
	auto s = Decide(build, probe);
	REQUIRE(s.decision == DecompositionDecision::DECOMPOSED);
	REQUIRE(s.overlap == Approx(1.0));
	REQUIRE(s.selectivity == Approx(1.0 / 8));
}

TEST_CASE("Decomposition guard: disjoint vocabularies fall back on overlap", "[predict]") {
	auto build = MakeSide("board", 256, 8, "lga ");
	auto probe = MakeSide("cpu", 64, 8, "LGA-");
	auto s = Decide(build, probe);
	REQUIRE(s.decision == DecompositionDecision::FALLBACK);
	REQUIRE(s.overlap == Approx(0.0));
	REQUIRE(s.reason.find("overlap") != string::npos);
}

TEST_CASE("Decomposition guard: unknown outputs fall back", "[predict]") {
	vector<std::pair<string, string>> unknown_rows;
	for (idx_t i = 0; i < 100; i++) {
		unknown_rows.emplace_back("board " + std::to_string(i), i % 2 == 0 ? "unknown" : "<NULL>");
	}
	auto build = MakeStats(unknown_rows);
	REQUIRE(build.UnknownRate() == Approx(1.0));
	auto probe = MakeSide("cpu", 64, 8, "S");
	auto s = Decide(build, probe);
	REQUIRE(s.decision == DecompositionDecision::FALLBACK);
	REQUIRE(s.reason.find("build unknown rate") != string::npos);

	vector<std::pair<string, string>> na_rows;
	for (idx_t i = 0; i < 64; i++) {
		na_rows.emplace_back("cpu " + std::to_string(i), i < 48 ? "N/A" : "S1");
	}
	auto probe_unknown = MakeStats(na_rows);
	s = Decide(MakeSide("board", 256, 8, "S"), probe_unknown);
	REQUIRE(s.decision == DecompositionDecision::FALLBACK);
	REQUIRE(s.reason.find("probe unknown rate") != string::npos);
}

TEST_CASE("Decomposition guard: an identity extraction falls back", "[predict]") {
	// Every build input gets its own output: the extraction copies the input rather than abstracting it.
	auto build = MakeSide("board", 100, 100, "board ");
	auto probe = MakeSide("cpu", 64, 100, "board ");
	auto s = Decide(build, probe);
	REQUIRE(s.decision == DecompositionDecision::FALLBACK);
	REQUIRE(s.reason.find("identity") != string::npos);
	// The rule only applies to VARCHAR equality.
	s = Decide(build, probe, ExpressionType::COMPARE_EQUAL, false);
	REQUIRE(s.reason.find("identity") == string::npos);
}

TEST_CASE("Decomposition guard: too coarse an extraction falls back on selectivity", "[predict]") {
	// Everything maps to one value: θ matches every pair.
	auto build = MakeSide("board", 256, 1, "S");
	auto probe = MakeSide("cpu", 64, 1, "S");
	auto s = Decide(build, probe);
	REQUIRE(s.decision == DecompositionDecision::FALLBACK);
	REQUIRE(s.selectivity == Approx(1.0));
	REQUIRE(s.reason.find("selectivity") != string::npos);
}

TEST_CASE("Decomposition guard: inequality selectivity respects the orientation of θ", "[predict]") {
	// task1 = build value, task2 = probe value; θ = task1 > task2
	DecompositionSideStats build, probe;
	for (int i = 0; i < 10; i++) {
		build.Add(Value("b" + std::to_string(i)), Value::DOUBLE(i));
		probe.Add(Value("p" + std::to_string(i)), Value::DOUBLE(i));
	}
	DecompositionGuardConfig config;
	config.max_selectivity = 1.0;
	// pairs with build > probe: 45 of 100
	auto s = ComputeDecompositionSignals(build, probe, ExpressionType::COMPARE_GREATERTHAN, true, false, config);
	REQUIRE(s.selectivity == Approx(0.45));
	REQUIRE(s.overlap < 0);
	REQUIRE(s.decision == DecompositionDecision::DECOMPOSED);
	// with task1 on the probe side, the same θ counts probe > build: also 45, but check a skewed case
	DecompositionSideStats high_probe;
	for (int i = 0; i < 10; i++) {
		high_probe.Add(Value("p" + std::to_string(i)), Value::DOUBLE(100 + i));
	}
	s = ComputeDecompositionSignals(build, high_probe, ExpressionType::COMPARE_GREATERTHAN, true, false, config);
	REQUIRE(s.selectivity == Approx(0.0));
	s = ComputeDecompositionSignals(build, high_probe, ExpressionType::COMPARE_GREATERTHAN, false, false, config);
	REQUIRE(s.selectivity == Approx(1.0));
}

TEST_CASE("Decomposition guard: shared state decides once", "[predict]") {
	DecompositionGuardConfig config;
	config.sample_size = 4;
	DecompositionGuardState state(config, ExpressionType::COMPARE_EQUAL, true, true);
	REQUIRE(state.GetDecision() == DecompositionDecision::UNDECIDED);
	REQUIRE(state.DecisionString().empty());

	Vector build_in(LogicalType::VARCHAR, 8), build_out(LogicalType::VARCHAR, 8);
	Vector probe_in(LogicalType::VARCHAR, 8), probe_out(LogicalType::VARCHAR, 8);
	for (idx_t i = 0; i < 8; i++) {
		build_in.SetValue(i, Value("board " + std::to_string(i)));
		build_out.SetValue(i, Value("lga " + std::to_string(i % 4)));
		probe_in.SetValue(i, Value("cpu " + std::to_string(i)));
		probe_out.SetValue(i, Value("LGA-" + std::to_string(i % 4)));
	}
	state.RecordExtraction(DecompositionGuardRole::BUILD_EXTRACTION, build_in, build_out, 8);
	state.RecordExtraction(DecompositionGuardRole::PROBE_EXTRACTION, probe_in, probe_out, 2);
	REQUIRE(!state.ProbeSampleReady());
	state.RecordExtraction(DecompositionGuardRole::PROBE_EXTRACTION, probe_in, probe_out, 8);
	REQUIRE(state.ProbeSampleReady());

	REQUIRE(state.Decide() == DecompositionDecision::FALLBACK);
	REQUIRE(state.IsFallback());
	REQUIRE(state.DecisionString().find("fallback") != string::npos);
	// later records and decisions do not change it
	state.RecordExtraction(DecompositionGuardRole::PROBE_EXTRACTION, build_in, build_out, 8);
	REQUIRE(state.Decide() == DecompositionDecision::FALLBACK);
}
