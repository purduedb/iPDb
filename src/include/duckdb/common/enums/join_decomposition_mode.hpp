//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/enums/join_decomposition_mode.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

//! How the optimizer treats an LLM join predicate that could be decomposed into per-side extractions
//! (see SplitPredictJoinToExtractPredicate).
enum class JoinDecompositionMode : uint8_t {
	//! Decompose only when the estimated cost of the decomposed plan is low enough (cost-based)
	AUTO,
	//! Always decompose (when the predicate qualifies), also under a LIMIT
	ALWAYS,
	//! Never decompose; keep pairwise inference
	NEVER
};

inline string JoinDecompositionModeToString(JoinDecompositionMode mode) {
	switch (mode) {
	case JoinDecompositionMode::AUTO:
		return "auto";
	case JoinDecompositionMode::ALWAYS:
		return "always";
	case JoinDecompositionMode::NEVER:
		return "never";
	default:
		throw InternalException("Unknown JoinDecompositionMode");
	}
}

inline JoinDecompositionMode JoinDecompositionModeFromString(const string &str) {
	auto lower = StringUtil::Lower(str);
	if (lower == "auto") {
		return JoinDecompositionMode::AUTO;
	}
	if (lower == "always" || lower == "on" || lower == "true") {
		return JoinDecompositionMode::ALWAYS;
	}
	if (lower == "never" || lower == "off" || lower == "false") {
		return JoinDecompositionMode::NEVER;
	}
	throw InvalidInputException("Unrecognized join decomposition mode \"%s\" (expected auto, always or never)", str);
}

} // namespace duckdb
