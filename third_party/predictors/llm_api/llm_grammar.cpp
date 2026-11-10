#include "llm_api.hpp"
#include "llm_common.hpp"

#include <sstream>
#include <string>

namespace duckdb {

namespace {

string type_to_string(const LogicalTypeId &type) {
	switch (type) {
	case LogicalTypeId::VARCHAR:
		return "string";
	case LogicalTypeId::INTEGER:
		return "integer";
	case LogicalTypeId::DOUBLE:
		return "number";
	case LogicalTypeId::BOOLEAN:
		return "boolean";
	default:
		throw InternalException("Unsupported result type");
	}
}

} // namespace

void LlmApiPredictor::GenerateGrammar() {
	std::vector<std::pair<std::string, LogicalTypeId>> attrs {};

	prompt_util.process_prompt_and_extract_types(attrs, this->prompt);

	std::stringstream ss;
	ss << "{";
#if IS_SCHEMA
	ss << "\"type\": \"object\",\n";
	ss << "\"properties\": {\n";
	bool is_first = true;
	for (const auto &[attribute, type] : attrs) {
		if (!is_first) {
			ss << ",\n";
		}
		ss << "\"" << attribute << "\": { \"type\": \"" << type_to_string(type) << "\" }";
		is_first = false;
	}
	ss << "\n},\n";
	ss << "\"required\": [";
	is_first = true;
	for (const auto &[attribute, type] : attrs) {
		if (!is_first) {
			ss << ", ";
		}
		ss << "\"" << attribute << "\"";
		is_first = false;
	}
	ss << "],\n";
	ss << "\"additionalProperties\": false\n";
#else
	bool first = true;
	for (const auto &[attribute, type] : attrs) {
		if (!first) {
			ss << R"(, )";
		} else {
			first = false;
		}
		ss << "\"<" << type_to_string(type) << ">\"";
		ss << R"(")" << attribute << R"(" : )";
	}
#endif
	ss << "}";
	this->grammar = ss.str();

	LLM_LOG("Prompt: " + this->prompt + "\n");
	LLM_LOG("Grammar:\n------------------\n" + this->grammar + "\n------------------\n");
}

std::string LlmApiPredictor::GenerateSystemMessage(const bool is_array) const {
	const std::string suffix =
	    R"(. Do not include any extra text, explanations, language specifier, produce {<key>: <single value>} for JSON objects. The JSON must be parsable by a standard parser.)";
	if (is_array) {
		return R"(You are a helpful assistant. Always respond **only** with valid single JSON array where each object is in format )" +
		       this->grammar + suffix;
	}
	return R"(You are a helpful assistant. Always respond **only** with valid JSON object (i.e. not an array) in format )" +
	       this->grammar + suffix;
}

nlohmann::json LlmApiPredictor::BuildSingleResponseFormat() const {
#if IS_SCHEMA
	std::stringstream sch;
	sch << "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"json_response\",\"strict\":true,";
	sch << "\"schema\":" << this->grammar << "}}";
	return PromptUtil::parse_json(sch.str());
#else
	return {};
#endif
}

// Keep this schema inside the subset every OpenAI-compatible provider accepts: no non-standard keywords
// (e.g. a nested "strict") and no minItems/maxItems > 1 -- Anthropic rejects both with HTTP 400, which
// silently turned every batch into per-row calls. The expected row count is stated in the user prompt
// and checked when the response is parsed.
nlohmann::json LlmApiPredictor::BuildArrayResponseFormat() const {
#if IS_SCHEMA
	std::stringstream sch;
	sch << "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"json_response\",\"strict\":true,";
	sch << "\"schema\":{\"type\":\"object\",\"additionalProperties\":false,\"required\":[\"output_array\"],";
	sch << "\"properties\":{\"output_array\":{\"type\":\"array\"";
	sch << ",\"items\":" << this->grammar << "}}}}}";
	return PromptUtil::parse_json(sch.str());
#else
	return {};
#endif
}

} // namespace duckdb