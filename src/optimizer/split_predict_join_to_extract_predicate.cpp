#include "duckdb/optimizer/split_predict_join_to_extract_predicate.hpp"

#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_predict_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/execution/operator/filter/decomposition_guard.hpp"

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
#include "openai.hpp"
#include "duckdb/common/prompt.hpp"
#include "duckdb/common/string_util.hpp"

#include <regex>
#endif

namespace duckdb {

// ---------------------------------------------------------------------------
// Local helper: check whether a join operator type is one we handle.
// ---------------------------------------------------------------------------
static bool IsSupportedJoin(LogicalOperatorType type) {
	return type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
		type == LogicalOperatorType::LOGICAL_JOIN ||
		type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		type == LogicalOperatorType::LOGICAL_ANY_JOIN;
}

// ---------------------------------------------------------------------------
// Local helper: from a filter's child, skip a chain of projections (e.g. a
// subquery's select list, or debug-build verification projections) and return
// the join below them, or nullptr. `chain` receives the projections, top first.
// ---------------------------------------------------------------------------
static LogicalOperator *FindJoinBelow(LogicalOperator &filter_child, vector<LogicalProjection *> *chain = nullptr) {
	LogicalOperator *current = &filter_child;
	while (current->type == LogicalOperatorType::LOGICAL_PROJECTION && current->children.size() == 1) {
		if (chain) {
			chain->push_back(&current->Cast<LogicalProjection>());
		}
		current = current->children[0].get();
	}
	return IsSupportedJoin(current->type) ? current : nullptr;
}

// ---------------------------------------------------------------------------
// Local helper: depth-first search for the first BoundPredictExpression in an
// expression tree.  Returns nullptr when none is found.
// ---------------------------------------------------------------------------
static BoundPredictExpression *FindFirstPredict(Expression &expr) {
	if (expr.expression_class == ExpressionClass::PREDICT) {
		return &expr.Cast<BoundPredictExpression>();
	}
	BoundPredictExpression *result = nullptr;
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		if (!result) {
			result = FindFirstPredict(child);
		}
	});
	return result;
}

// ---------------------------------------------------------------------------
// Local helper: collect all BoundColumnRefExpression bindings in an expression
// tree into `out`.
// ---------------------------------------------------------------------------
static void CollectColumnBindings(Expression &expr, vector<ColumnBinding> &out) {
	if (expr.expression_class == ExpressionClass::BOUND_COLUMN_REF) {
		out.push_back(expr.Cast<BoundColumnRefExpression>().binding);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		CollectColumnBindings(child, out);
	});
}

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)

// ---------------------------------------------------------------------------
// Local helper: recursively collect LogicalGet nodes and build a schema string
// like "CPUs(PRODUCT_ID, PRODUCT_NAME), Motherboards(PRODUCT_ID, PRODUCT_NAME)".
// ---------------------------------------------------------------------------
static void AppendTableSchemas(LogicalOperator &op, std::string &out) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (!out.empty()) {
			out += ", ";
		}
		out += get.GetName() + "(";
		for (idx_t i = 0; i < get.names.size(); i++) {
			if (i > 0) {
				out += ", ";
			}
			out += get.names[i];
		}
		out += ")";
	}
	for (auto &child : op.children) {
		AppendTableSchemas(*child, out);
	}
}

// ---------------------------------------------------------------------------
// Holds the structured response from the decomposition LLM call.
// ---------------------------------------------------------------------------
struct DecomposedPredict {
	std::string task1;
	std::string task1_output_column;
	LogicalType task1_output_type = LogicalType::VARCHAR;
	std::string task2;
	std::string task2_output_column;
	LogicalType task2_output_type = LogicalType::VARCHAR;
	std::string predicate;
};

// ---------------------------------------------------------------------------
// Local helper: collect the `{{input}}` placeholders of a prompt, in order.
// ---------------------------------------------------------------------------
static vector<std::string> ExtractPromptInputs(const std::string &prompt) {
	static const std::regex in_re(R"(\{\{([A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?)\}\})");
	vector<std::string> inputs;
	for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), in_re); it != std::sregex_iterator(); ++it) {
		inputs.push_back((*it)[1].str());
	}
	return inputs;
}

// ---------------------------------------------------------------------------
// Local helper: validate one decomposed task prompt against iPDb PREDICT
// syntax. It must reference exactly the expected `{{input}}` placeholder and
// declare exactly one `{<output_column> <TYPE>}` output whose name matches
// `expected_output`. On success the declared type is written to `out_type`.
// Returns an empty string on success, otherwise a description of the problem.
// ---------------------------------------------------------------------------
static std::string ValidateTaskPrompt(const std::string &prompt, const std::string &expected_input,
                                      const std::string &expected_output, LogicalType &out_type) {
	auto inputs = ExtractPromptInputs(prompt);
	if (inputs.size() != 1 || !StringUtil::CIEquals(inputs[0], expected_input)) {
		return "must reference exactly one input placeholder {{" + expected_input + "}}";
	}
	// Output declaration: `{name TYPE}` (single braces, not part of `{{...}}`).
	static const std::regex out_re(R"((^|[^{])\{\s*([A-Za-z_][A-Za-z0-9_]*)\s+(INTEGER|VARCHAR|BOOLEAN|BOOL|DOUBLE)\s*\}(?!\}))",
	                               std::regex_constants::icase);
	idx_t n_outputs = 0;
	std::string out_name;
	std::string out_type_str;
	for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), out_re); it != std::sregex_iterator(); ++it) {
		out_name = (*it)[2].str();
		out_type_str = StringUtil::Upper((*it)[3].str());
		n_outputs++;
	}
	if (n_outputs != 1) {
		return "must declare exactly one output as {<output_column> <TYPE>}";
	}
	if (out_name != expected_output) {
		return "declared output column '" + out_name + "' does not match output column '" + expected_output + "'";
	}
	out_type = LogicalType(Prompt::type_to_logical_type(out_type_str));
	return "";
}

// ---------------------------------------------------------------------------
// Local helper: rename the `{old_name TYPE}` output declaration of a
// (validated) task prompt to `{new_name TYPE}`, leaving `{{input}}` intact.
// ---------------------------------------------------------------------------
static void RenamePromptOutput(std::string &prompt, const std::string &old_name, const std::string &new_name) {
	static const std::regex out_re(R"((^|[^{])\{(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s+(?:INTEGER|VARCHAR|BOOLEAN|BOOL|DOUBLE)\s*)\}(?!\}))",
	                               std::regex_constants::icase);
	std::smatch m;
	auto begin = prompt.cbegin();
	while (std::regex_search(begin, prompt.cend(), m, out_re)) {
		if (m[3].str() == old_name) {
			auto pos = static_cast<size_t>(m[3].first - prompt.cbegin());
			prompt.replace(pos, old_name.size(), new_name);
			return;
		}
		begin = m[0].second;
	}
}

// ---------------------------------------------------------------------------
// Local helper: derive a snake_case suffix from an input placeholder, using
// its table qualifier (`a.review` -> `a`), or `fallback` if unqualified.
// ---------------------------------------------------------------------------
static std::string InputSuffix(const std::string &input, const std::string &fallback) {
	auto dot = input.find('.');
	if (dot == std::string::npos || dot == 0) {
		return fallback;
	}
	return StringUtil::Lower(input.substr(0, dot));
}

// ---------------------------------------------------------------------------
// Local helper: make sure the two sub-task output columns never contend for
// the same name (e.g. both tasks extracting `sentiment`). Identifiers are
// case-insensitive in DuckDB, so compare accordingly. On a clash both columns
// are suffixed with their input's table alias (sentiment_a / sentiment_b),
// falling back to positional suffixes, and the prompts are rewritten to match.
// ---------------------------------------------------------------------------
static void DisambiguateOutputColumns(DecomposedPredict &result, const std::string &task1_input,
                                      const std::string &task2_input) {
	if (!StringUtil::CIEquals(result.task1_output_column, result.task2_output_column)) {
		return;
	}
	auto s1 = InputSuffix(task1_input, "1");
	auto s2 = InputSuffix(task2_input, "2");
	if (s1 == s2) {
		s1 = "1";
		s2 = "2";
	}
	auto new1 = result.task1_output_column + "_" + s1;
	auto new2 = result.task2_output_column + "_" + s2;
	RenamePromptOutput(result.task1, result.task1_output_column, new1);
	RenamePromptOutput(result.task2, result.task2_output_column, new2);
	result.task1_output_column = new1;
	result.task2_output_column = new2;
}

// ---------------------------------------------------------------------------
// Local helper: call the LLM to decompose a cross-join PREDICT into two
// single-input extraction tasks plus a relational predicate.
// Returns false (and leaves the plan untouched) if the response does not
// conform to iPDb PREDICT syntax.
// ---------------------------------------------------------------------------
static bool CallDecompositionLLM(Optimizer &optimizer, const BoundPredictInfo &predict_info,
                                 LogicalOperator &join, const std::string &task1_input,
                                 const std::string &task2_input, DecomposedPredict &result) {
	using Json = nlohmann::json;

	auto &db = DatabaseInstance::GetDatabase(optimizer.context);
	auto api = OpenAI::createInstance(db, predict_info.base_api, predict_info.secret);

	// Build the schema string from the tables under the join.
	std::string schemas;
	AppendTableSchemas(join, schemas);

	// Fixed system message describing iPDb semantic-task (PREDICT prompt) syntax.
	std::string sys_msg =
	    "You are a database query optimizer for iPDb, a SQL engine with LLM-powered semantic predicates.\n"
	    "Your job is to decompose a semantic predicate that compares two inputs (one from each side of a join)"
	    " into two independent single-input extraction tasks plus a relational comparison, so that each"
	    " extraction runs once per row of one table instead of once per row pair.\n"
	    "\n"
	    "iPDb semantic task (prompt) syntax -- follow it EXACTLY:\n"
	    "- Inputs are referenced with double braces holding the fully qualified column name, e.g."
	    " {{t.col}}. Copy the input placeholder verbatim from the original task.\n"
	    "- The output is declared inline with single braces as {<output_column> <TYPE>}, where <TYPE> is one of"
	    " BOOL, INTEGER, DOUBLE, VARCHAR. Exactly one output declaration per task.\n"
	    "- <output_column> is a snake_case identifier ([a-z_][a-z0-9_]*) and must be identical to the"
	    " corresponding *_output_column field.\n"
	    "- Never describe the output space in prose (e.g. '(positive or negative)', 'yes/no', 'one of A, B, C')."
	    " Instead encode it in the output column and type. Prefer BOOL for binary attributes"
	    " (e.g. {is_positive BOOL}), INTEGER/DOUBLE for numeric attributes (e.g. {price DOUBLE}), and VARCHAR"
	    " only for open-ended categorical values.\n"
	    "- task1_output_column and task2_output_column MUST be different names, even when both tasks extract the"
	    " same attribute: suffix the attribute with the input's table alias (e.g. sentiment_a and sentiment_b for"
	    " {{a.review}} and {{b.review}}).\n"
	    "- Use the same <TYPE> for both tasks, and phrase both tasks identically except for the input, so the"
	    " two outputs are directly comparable.\n"
	    "\n"
	    "Predicate: one of the comparison operators =, !=, <, >, <=, >= applied as"
	    " `task1_output_column <op> task2_output_column`. Choose the operator so that the comparison is true"
	    " exactly when the original task's output would be true.\n"
	    "\n"
	    "Example 1\n"
	    "Original task: Is the product {{a.description}} more expensive than {{b.description}}? {pricier BOOL}\n"
	    "task1: Extract the price in USD of the product {{a.description}} {price_a DOUBLE}\n"
	    "task1_output_column: price_a\n"
	    "task2: Extract the price in USD of the product {{b.description}} {price_b DOUBLE}\n"
	    "task2_output_column: price_b\n"
	    "predicate: >\n"
	    "\n"
	    "Example 2\n"
	    "Original task: Do {{p.abstract}} and {{q.abstract}} belong to the {same_field BOOL} research field?\n"
	    "task1: Extract the research field of the paper {{p.abstract}} {field_p VARCHAR}\n"
	    "task1_output_column: field_p\n"
	    "task2: Extract the research field of the paper {{q.abstract}} {field_q VARCHAR}\n"
	    "task2_output_column: field_q\n"
	    "predicate: =";

	// User message: the concrete task to split.
	std::string user_msg =
	    "Original task: " + predict_info.prompt + "\n"
	    "task1 must use exactly the input {{" + task1_input + "}} and nothing else.\n"
	    "task2 must use exactly the input {{" + task2_input + "}} and nothing else.\n"
	    "Schemas: " + schemas;

	// Response schema requiring exactly the five fields; predicate is constrained
	// to a single comparison operator.
	Json response_schema;
	response_schema["type"] = "json_schema";
	response_schema["json_schema"] = {
	    {"name", "json_response"},
	    {"strict", true},
	    {"schema", {
	        {"type", "object"},
	        {"additionalProperties", false},
	        {"required", {"task1", "task1_output_column", "task2", "task2_output_column", "predicate"}},
	        {"properties", {
	            {"task1",                {{"type", "string"}}},
	            {"task1_output_column",  {{"type", "string"}}},
	            {"task2",                {{"type", "string"}}},
	            {"task2_output_column",  {{"type", "string"}}},
	            {"predicate",            {{"type", "string"}, {"enum", {"=", "!=", "<", ">", "<=", ">="}}}}
	        }}
	    }}
	};

	Json request;
	request["model"] = predict_info.model_path;
	request["messages"] = {
	    {{"role", "system"}, {"content", sys_msg}},
	    {{"role", "user"},   {"content", user_msg}}
	};
	request["response_format"] = response_schema;

	// A failed or unparsable planning call keeps the original (pairwise) join instead of failing the query.
	// DEBUG: a canned planning response (debug_join_decomposition_response) replaces the API call.
	const auto &debug_response = ClientConfig::GetConfig(optimizer.context).debug_join_decomposition_response;
	try {
		std::string content;
		if (!debug_response.empty()) {
			content = debug_response;
		} else {
			auto completion = api->post("chat/completions", request);
			if (completion.contains("error")) {
				std::cout << "\n[SplitPredictJoin] Skipping rewrite, planning call failed: "
				          << completion["error"].dump() << "\n" << std::endl;
				return false;
			}
			content = completion["choices"][0]["message"]["content"].get<std::string>();
		}
		auto parsed = Json::parse(PromptUtil::extract_json(content));
		result.task1               = parsed["task1"].get<std::string>();
		result.task1_output_column = parsed["task1_output_column"].get<std::string>();
		result.task2               = parsed["task2"].get<std::string>();
		result.task2_output_column = parsed["task2_output_column"].get<std::string>();
		result.predicate           = parsed["predicate"].get<std::string>();
	} catch (const std::exception &e) {
		std::cout << "\n[SplitPredictJoin] Skipping rewrite, unusable planning response: " << e.what() << "\n"
		          << std::endl;
		return false;
	}

	// Reject responses that do not follow iPDb PREDICT syntax rather than
	// building a plan around malformed prompts.
	auto err = ValidateTaskPrompt(result.task1, task1_input, result.task1_output_column, result.task1_output_type);
	if (err.empty()) {
		err = ValidateTaskPrompt(result.task2, task2_input, result.task2_output_column, result.task2_output_type);
		if (!err.empty()) {
			err = "task2 " + err;
		}
	} else {
		err = "task1 " + err;
	}
	if (err.empty()) {
		DisambiguateOutputColumns(result, task1_input, task2_input);
	}
	if (err.empty() && result.task1_output_type != result.task2_output_type) {
		err = "task1 and task2 must declare the same output type";
	}
	if (!err.empty()) {
		std::cout << "\n[SplitPredictJoin] Skipping rewrite, invalid decomposition: " << err << "\n"
		          << "  task1: " << result.task1 << "\n"
		          << "  task2: " << result.task2 << "\n" << std::endl;
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Local helper: infer a comparison ExpressionType from the LLM-generated
// predicate string. Checks multi-char tokens before single-char ones.
// ---------------------------------------------------------------------------
static ExpressionType ParseComparisonOp(const std::string &pred) {
	if (pred.find("!=") != std::string::npos || pred.find("<>") != std::string::npos) {
		return ExpressionType::COMPARE_NOTEQUAL;
	}
	if (pred.find("<=") != std::string::npos) {
		return ExpressionType::COMPARE_LESSTHANOREQUALTO;
	}
	if (pred.find(">=") != std::string::npos) {
		return ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	}
	if (pred.find("<") != std::string::npos) {
		return ExpressionType::COMPARE_LESSTHAN;
	}
	if (pred.find(">") != std::string::npos) {
		return ExpressionType::COMPARE_GREATERTHAN;
	}
	return ExpressionType::COMPARE_EQUAL;
}

// ---------------------------------------------------------------------------
// T1: plan-time cost comparison between pairwise inference and decomposition.
//
// Costs are counted in "items" (input values sent to the LLM, a proxy for input tokens):
//   pairwise   = P_eff                            P_eff = min(P, k / s) under LIMIT k, else P
//   decomposed = c_rewrite + d_R + d_S
// where P is the estimated number of row pairs reaching the predicate, d_R / d_S are the estimated number of
// rows on each side of the join (an upper bound on the distinct inputs each extraction sees), s is the assumed
// predicate selectivity (join_decomposition_selectivity), and c_rewrite is the planning call's cost in
// item-equivalents. PredictJoin sends roughly one value per pair when the pairs come from a cross product
// (one row of one side against a vector of the other), which is why a pair counts as one item.
// We decompose iff pairwise >= min_speedup * decomposed (join_decomposition_min_speedup); the margin accounts
// for the quality risk of the approximate rewrite.
// ---------------------------------------------------------------------------
static constexpr double DECOMPOSITION_REWRITE_COST = 10.0;

struct DecompositionEstimate {
	double pairs = 0;
	double pairs_eff = 0;
	double left = 0;
	double right = 0;
	double pairwise = 0;
	double decomposed = 0;
	bool decompose = false;
};

static double OperatorCardinality(ClientContext &context, LogicalOperator &op) {
	auto cardinality = op.has_estimated_cardinality ? op.estimated_cardinality : op.EstimateCardinality(context);
	return static_cast<double>(cardinality);
}

static DecompositionEstimate EstimateDecomposition(ClientContext &context, LogicalOperator &filter_child,
                                                   int64_t limit, double selectivity, double min_speedup) {
	DecompositionEstimate est;
	// The filter's child is the join, or a chain of projections on top of it (see CanOptimize).
	LogicalOperator *join = FindJoinBelow(filter_child);
	if (!join) {
		join = &filter_child;
	}
	if (join->children.size() == 2) {
		est.left = OperatorCardinality(context, *join->children[0]);
		est.right = OperatorCardinality(context, *join->children[1]);
	}
	if (join->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT || join->children.size() != 2) {
		est.pairs = join->children.size() == 2 ? est.left * est.right : OperatorCardinality(context, *join);
	} else {
		est.pairs = OperatorCardinality(context, *join);
	}
	est.pairs_eff = est.pairs;
	if (limit >= 0) {
		est.pairs_eff = MinValue<double>(est.pairs, static_cast<double>(limit) / selectivity);
	}
	est.pairwise = est.pairs_eff;
	est.decomposed = DECOMPOSITION_REWRITE_COST + est.left + est.right;
	est.decompose = est.pairwise >= min_speedup * est.decomposed;
	return est;
}

static string FormatCost(double value) {
	return std::to_string(static_cast<int64_t>(value + 0.5));
}

static string DecompositionNote(const DecompositionEstimate &est, double min_speedup) {
	auto speedup = std::to_string(min_speedup);
	speedup.erase(speedup.find_last_not_of('0') + 1);
	if (!speedup.empty() && speedup.back() == '.') {
		speedup.pop_back();
	}
	return string("auto, pairwise ") + FormatCost(est.pairwise) + (est.decompose ? " >= " : " < ") + speedup +
	       "x decomposed " + FormatCost(est.decomposed);
}

#endif // ENABLE_PREDICT && ENABLE_LLM_API

// ===========================================================================
// SplitPredictJoinToExtractPredicate
// ===========================================================================

SplitPredictJoinToExtractPredicate::SplitPredictJoinToExtractPredicate(Optimizer &optimizer)
    : optimizer(optimizer) {
}

// ---------------------------------------------------------------------------
bool SplitPredictJoinToExtractPredicate::CanOptimize(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_FILTER) {
		return false;
	}
	// A limit propagated into the filter (LIMIT above, pairwise inference stops early) is handled by the cost
	// decision in TransformFilterJoin.
	if (op.children.empty()) {
		return false;
	}

	auto &direct_child = *op.children[0];
	// FILTER → JOIN, or FILTER → PROJECTION → ... → PROJECTION → JOIN
	if (!FindJoinBelow(direct_child)) {
		return false;
	}

	// Get the output column bindings of the direct child (join or project).
	// When there is a PROJECT between the FILTER and the JOIN, the PREDICT's
	// column refs point to the PROJECT's output — not to the original join
	// table indices — so we must match against the direct child's bindings.
	auto child_output = direct_child.GetColumnBindings();

	auto &filter = op.Cast<LogicalFilter>();
	for (auto &expr : filter.expressions) {
		auto *predict = FindFirstPredict(*expr);
		if (!predict) {
			continue;
		}

		// Collect every BoundColumnRefExpression binding inside the PREDICT.
		vector<ColumnBinding> predict_refs;
		CollectColumnBindings(*predict, predict_refs);

		// Count how many distinct child-output columns the PREDICT references.
		idx_t matched = 0;
		for (auto &cb : child_output) {
			for (auto &ref : predict_refs) {
				if (ref == cb) {
					matched++;
					break;
				}
			}
		}
		if (matched >= 2) {
			return true;
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// Local helper: build the BoundPredictInfo of one extraction task.
// ---------------------------------------------------------------------------
static unique_ptr<BoundPredictInfo> MakeTaskInfo(const BoundPredictInfo &orig, const string &prompt,
                                                 const string &output_column, const LogicalType &output_type,
                                                 const string &input_name) {
	auto info = make_uniq<BoundPredictInfo>();
	info->model_type = orig.model_type;
	info->model_name = orig.model_name;
	info->model_path = orig.model_path;
	info->base_api = orig.base_api;
	info->secret = orig.secret;
	info->prompt = prompt;
	info->result_set_names = {output_column};
	info->result_set_types = {output_type};
	info->input_set_names = {input_name};
	info->options = orig.options;
	return info;
}

// ---------------------------------------------------------------------------
unique_ptr<LogicalOperator>
SplitPredictJoinToExtractPredicate::TransformFilterJoin(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_FILTER);

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
	auto &filter = op->Cast<LogicalFilter>();
	// The filter's child is the join, or a chain of projections on top of it (see CanOptimize).
	auto &direct_child = *op->children[0];
	vector<LogicalProjection *> chain; // top first
	auto *join_ptr = FindJoinBelow(direct_child, &chain);
	if (!join_ptr || join_ptr->children.size() != 2) {
		return op;
	}
	auto &join = *join_ptr;

	// ------------------------------------------------------------------
	// 1-2. Find the qualifying PREDICT: a top-level filter conjunct (only
	//      a conjunct can be replaced by a comparison; the others are kept)
	//      with exactly two inputs, which come from the two join sides.
	// ------------------------------------------------------------------
	ColumnBinding input_binding[2];
	idx_t input_side[2];
	vector<std::string> orig_inputs;
	// Maps one PREDICT input to the join side (child) that produces it; false if it can't.
	auto map_input = [&](Expression &child, ColumnBinding &binding_out, idx_t &side_out) {
		if (child.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		auto binding = child.Cast<BoundColumnRefExpression>().binding;
		// Follow the binding through the projections between the filter and the join.
		for (auto *proj : chain) {
			if (binding.table_index != proj->table_index || binding.column_index >= proj->expressions.size()) {
				return false;
			}
			auto &projected = *proj->expressions[binding.column_index];
			if (projected.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			binding = projected.Cast<BoundColumnRefExpression>().binding;
		}
		for (idx_t side = 0; side < 2; side++) {
			for (auto &cb : join.children[side]->GetColumnBindings()) {
				if (cb == binding) {
					binding_out = binding;
					side_out = side;
					return true;
				}
			}
		}
		return false;
	};
	idx_t predict_idx = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < filter.expressions.size() && predict_idx == DConstants::INVALID_INDEX; i++) {
		auto &expr = *filter.expressions[i];
		if (expr.GetExpressionClass() != ExpressionClass::PREDICT) {
			continue;
		}
		auto &candidate = expr.Cast<BoundPredictExpression>();
		if (!candidate.bound_predict || candidate.children.size() < 2) {
			continue;
		}
		// PREDICT children are created in placeholder order, so the i-th `{{...}}`
		// in the prompt corresponds to children[i].
		auto inputs = ExtractPromptInputs(candidate.bound_predict->prompt);
		if (inputs.size() != 2) {
			continue;
		}
		if (!map_input(*candidate.children[0], input_binding[0], input_side[0]) ||
		    !map_input(*candidate.children[1], input_binding[1], input_side[1]) || input_side[0] == input_side[1]) {
			continue;
		}
		predict_idx = i;
		orig_inputs = std::move(inputs);
	}
	if (predict_idx == DConstants::INVALID_INDEX) {
		return op;
	}
	auto &orig_info = *filter.expressions[predict_idx]->Cast<BoundPredictExpression>().bound_predict;

	// ------------------------------------------------------------------
	// 3. Decide whether to decompose (T1 + T3), before spending the
	//    planning call. The session setting can be overridden per model
	//    with OPTIONS {"join_decomposition": 'auto' | 'always' | 'never'}.
	// ------------------------------------------------------------------
	const auto &config = ClientConfig::GetConfig(optimizer.context);
	auto mode = config.join_decomposition;
	auto mode_option = orig_info.options.find("join_decomposition");
	if (mode_option != orig_info.options.end()) {
		mode = JoinDecompositionModeFromString(mode_option->second.ToString());
	}
	string decision_note;
	switch (mode) {
	case JoinDecompositionMode::NEVER:
		orig_info.decomposition_note = "never";
		return op;
	case JoinDecompositionMode::ALWAYS:
		decision_note = "always";
		break;
	case JoinDecompositionMode::AUTO: {
		auto est = EstimateDecomposition(optimizer.context, direct_child, filter.limit,
		                                 config.join_decomposition_selectivity, config.join_decomposition_min_speedup);
		decision_note = DecompositionNote(est, config.join_decomposition_min_speedup);
		if (!est.decompose) {
			orig_info.decomposition_note = decision_note;
			return op;
		}
		break;
	}
	}
	// T2: in AUTO mode the decomposition is checked at runtime and can fall back to the pairwise predicate.
	const bool guarded = mode == JoinDecompositionMode::AUTO && config.join_decomposition_check;

	DecomposedPredict decomposed;
	if (!CallDecompositionLLM(optimizer, orig_info, direct_child, orig_inputs[0], orig_inputs[1], decomposed)) {
		// The planning call failed or returned an invalid decomposition; keep the pairwise join.
		orig_info.decomposition_note = decision_note + ", no usable rewrite";
		return op;
	}
	if (guarded) {
		decision_note += ", checked at runtime";
	}
	const auto comparison = ParseComparisonOp(decomposed.predicate);
	const auto output_type = decomposed.task1_output_type;
	const std::string orig_prompt = orig_info.prompt;
	const std::string orig_model = orig_info.model_path;

	shared_ptr<DecompositionGuardState> guard;
	if (guarded) {
		DecompositionGuardConfig guard_config;
		guard_config.sample_size = config.join_decomposition_sample;
		guard_config.max_unknown_rate = config.join_decomposition_max_unknown;
		guard_config.min_overlap = config.join_decomposition_min_overlap;
		guard_config.max_selectivity = config.join_decomposition_max_selectivity;
		// The join materializes children[1] (build side) before children[0] (probe side) streams through it.
		const bool build_is_task1 = input_side[0] == 1;
		guard = make_shared_ptr<DecompositionGuardState>(guard_config, comparison, build_is_task1,
		                                                 output_type.id() == LogicalTypeId::VARCHAR);
	}

	// ------------------------------------------------------------------
	// 4. Put one extraction on each side of the join:
	//      JOIN ─┬─ PROJ_0 [side-0 cols + PREDICT(task on side 0)]
	//            └─ PROJ_1 [side-1 cols + PREDICT(task on side 1)]
	//    Each extraction then runs once per row of its side (with that
	//    side's own dedup cache), and the build side (children[1]) is fully
	//    extracted before any pair is formed, which the runtime check uses.
	// ------------------------------------------------------------------
	// Number of output columns of the join's left side before the rewrite (for remapping projection maps).
	const idx_t old_left_output_count = join.children[0]->GetColumnBindings().size();
	ColumnBinding task_output[2]; // indexed by task (0 = task1, 1 = task2)
	for (idx_t task = 0; task < 2; task++) {
		const idx_t side = input_side[task];
		auto &child = join.children[side];
		child->ResolveOperatorTypes();
		auto child_bindings = child->GetColumnBindings();
		auto &child_types = child->types;
		D_ASSERT(child_bindings.size() == child_types.size());

		idx_t input_col = DConstants::INVALID_INDEX;
		vector<unique_ptr<Expression>> exprs;
		for (idx_t i = 0; i < child_bindings.size(); i++) {
			exprs.push_back(make_uniq<BoundColumnRefExpression>(child_types[i], child_bindings[i]));
			if (child_bindings[i] == input_binding[task]) {
				input_col = i;
			}
		}
		D_ASSERT(input_col != DConstants::INVALID_INDEX);

		auto info = task == 0 ? MakeTaskInfo(orig_info, decomposed.task1, decomposed.task1_output_column,
		                                     decomposed.task1_output_type, orig_inputs[0])
		                      : MakeTaskInfo(orig_info, decomposed.task2, decomposed.task2_output_column,
		                                     decomposed.task2_output_type, orig_inputs[1]);
		if (task == 0) {
			info->decomposition_note = decision_note;
		}
		if (guard) {
			info->decomposition_guard = guard;
			info->guard_role =
			    side == 1 ? DecompositionGuardRole::BUILD_EXTRACTION : DecompositionGuardRole::PROBE_EXTRACTION;
		}
		vector<unique_ptr<Expression>> args;
		args.push_back(make_uniq<BoundColumnRefExpression>(child_types[input_col], child_bindings[input_col]));
		auto task_predict = make_uniq<BoundPredictExpression>(info->result_set_types[0], std::move(args));
		task_predict->bound_predict = std::move(info);
		exprs.push_back(std::move(task_predict));

		const idx_t proj_idx = optimizer.binder.GenerateTableIndex();
		auto proj = make_uniq<LogicalProjection>(proj_idx, std::move(exprs));
		task_output[task] = ColumnBinding(proj_idx, child_bindings.size());

		// Operators above the projection now see the side's columns through it.
		BindingFix fix;
		fix.stop_operator = proj.get();
		for (idx_t i = 0; i < child_bindings.size(); i++) {
			fix.old_bindings.push_back(child_bindings[i]);
			fix.new_bindings.push_back(ColumnBinding(proj_idx, i));
		}
		pending_fixes.push_back(std::move(fix));

		if (child->has_estimated_cardinality) {
			proj->SetEstimatedCardinality(child->estimated_cardinality);
		}
		proj->children.push_back(std::move(child));
		proj->ResolveOperatorTypes();
		child = std::move(proj);

		// A join with a projection map on this side must also output the extracted column.
		if (join.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    join.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &logical_join = join.Cast<LogicalJoin>();
			auto &side_map = side == 0 ? logical_join.left_projection_map : logical_join.right_projection_map;
			if (!side_map.empty()) {
				side_map.push_back(child_bindings.size());
			}
		}
	}
	// The left side gained a column: shift indexes of right-side columns in a direct filter's projection map.
	if (chain.empty() && !filter.projection_map.empty()) {
		for (auto &idx : filter.projection_map) {
			if (idx >= old_left_output_count) {
				idx++;
			}
		}
	}

	// ------------------------------------------------------------------
	// 5. Replace the PREDICT conjunct with θ = task1 <op> task2.
	// ------------------------------------------------------------------
	auto orig_expr = std::move(filter.expressions[predict_idx]);
	filter.expressions.erase_at(predict_idx);
	filter.limit = -1;
	string plan_shape;

	if (!guarded && join.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		// Unchecked decomposition over a cross product: θ becomes the join condition (hash join for =).
		JoinCondition cond;
		const bool task1_left = input_side[0] == 0;
		cond.left = make_uniq<BoundColumnRefExpression>(output_type, task1_left ? task_output[0] : task_output[1]);
		cond.right = make_uniq<BoundColumnRefExpression>(output_type, task1_left ? task_output[1] : task_output[0]);
		cond.comparison = task1_left ? comparison : FlipComparisonExpression(comparison);
		auto comparison_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
		comparison_join->conditions.push_back(std::move(cond));
		comparison_join->children = std::move(join.children);
		if (join.has_estimated_cardinality) {
			comparison_join->SetEstimatedCardinality(join.estimated_cardinality);
		}
		auto &join_parent = chain.empty() ? *op : *chain.back();
		join_parent.children[0] = std::move(comparison_join);
		plan_shape = "COMPARISON_JOIN on the extracted columns";
	} else {
		// The filter references the extracted columns, passed up through the projection chain (bottom first).
		ColumnBinding filter_side[2] = {task_output[0], task_output[1]};
		for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
			auto &proj = **it;
			for (idx_t task = 0; task < 2; task++) {
				proj.expressions.push_back(make_uniq<BoundColumnRefExpression>(output_type, filter_side[task]));
				filter_side[task] = ColumnBinding(proj.table_index, proj.expressions.size() - 1);
			}
			proj.ResolveOperatorTypes();
		}
		auto theta = make_uniq<BoundComparisonExpression>(
		    comparison, make_uniq<BoundColumnRefExpression>(output_type, filter_side[0]),
		    make_uniq<BoundColumnRefExpression>(output_type, filter_side[1]));
		if (guarded) {
			filter.guard_condition = std::move(theta);
			filter.guard_fallback = std::move(orig_expr);
			filter.decomposition_guard = guard;
			plan_shape = "LLM_JOIN_GUARD [theta | fallback: original PREDICT]";
		} else {
			filter.expressions.push_back(std::move(theta));
			plan_shape = "FILTER [theta]";
		}
	}

	const char *op_sym = "=";
	switch (comparison) {
	case ExpressionType::COMPARE_NOTEQUAL:             op_sym = "!="; break;
	case ExpressionType::COMPARE_LESSTHAN:             op_sym = "<"; break;
	case ExpressionType::COMPARE_GREATERTHAN:          op_sym = ">"; break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:    op_sym = "<="; break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO: op_sym = ">="; break;
	default:                                           break;
	}
	std::cout << "\n[SplitPredictJoin] Plan rewrite (" << decision_note << ")\n"
	          << "  Original PREDICT prompt : " << orig_prompt << "\n"
	          << "  Model                   : " << orig_model << "\n"
	          << "  Decomposed task1 prompt : " << decomposed.task1 << "\n"
	          << "  Decomposed task2 prompt : " << decomposed.task2 << "\n"
	          << "  theta                   : " << decomposed.task1_output_column << " " << op_sym << " "
	          << decomposed.task2_output_column << "\n"
	          << "  New plan structure      : " << plan_shape << " over the join of\n"
	          << "                            PROJ [side cols + PREDICT(task)] on each side\n" << std::endl;
	transformed = true;
#endif // ENABLE_PREDICT && ENABLE_LLM_API

	return op;
}

// ---------------------------------------------------------------------------
unique_ptr<LogicalOperator>
SplitPredictJoinToExtractPredicate::Transform(unique_ptr<LogicalOperator> op) {
	if (CanOptimize(*op)) {
		op = TransformFilterJoin(std::move(op));
	}
	for (auto &child : op->children) {
		child = Transform(std::move(child));
	}
	return op;
}

// ---------------------------------------------------------------------------
unique_ptr<LogicalOperator>
SplitPredictJoinToExtractPredicate::Optimize(unique_ptr<LogicalOperator> plan) {
	// Phase 1: find and transform all qualifying filter+join patterns.
	plan = Transform(std::move(plan));

	// Phase 2: for every extraction projection inserted below a join, update
	// the operators above it (the join, the filter and everything up to the
	// root) from the side's old bindings to the projection's bindings.
	for (auto &fix : pending_fixes) {
		D_ASSERT(fix.old_bindings.size() == fix.new_bindings.size());

		ColumnBindingReplacer replacer;
		for (idx_t i = 0; i < fix.old_bindings.size(); i++) {
			replacer.replacement_bindings.emplace_back(fix.old_bindings[i], fix.new_bindings[i]);
		}
		// Stop at the projection itself: its pass-through expressions refer to the old bindings.
		replacer.stop_operator = fix.stop_operator;
		replacer.VisitOperator(*plan);
	}
	if (transformed) {
		plan->ResolveOperatorTypes();
	}

	return plan;
}

} // namespace duckdb
