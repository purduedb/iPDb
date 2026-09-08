#pragma once

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/prompt.hpp"
#include "duckdb/common/string_util.hpp"

#include "nlohmann/json.hpp"
#include <regex>
#include <iostream>
#include <sstream>

namespace duckdb {

class PromptUtil {
public:
	static std::string extract_json(const std::string &text) {
		const size_t start = text.find_first_of("{[");
		if (start == std::string::npos) {
			throw std::runtime_error("No JSON start found");
		}

		const char open = text[start];
		const char close = open == '{' ? '}' : ']';

		int depth = 0;
		for (size_t i = start; i < text.size(); i++) {
			if (text[i] == open) {
				depth++;
			} else if (text[i] == close) {
				depth--;
				if (depth == 0) {
					return text.substr(start, i - start + 1);
				}
			}
		}

		throw std::runtime_error("No matching JSON end found");
	}

	static nlohmann::json parse_json(std::string json_str) {
		return nlohmann::json::parse(json_str);
	}

	static Value extract_longest_integer(const std::string &input) {
		std::regex re("\\d+");
		std::sregex_iterator it(input.begin(), input.end(), re);

		std::string longest;
		for (std::sregex_iterator end; it != end; ++it) {
			if (it->str().size() > longest.size()) {
				longest = it->str();
			}
		}
		if (!longest.empty())
			return Value(std::stoi(longest));
		return Value(LogicalTypeId::INTEGER);
	}

	static double extract_double(const nlohmann::json &value) {
		if (value.is_number_float() || value.is_number_integer()) {
			return value.get<double>();
		}
		if (value.is_string()) {
			try {
				return std::stod(value.get<std::string>());
			} catch (...) {
				throw std::runtime_error("Invalid number string");
			}
		}
		throw std::runtime_error("Value is not a number or string");
	}

	static std::string strip_code_fences(const std::string &input) {
		std::istringstream ss(input);
		std::string line;
		std::ostringstream out;
		while (std::getline(ss, line)) {
			// skip opening and closing ``` lines
			if (line.rfind("```", 0) == 0)
				continue;
			out << line << "\n";
		}
		return out.str();
	}

	static void process_prompt_and_extract_types(std::vector<std::pair<std::string, LogicalTypeId>> &attrs,
	                                             std::string &prompt) {
		const std::regex out_re(Prompt::OUT_REGEX, std::regex_constants::icase);
		auto words_begin = std::sregex_iterator(prompt.begin(), prompt.end(), out_re);
		auto words_end = std::sregex_iterator();

		for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
			std::smatch match = *i;
			std::string match_str = match.str();
			auto attr = match[1].str();
			auto type = match[2].str();
			attrs.push_back(std::make_pair(attr, Prompt::type_to_logical_type(type)));
		}
	}

	static std::string embed_prompt(const idx_t row, const DataChunk &input, const PredictInfo &info,
	                                const bool is_multi = false) {
		const std::string line_end = is_multi ? "`, " : "`;\n";
		std::stringstream ss;
		idx_t col_i = 0;
		for (const auto mask_i : info.input_mask) {
			const auto &col_name = info.input_set_names[col_i];
			bool is_emb_col = false;
			for (const auto &[src, emb] : info.embedding_column_map) {
				if (StringUtil::CIEquals(col_name, emb)) {
					is_emb_col = true;
					break;
				}
			}
			if (!is_emb_col) {
				ss << col_name << " = `";
				ss << input.GetValue(mask_i, row).ToSQLString();
				if (col_i <= info.input_mask.size())
					ss << line_end;
			}
			col_i++;
		}
		return ss.str();
	}

	template <typename Func>
	static void extract_array_data(const std::string &llm_out, DataChunk &output,
	                               const map<string, vector<idx_t>> &unprocessed, const idx_t i,
	                               const PredictInfo &info, const idx_t n_rows, Func cache_update_func) {
		bool is_fail = false;
		idx_t failed_from = i;
		idx_t failed_to = i + n_rows;
		try {
			auto out_json = nlohmann::json::parse(extract_json(llm_out));
			if (out_json.contains("output_array")) {
				out_json = out_json["output_array"];
			}
			if (out_json.is_array()) {
				int64_t row = static_cast<int64_t>(i);
				for (auto it = out_json.begin(); it != out_json.end(); ++it) {
					const auto unprocessed_row = std::next(unprocessed.begin(), row);
					cache_update_func(unprocessed_row->first, it->dump());
					for (const auto &tuple_id : unprocessed_row->second) {
						populate_row_data(*it, tuple_id, output, info);
					}
					row++;
				}
				if (row < i + n_rows - 1) {
					is_fail = true;
					failed_from = row;
				}
			} else {
				std::cout << "JSON parse issue: Array not found" << std::endl;
				is_fail = true;
			}
		} catch (const std::runtime_error &e) {
			std::cout << "Runtime error: " << e.what() << std::endl;
			is_fail = true;
		} catch (const nlohmann::json::parse_error &e) {
			std::cout << "JSON parse issue: " << e.what() << std::endl;
			is_fail = true;
		}
		if (is_fail) {
			for (idx_t row = failed_from; row < failed_to; ++row) {
				fill_null(row, output, info);
			}
		}
	}

	static void extract_array_data(const std::string &llm_out, DataChunk &output, const idx_t i,
	                               const PredictInfo &info, const bool resize = false, const idx_t n_rows = 0) {
		bool is_fail = false;
		idx_t failed_from = i;
		idx_t failed_to = i + n_rows;
		try {
			auto extracted_json = extract_json(llm_out);
			auto out_json = nlohmann::json::parse(extracted_json);
			auto out_array = out_json["output_array"];
			if (out_array.is_array()) {
				if (resize)
					output.SetCardinality(out_array.size());
				idx_t row = i;
				for (auto it = out_array.begin(); it != out_array.end(); ++it) {
					populate_row_data(*it, row, output, info);
					row++;
				}
				if (row < i + n_rows) {
					is_fail = true;
					failed_from = row;
				}
			} else {
				std::cout << "JSON parse issue: Array not found" << std::endl;
				is_fail = true;
			}
		} catch (const std::runtime_error &e) {
			std::cout << "Runtime error: " << e.what() << std::endl;
			is_fail = true;
		} catch (const nlohmann::json::parse_error &e) {
			std::cout << "JSON parse issue: " << e.what() << std::endl;
			is_fail = true;
		} catch (const nlohmann::json::type_error &e) {
			std::cout << "JSON type issue: " << e.what() << ". For: " << llm_out << std::endl;
			is_fail = true;
		}
		if (is_fail) {
			for (idx_t row = failed_from; row < failed_to; ++row) {
				fill_null(row, output, info);
			}
		}
	}

	static void extract_array_data(const std::string &llm_out, DataChunk &output, const vector<idx_t> &orig_rows,
	                               const PredictInfo &info, const idx_t n_rows) {
		bool is_fail = false;
		idx_t failed_local = 0;
		try {
			auto extracted_json = extract_json(llm_out);
			auto out_json = nlohmann::json::parse(extracted_json);
			auto out_array = out_json["output_array"];
			if (out_array.is_array()) {
				idx_t local_i = 0;
				for (auto it = out_array.begin(); it != out_array.end() && local_i < n_rows; ++it, ++local_i) {
					populate_row_data(*it, orig_rows[local_i], output, info);
				}
				if (local_i < n_rows) {
					is_fail = true;
					failed_local = local_i;
				}
			} else {
				std::cout << "JSON parse issue: Array not found" << std::endl;
				is_fail = true;
			}
		} catch (const std::runtime_error &e) {
			std::cout << "Runtime error: " << e.what() << std::endl;
			is_fail = true;
		} catch (const nlohmann::json::parse_error &e) {
			std::cout << "JSON parse issue: " << e.what() << std::endl;
			is_fail = true;
		} catch (const nlohmann::json::type_error &e) {
			std::cout << "JSON type issue: " << e.what() << std::endl;
			is_fail = true;
		}
		if (is_fail) {
			for (idx_t i = failed_local; i < n_rows; ++i) {
				fill_null(orig_rows[i], output, info);
			}
		}
	}

	static void extract_row_data(const std::string &llm_out, const idx_t row, DataChunk &output,
	                             const PredictInfo &info) {
		try {
			auto json_str = extract_json(llm_out);
			const auto out_json = nlohmann::json::parse(json_str);
			populate_row_data(out_json, row, output, info);
		} catch (const std::runtime_error &e) {
			std::cout << "Runtime error: " << e.what() << std::endl;
			fill_null(row, output, info);
		} catch (const nlohmann::json::parse_error &e) {
			std::cout << "JSON parse issue: " << e.what() << std::endl;
			fill_null(row, output, info);
		}
	}

	static void populate_row_data(const nlohmann::json &out_json, const idx_t row, DataChunk &output,
	                              const PredictInfo &info) {
		for (size_t j = 0; j < info.result_set_names.size(); j++) {
			auto output_type = info.result_set_types[j];
			auto col_name = info.result_set_names[j];

			try {
				if (!out_json.contains(col_name)) {
					output.SetValue(j, row, Value(output_type));
					continue;
				}

				if (output_type == LogicalTypeId::VARCHAR && out_json[col_name].is_string()) {
					auto value = out_json[col_name].get<std::string>();
					output.SetValue(j, row, Value(value));
				} else if (output_type == LogicalTypeId::INTEGER && out_json[col_name].is_string()) {
					auto value = extract_longest_integer(out_json[col_name].get<std::string>());
					output.SetValue(j, row, value);
				} else if (output_type == LogicalTypeId::INTEGER && out_json[col_name].is_number()) {
					int value = out_json[col_name].get<int>();
					output.SetValue(j, row, Value(value));
				} else if (output_type == LogicalTypeId::DOUBLE) {
					double value = extract_double(out_json[col_name]);
					output.SetValue(j, row, Value(value));
				} else if (output_type == LogicalTypeId::BOOLEAN && out_json[col_name].is_boolean()) {
					bool value = out_json[col_name].get<bool>();
					output.SetValue(j, row, Value(value));
				} else {
					output.SetValue(j, row, Value(output_type));
				}
			} catch (const nlohmann::json::parse_error &e) {
				std::cout << "JSON parse issue: " << e.what() << std::endl;
				output.SetValue(j, row, Value(output_type));
			} catch (const std::runtime_error &e) {
				std::cout << "Conversion error: " << e.what() << std::endl;
				output.SetValue(j, row, Value(output_type));
			}
		}
	}

	static void fill_null(const idx_t row, DataChunk &output, const PredictInfo &info) {
		for (size_t j = 0; j < info.result_set_names.size(); j++) {
			const auto output_type = info.result_set_types[j];
			auto col_name = info.result_set_names[j];

			auto &flat_vector = output.data[j];
			FlatVector::SetNull(flat_vector, row, true);
		}
	}
};

} // namespace duckdb