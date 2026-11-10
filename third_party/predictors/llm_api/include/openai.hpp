// The MIT License (MIT)
//
// Copyright (c) 2023 Olrea, Florian Dang
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef OPENAI_HPP_
#define OPENAI_HPP_

#if OPENAI_VERBOSE_OUTPUT
#pragma message("OPENAI_VERBOSE_OUTPUT is ON")
#endif

#include "https_client.hpp"
#include "duckdb/common/exception/http_exception.hpp"
#include "duckdb/main/database_file_opener.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include "../../common/common.hpp" // nlohmann/json

namespace duckdb {

// Json alias
using Json = nlohmann::json;

// OpenAI
class OpenAI {
	HTTPSUtil http_util_ = HTTPSUtil();
	HTTPHeaders headers_;
	std::unique_ptr<HTTPParams> params_;
	std::string token_;
	std::string base_url_;
	bool throw_exception_;

public:
	explicit OpenAI(DatabaseInstance &db, const std::string &token = "", const bool throw_exception = true,
	                const std::string &api_base_url = "")
	    : token_ {token}, base_url_ {api_base_url}, throw_exception_ {throw_exception} {
		if (token.empty()) {
			if (const char *env_p = std::getenv("OPENAI_API_KEY")) {
				token_ = std::string {env_p};
			}
		}

		if (api_base_url.empty()) {
			if (const char *env_p = std::getenv("OPENAI_API_BASE")) {
				base_url_ = std::string {env_p} + "/";
			} else {
				base_url_ = api_base_url;
			}
		}

		DatabaseFileOpener opener {db};
		FileOpenerInfo info;
		info.file_path = base_url_;
		params_ = http_util_.InitializeParameters(&opener, &info);
	}

	OpenAI(const OpenAI &) = delete;
	OpenAI &operator=(const OpenAI &) = delete;
	OpenAI(OpenAI &&) = delete;
	OpenAI &operator=(OpenAI &&) = delete;

	void setToken(const std::string &token) {
		token_ = token;
		headers_.Insert("Authorization", StringUtil::Format("Bearer %s", token_));
	}

	void setThrowException(const bool throw_exception) {
		throw_exception_ = throw_exception;
	}

	Json post(const std::string &suffix, const std::string &data) {
		// Unclear what's peculiar about extension install flow, but those two parameters are needed
		// to avoid lengthy retry on 304
		params_->follow_location = false;
		params_->keep_alive = false;

		auto url = base_url_ + suffix;
#if OPENAI_VERBOSE_OUTPUT
		std::cout << "<< request: " << url << "  " << data << '\n';
#endif

		PostRequestInfo post_request(url, headers_, *params_, const_data_ptr_cast(data.c_str()),
		                                     data.size());
		post_request.try_request = true;

		auto response = http_util_.Request(post_request);
		if (!response->Success()) {
			if (response->HasRequestError()) {
				Json out {};
				out["error"] = response->GetRequestError();
				out["code"] = -1;
				return out;
			}
			Json out {};
			out["error"] = response->reason;
			out["code"] = response->status;
			// Keep the provider's error body (e.g. a 400 naming the unsupported schema keyword); it is the
			// only place the actual cause is reported.
			out["body"] = post_request.buffer_out;
			return out;
		}

		Json json {};
		auto parsable_out = PromptUtil::extract_json(post_request.buffer_out);
		if (isJson(parsable_out)) {
			json = Json::parse(parsable_out);
		} else {
			std::cout << "Response is not a valid JSON";
			std::cout << "<< " << post_request.buffer_out << "\n";
		}

		return json;
	}

	Json post(const std::string &suffix, const Json &json) {
		return post(suffix, json.dump());
	}

	void debug() const {
		std::cout << token_ << '\n';
	}

	void setBaseUrl(const std::string &url) {
		base_url_ = url;
	}

	std::string getBaseUrl() const {
		return base_url_;
	}

private:
	void checkResponse(const Json &json) const {
		if (json.count("error")) {
			const auto reason = json["error"].dump();
			trigger_error(reason);

#if OPENAI_VERBOSE_OUTPUT
			std::cerr << ">> response error :\n" << json.dump(2) << "\n";
#endif
		}
	}

	// as of now the only way
	static bool isJson(const std::string &data) {
		bool rc = true;
		try {
			auto json = Json::parse(data); // throws if no json
		} catch (std::exception &) {
			rc = false;
		}
		return rc;
	}

	void trigger_error(const std::string &msg) const {
		if (throw_exception_) {
			throw std::runtime_error(msg);
		}
		std::cerr << "[OpenAI] error. Reason: " << msg << '\n';
	}

public:
	static unique_ptr<OpenAI> createInstance(DatabaseInstance &db, const std::string &api_base_url = "", const std::string &token = "",
						 const bool throw_exception = true) {
		auto instance = make_uniq<OpenAI>(db, token, throw_exception, api_base_url);
		instance->setBaseUrl(api_base_url);
		instance->setToken(token);
		return instance;
	}
};

} // namespace openai

#endif // OPENAI_HPP_
