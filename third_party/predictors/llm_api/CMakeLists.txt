if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(include ${DUCKDB_MODULE_BASE_DIR}/third_party/httplib)

add_library(duckdb_llm_api STATIC
        llm_api.cpp
        https_client.cpp
        llm_grammar.cpp
        llm_batching.cpp
        llm_predict.cpp
        llm_dedup.cpp)
target_compile_definitions(duckdb_llm_api PRIVATE $<$<BOOL:${ENABLE_LLM_API}>:ENABLE_LLM_API>)

target_link_libraries(duckdb_llm_api ${CURL_LIBRARIES})
target_link_libraries(duckdb_llm_api duckdb_mbedtls ${OPENSSL_LIBRARIES})

target_include_directories(
  duckdb_llm_api
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

set_target_properties(duckdb_llm_api PROPERTIES EXPORT_NAME duckdb_llm_api)

install(TARGETS duckdb_llm_api
        EXPORT "${DUCKDB_EXPORT_SET}"
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

disable_target_warnings(duckdb_llm_api)
