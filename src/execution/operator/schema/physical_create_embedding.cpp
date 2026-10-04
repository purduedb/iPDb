#include "duckdb/execution/operator/schema/physical_create_embedding.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/model_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/column_fill_callback.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
#include "llm_api.hpp"
#endif

namespace duckdb {

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
// Fill callback that writes pre-computed embedding vectors into the new column in row-sequential order.
// Rows are visited in physical order; rows without an embedding (NULL source) get a NULL array value.
struct EmbeddingFillCallback : public ColumnFillCallback {
	const vector<vector<float>> &embeddings;
	const vector<row_t> &emb_row_ids; // sorted ascending; parallel to embeddings
	const int dim;
	const bool is_array;
	idx_t pos;    // global physical row position (fills sequentially)
	idx_t emb_i;  // index into emb_row_ids / embeddings

	EmbeddingFillCallback(const vector<vector<float>> &embs, const vector<row_t> &row_ids, int d, bool arr)
	    : embeddings(embs), emb_row_ids(row_ids), dim(d), is_array(arr), pos(0), emb_i(0) {
	}

	void Fill(Vector &result, idx_t count) override {
		if (is_array) {
			auto &child_vec = ArrayVector::GetEntry(result);
			auto *child_data = FlatVector::GetData<float>(child_vec);
			auto &validity = FlatVector::Validity(result);

			for (idx_t i = 0; i < count; i++) {
				auto row_id = static_cast<row_t>(pos + i);
				if (emb_i < emb_row_ids.size() && emb_row_ids[emb_i] == row_id) {
					memcpy(child_data + i * dim, embeddings[emb_i].data(), dim * sizeof(float));
					emb_i++;
				} else {
					validity.SetInvalid(i);
					// Zero out child data for the null row (not strictly required, but clean)
					memset(child_data + i * dim, 0, dim * sizeof(float));
				}
			}
		} else {
			// LIST type
			auto *list_entries = FlatVector::GetData<list_entry_t>(result);
			auto &validity = FlatVector::Validity(result);
			ListVector::Reserve(result, count * dim);
			auto &child_vec = ListVector::GetEntry(result);
			auto *child_data = FlatVector::GetData<float>(child_vec);
			idx_t child_offset = 0;

			for (idx_t i = 0; i < count; i++) {
				auto row_id = static_cast<row_t>(pos + i);
				if (emb_i < emb_row_ids.size() && emb_row_ids[emb_i] == row_id) {
					list_entries[i].offset = child_offset;
					list_entries[i].length = static_cast<idx_t>(dim);
					memcpy(child_data + child_offset, embeddings[emb_i].data(), dim * sizeof(float));
					child_offset += static_cast<idx_t>(dim);
					emb_i++;
				} else {
					list_entries[i].offset = child_offset;
					list_entries[i].length = 0;
					validity.SetInvalid(i);
				}
			}
			ListVector::SetListSize(result, child_offset);
		}
		pos += count;
	}
};
#endif

SourceResultType PhysicalCreateEmbedding::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	auto &client = context.client;
	auto &catalog = Catalog::GetCatalog(client, info->catalog);

	// Determine the vector column type: fixed ARRAY when size is known, LIST otherwise.
	const LogicalType embedding_col_type = info->embedding_size > 0
	                                           ? LogicalType::ARRAY(LogicalType::FLOAT, info->embedding_size)
	                                           : LogicalType::LIST(LogicalType::FLOAT);

#if defined(ENABLE_PREDICT) && defined(ENABLE_LLM_API)
	// --- Phase 1: Gather source strings from the existing table (before any schema changes) ---

	auto &model_entry = Catalog::GetEntry(client, CatalogType::MODEL_ENTRY,
	                                      info->model_catalog, info->model_schema, info->model_name);
	auto model_data = model_entry.Cast<ModelCatalogEntry>().GetData();

	// Resolve the API key from the secret store.
	std::string api_key;
	if (!model_data.secret.empty()) {
		auto &secret_manager = SecretManager::Get(client);
		auto cat_txn = CatalogTransaction::GetSystemCatalogTransaction(client);
		if (const auto sec = secret_manager.GetSecretByName(cat_txn, model_data.secret)) {
			api_key = dynamic_cast<const KeyValueSecret &>(*sec->secret).TryGetValue("bearer_token").ToString();
		}
	}

	// Get the table as it exists now (before the ALTER).
	auto &table_entry_pre = Catalog::GetEntry(client, CatalogType::TABLE_ENTRY,
	                                          info->catalog, info->schema, info->table);
	auto &table_cat_pre = table_entry_pre.Cast<TableCatalogEntry>();
	auto &storage_pre = table_cat_pre.GetStorage();

	auto src_col_name = info->column;
	auto src_logical = table_cat_pre.GetColumnIndex(src_col_name);
	auto src_physical = table_cat_pre.GetColumns().LogicalToPhysical(src_logical);

	vector<StorageIndex> scan_col_ids = {
	    StorageIndex(COLUMN_IDENTIFIER_ROW_ID),
	    StorageIndex(src_physical.index),
	};
	vector<LogicalType> scan_types = {
	    LogicalType::BIGINT,
	    table_cat_pre.GetColumn(src_logical).Type(),
	};

	auto &txn = DuckTransaction::Get(client, storage_pre.db);
	TableScanState scan_state;
	storage_pre.InitializeScan(client, txn, scan_state, scan_col_ids);

	DataChunk scan_chunk;
	scan_chunk.Initialize(Allocator::Get(client), scan_types);

	vector<string> source_strings;
	vector<row_t> source_row_ids;

	while (true) {
		scan_chunk.Reset();
		storage_pre.Scan(txn, scan_chunk, scan_state);
		if (scan_chunk.size() == 0) {
			break;
		}
		for (idx_t i = 0; i < scan_chunk.size(); i++) {
			auto src_val = scan_chunk.data[1].GetValue(i);
			if (src_val.IsNull()) {
				continue;
			}
			source_strings.push_back(src_val.ToString());
			source_row_ids.push_back(scan_chunk.data[0].GetValue(i).GetValue<row_t>());
		}
	}

	// --- Phase 2: Call the embeddings API ---

	vector<vector<float>> embeddings;

	if (!source_strings.empty()) {
		auto &db = DatabaseInstance::GetDatabase(client);
		auto openai = OpenAI::createInstance(db, model_data.base_api, api_key);

		using json = nlohmann::json;
		json request;
		request["model"] = model_data.model_path;
		request["input"] = source_strings;
		if (info->embedding_size > 0) {
			request["dimensions"] = info->embedding_size;
		}

		auto response = openai->post("embeddings", request);
		if (response.contains("error")) {
			throw IOException("Embedding API error: %s", response["error"].get<std::string>());
		}

		if (response.contains("data")) {
			for (auto &entry : response["data"]) {
				if (entry.contains("embedding") && entry["embedding"].is_array()) {
					embeddings.push_back(entry["embedding"].get<std::vector<float>>());
				}
			}
		}

		if (embeddings.size() != source_row_ids.size()) {
			throw IOException("Embedding API returned %llu embeddings for %llu rows",
			                  (unsigned long long)embeddings.size(), (unsigned long long)source_row_ids.size());
		}
	}

	// --- Phase 3: ALTER TABLE to add the column, filling it with embeddings at creation time ---

	const int dim = embeddings.empty() ? static_cast<int>(info->embedding_size) : static_cast<int>(embeddings[0].size());
	const bool is_array = info->embedding_size > 0;
	auto fill_cb = make_uniq<EmbeddingFillCallback>(embeddings, source_row_ids, dim, is_array);

	AlterEntryData entry_data(info->catalog, info->schema, info->table, OnEntryNotFound::THROW_EXCEPTION);
	ColumnDefinition col_def(info->embedding_name, embedding_col_type);
	auto add_col =
	    make_uniq<AddColumnInfo>(std::move(entry_data), std::move(col_def), /*if_column_not_exists=*/true);
	add_col->fill_callback = std::move(fill_cb);
	catalog.Alter(client, *add_col);

#else
	// Without LLM API support, just add the NULL-filled column.
	AlterEntryData entry_data(info->catalog, info->schema, info->table, OnEntryNotFound::THROW_EXCEPTION);
	ColumnDefinition col_def(info->embedding_name, embedding_col_type);
	auto add_col =
	    make_uniq<AddColumnInfo>(std::move(entry_data), std::move(col_def), /*if_column_not_exists=*/true);
	catalog.Alter(client, *add_col);
#endif

	// Register the catalog entry that records table.column → embedding_name.
	catalog.CreateEmbedding(client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb
