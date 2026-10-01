#include "duckdb/execution/operator/schema/physical_drop.hpp"
#include "duckdb/catalog/catalog_entry/embedding_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalDrop::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	switch (info->type) {
	case CatalogType::PREPARED_STATEMENT: {
		// DEALLOCATE silently ignores errors
		auto &statements = ClientData::Get(context.client).prepared_statements;
		auto stmt_iter = statements.find(info->name);
		if (stmt_iter != statements.end()) {
			statements.erase(stmt_iter);
		}
		break;
	}
	case CatalogType::SCHEMA_ENTRY: {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
		catalog.DropEntry(context.client, *info);

		// Check if the dropped schema was set as the current schema
		auto &client_data = ClientData::Get(context.client);
		auto &default_entry = client_data.catalog_search_path->GetDefault();
		auto &current_catalog = default_entry.catalog;
		auto &current_schema = default_entry.schema;
		D_ASSERT(info->name != DEFAULT_SCHEMA);

		if (info->catalog == current_catalog && current_schema == info->name) {
			// Reset the schema to default
			SchemaSetting::SetLocal(context.client, DEFAULT_SCHEMA);
		}
		break;
	}
	case CatalogType::SECRET_ENTRY: {
		// Note: the schema param is used to optionally pass the storage to drop from
		D_ASSERT(info->extra_drop_info);
		auto &extra_info = info->extra_drop_info->Cast<ExtraDropSecretInfo>();
		SecretManager::Get(context.client)
		    .DropSecretByName(context.client, info->name, info->if_not_found, extra_info.persist_mode,
		                      extra_info.secret_storage);
		break;
	}
	case CatalogType::EMBEDDING_ENTRY: {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);

		// Fetch the entry so we can recover the table name before dropping it.
		EntryLookupInfo lookup(CatalogType::EMBEDDING_ENTRY, info->name);
		auto entry_ptr = Catalog::GetEntry(context.client, info->catalog, info->schema, lookup, info->if_not_found);
		if (entry_ptr) {
			auto &embedding = entry_ptr->Cast<EmbeddingCatalogEntry>();
			auto emb_data = embedding.GetData();

			// Remove the vector column from the table (IF EXISTS so retries are safe).
			AlterEntryData alter_data(info->catalog, info->schema, emb_data.table, OnEntryNotFound::RETURN_NULL);
			RemoveColumnInfo remove_col(std::move(alter_data), info->name,
			                           /*if_column_exists=*/true, /*cascade=*/info->cascade);
			catalog.Alter(context.client, remove_col);

			// Drop the embedding catalog entry.
			catalog.DropEntry(context.client, *info);
		}
		break;
	}
	default: {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
		catalog.DropEntry(context.client, *info);
		break;
	}
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb
