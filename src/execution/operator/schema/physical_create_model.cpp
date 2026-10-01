#include "duckdb/execution/operator/schema/physical_create_model.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception/catalog_exception.hpp"
#include "duckdb/main/secret/secret_manager.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateModel::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	if (!info->secret.empty()) {
		auto &secret_manager = SecretManager::Get(context.client);
		auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context.client);

		auto secret_entry = secret_manager.GetSecretByName(transaction, info->secret);
		
		if (secret_entry == nullptr) {
			throw CatalogException("Secret not found in the catalog!");
		}
	}

	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.CreateModel(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb
