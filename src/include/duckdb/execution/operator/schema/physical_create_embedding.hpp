//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/schema/physical_create_embedding.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/parsed_data/create_embedding_info.hpp"

namespace duckdb {

//! PhysicalCreateEmbedding represents a CREATE EMBEDDING command.
//! It adds a vector column to the target table and registers a catalog entry
//! tracking the table.column → embedding_column relationship.
class PhysicalCreateEmbedding : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CREATE_EMBEDDING;

public:
	explicit PhysicalCreateEmbedding(PhysicalPlan &physical_plan, unique_ptr<CreateEmbeddingInfo> info,
	                                 idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::CREATE_EMBEDDING, {LogicalType::BIGINT},
	                       estimated_cardinality),
	      info(std::move(info)) {
	}

	unique_ptr<CreateEmbeddingInfo> info;

public:
	// Source interface
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}
};

} // namespace duckdb