#pragma once

#define LLM_USE_THREADS 1
#define LLM_USE_CLUSTER 0
#define IS_SCHEMA 1

// Clustering algorithm — only used when LLM_USE_CLUSTER is 1 and embeddings are available.
// Override at compile time, e.g. -DLLM_CLUSTER_TECHNIQUE=LLM_CLUSTER_KNN
//
// LLM_CLUSTER_GREEDY: grow clusters greedily; each row joins the first existing cluster
//   whose representative's cosine-similarity exceeds LLM_CLUSTER_SIMILARITY_THRESHOLD,
//   otherwise a new cluster is started.
// LLM_CLUSTER_KNN: assign rows to the nearest of LLM_CLUSTER_K centroids, chosen via
//   k-means++ initialisation (deterministic: seed = row 0).
// LLM_CLUSTER_AHC: agglomerative hierarchical clustering with centroid linkage; starts
//   with one cluster per row and repeatedly merges the most similar pair (by centroid
//   cosine-similarity) until no pair exceeds LLM_CLUSTER_SIMILARITY_THRESHOLD.
//   Centroid of the merged cluster is the weighted average of the two centroids.
//   O(n³) worst-case — fine for typical DuckDB chunk sizes.
#define LLM_CLUSTER_GREEDY 0
#define LLM_CLUSTER_KNN    1
#define LLM_CLUSTER_AHC    2
#ifndef LLM_CLUSTER_TECHNIQUE
#define LLM_CLUSTER_TECHNIQUE LLM_CLUSTER_AHC
#endif
#ifndef LLM_CLUSTER_SIMILARITY_THRESHOLD
#define LLM_CLUSTER_SIMILARITY_THRESHOLD 0.7f
#endif
#ifndef LLM_CLUSTER_K
#define LLM_CLUSTER_K 512
#endif
// Maximum rows sampled per cluster for verification.
// Actual per-cluster sample = min(floor(sqrt(cluster_size - 1)), this cap).
#ifndef LLM_CLUSTER_VERIFY_SAMPLE_MAX
#define LLM_CLUSTER_VERIFY_SAMPLE_MAX 3
#endif

#ifdef NDEBUG
#define LLM_LOG(x) do {} while(0)
#else
#include <iostream>
// #define LLM_LOG(x) do {} while(0)
#define LLM_LOG(x) do { std::cout << x; } while(0)
#endif