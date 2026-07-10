# Architecture

## Layered Structure

```
┌─────────────────────────────────────────────────────┐
│                     Application                       │
│  main.cpp — CLI, interactive mode, benchmarks         │
├─────────────────────────────────────────────────────┤
│                   Core Library (lib)                   │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Classifiers                                     │ │
│  │  Knn (base)  →  KnnOptimized (KD-tree + threads)│ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────┐ ┌────────────────────────────┐ │
│  │  Data Structures  │ │  Dataset Utilities          │ │
│  │  Point            │ │  generation, split, CSV,    │ │
│  │  KD-tree (KdNode) │ │  normalize, metrics         │ │
│  └──────────────────┘ └────────────────────────────┘ │
│                                                       │
│  ┌──────────────────┐ ┌────────────────────────────┐ │
│  │  Visualization    │ │  Cross-Validation           │ │
│  │  ANSI ASCII map   │ │  k-fold CV, optimal K      │ │
│  └──────────────────┘ └────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Benchmark Suite                                 │ │
│  │  timing, speedup, CSV export                     │ │
│  └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│                   Standard Library                     │
│  STL — vector, memory, thread, chrono, random, I/O   │
└─────────────────────────────────────────────────────┘
```

## File Map

| File | Purpose |
|------|---------|
| `include/point.hpp` | 2D point with distance, comparisons |
| `include/knn.hpp` | Base KNN classifier |
| `include/kdtree.hpp` | KD-tree spatial index |
| `include/knn_optimized.hpp` | KD-tree + parallel + weighted + CV |
| `include/dataset.hpp` | Dataset generation, split, metrics |
| `include/visualizer.hpp` | ASCII ANSI console drawing |
| `include/benchmark.hpp` | Performance measurement |
| `bin/main.cpp` | Entry point, demo modes |

## Data Flow

```
Training:
  Clusters → generateClusters() → Vector<Point>
                                     ↓
                               trainTestSplit()
                                     ↓
                          Train → Knn / KnnOptimized

Prediction:
  Query → predict(k) → [KD-tree / linear] → neighbors → votes → label

Visualization:
  Classifier → grid cells → predict each cell → ANSI → console
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| `Knn` base + `KnnOptimized` derived | Swap implementations without changing API |
| KD-tree via `std::unique_ptr` | Automatic cleanup, no leaks |
| Namespace `cross_validation` instead of class | Stateless utility, no instance needed |
| `measure<Func>` template | Generic — works with any callable |
| `CoutGuard` RAII in tests | No manual cout.rdbuf / restore pairs |
| `const` methods + no shared state | Thread-safe parallel prediction |

## Complexity

| Operation | Knn | KnnOptimized |
|-----------|-----|--------------|
| Train     | O(1) | O(n log n) |
| Predict   | O(n) | O(log n) avg |
| Batch m queries | O(m n) | O(m log n) |
| Parallel batch | — | O(m log n / p) |

n = training size, m = queries, p = threads.
