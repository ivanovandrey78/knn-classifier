# Architecture Documentation

## Overview

KNN Classifier is a production-ready C++20 machine learning library implementing K-Nearest Neighbors classification with advanced optimizations and visualization capabilities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Interactive │  │     CLI      │  │  Benchmark   │      │
│  │     Mode     │  │    Tools     │  │    Suite     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Core Library (lib)                        │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Classification Module                    │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │   KNN    │  │ KNN Optimized │  │   Neighbor    │  │   │
│  │  │  (Base)  │  │  (KD-Tree +  │  │   Structure   │  │   │
│  │  │          │  │   Parallel)   │  │               │  │   │
│  │  └──────────┘  └──────────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Data Structures Module                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │  Point   │  │ KD-Tree  │  │  Cluster Config  │   │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Dataset Module                           │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌────────────┐   │   │
│  │  │ Generation │  │ Train/Test   │  │  Metrics   │   │   │
│  │  │ (Gaussian) │  │    Split     │  │ (Accuracy, │   │   │
│  │  │            │  │              │  │ Confusion) │   │   │
│  │  └────────────┘  └──────────────┘  └────────────┘   │   │
│  │  ┌────────────┐  ┌──────────────┐                   │   │
│  │  │  CSV I/O   │  │Normalization │                   │   │
│  │  └────────────┘  └──────────────┘                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Visualization Module                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │   │
│  │  │   Decision   │  │ Scatter Plot │  │ K-Nearest│   │   │
│  │  │   Boundary   │  │ Rendering    │  │ Neighbor │   │   │
│  │  │  (ASCII+ANSI)│  │              │  │Highlight │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘   │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ Color System │  │ Config       │                 │   │
│  │  │ (8+ classes) │  │ Management   │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Cross-Validation & Utilities                 │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ K-Fold   │  │  Find Optimal│  │  Benchmark   │   │   │
│  │  │    CV    │  │      K       │  │    Suite     │   │   │
│  │  └──────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Standard Library                           │
│  STL, Threading, Filesystem, Chrono, Random                  │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Point Class
**File**: `include/point.hpp`

**Responsibility**: Fundamental 2D point representation

**Key Features**:
- Coordinates (x, y) as doubles
- Label (class) as int
- Euclidean distance calculation
- Comparison operators with epsilon tolerance
- Stream output operator

**Dependencies**: `<cmath>`, `<ostream>`

---

### 2. KNN Classifier (Base)
**Files**: `include/knn.hpp`, `src/knn.cpp`

**Responsibility**: Core K-Nearest Neighbors algorithm

**Key Features**:
- Training data storage
- Distance-based prediction
- K-nearest neighbor retrieval
- Confidence scoring
- Batch prediction
- Tie-breaking by smallest label

**Algorithm Complexity**:
- Training: O(1) - just stores data
- Prediction: O(n) - linear scan
- Memory: O(n) where n = training set size

**Dependencies**: `Point`, `<vector>`, `<map>`, `<algorithm>`

---

### 3. KNN Optimized
**Files**: `include/knn_optimized.hpp`, `src/knn_optimized.cpp`

**Responsibility**: Performance-optimized KNN with spatial indexing

**Key Features**:
- KD-Tree for O(log n) neighbor search
- Parallel batch prediction
- Weighted voting strategies:
  - Uniform (standard)
  - Inverse distance (1/d)
  - Squared inverse (1/d²)
  - Gaussian kernel
- Thread pool for parallelization

**Algorithm Complexity**:
- Build KD-tree: O(n log n)
- Prediction: O(log n) average, O(n) worst case
- Parallel batch: O(m log n / p) where m = queries, p = threads

**Dependencies**: `KNN`, `KDTree`, `<thread>`, `<future>`

---

### 4. KD-Tree
**Files**: `include/kdtree.hpp`, `src/kdtree.cpp`

**Responsibility**: Spatial indexing for fast neighbor search

**Key Features**:
- Binary space partitioning
- Recursive construction
- K-nearest neighbor search
- Range queries
- Axis-aligned splits

**Algorithm**: 
- Construction: Median-based partitioning
- Search: Depth-first with pruning
- Complexity: O(log n) average for balanced tree

**Dependencies**: `Point`, `<memory>`, `<algorithm>`

---

### 5. Dataset Module
**Files**: `include/dataset.hpp`, `src/dataset.cpp`

**Responsibility**: Data generation, manipulation, and evaluation

**Key Features**:
- Gaussian cluster generation
- Train/test splitting with shuffling
- Accuracy and confusion matrix calculation
- CSV serialization
- Data normalization
- Statistical bounds calculation

**Utilities**:
- `generateClusters()`: Multi-class generation
- `generateSimpleDataset()`: Quick 2-class setup
- `trainTestSplit()`: Stratified splitting
- `calculateAccuracy()`: Classification metrics
- `confusionMatrix()`: Error analysis
- `normalize()`: Feature scaling to [0,1]

**Dependencies**: `Point`, `<random>`, `<fstream>`, `<map>`

---

### 6. Visualizer Module
**Files**: `include/visualizer.hpp`, `src/visualizer.cpp`

**Responsibility**: ASCII console visualization with ANSI colors

**Key Features**:
- Decision boundary rendering
- Training data overlay
- Query point highlighting
- K-nearest neighbors display
- Multi-class color support (8+ classes)
- Configurable grid resolution
- Legend and axis labeling

**Visualization Modes**:
1. `drawMap()`: Decision boundaries only
2. `drawMapWithData()`: Boundaries + training points
3. `drawDataPoints()`: Scatter plot
4. `drawMapWithQuery()`: Highlight query
5. `drawMapWithNeighbors()`: Show k neighbors

**ANSI Colors**:
- Class colors: Blue, Red, Green, Yellow, Magenta, Cyan
- Special markers: Green (query), Yellow (neighbors)
- Background colors for decision regions

**Dependencies**: `KNN`, `Point`, `<iostream>`, `<sstream>`

---

### 7. Cross-Validation
**File**: `include/knn_optimized.hpp` (namespace CrossValidation)

**Responsibility**: Model evaluation and hyperparameter tuning

**Key Features**:
- K-fold cross-validation
- Optimal k selection
- Stratified sampling
- Statistical analysis

**Functions**:
- `kFoldCV()`: Perform k-fold validation
- `findOptimalK()`: Grid search for best k

**Dependencies**: `KNNOptimized`, `Dataset`, `<numeric>`

---

### 8. Benchmark Suite
**Files**: `include/benchmark.hpp`, `src/benchmark.cpp`

**Responsibility**: Performance measurement and comparison

**Key Features**:
- High-precision timing (microseconds)
- Statistical analysis (mean, min, max, stddev)
- Comparison benchmarks:
  - Standard vs Optimized KNN
  - Linear vs KD-tree search
  - Sequential vs Parallel
  - Weight strategies
- Scalability testing
- CSV export of results

**Metrics**:
- Execution time
- Throughput (predictions/sec)
- Speedup ratios
- Memory usage (conceptual)

**Dependencies**: `KNN`, `KNNOptimized`, `<chrono>`, `<functional>`

---

## Data Flow

### 1. Training Flow
```
Raw Data → ClusterConfig → Dataset::generateClusters()
                ↓
        Vector<Point> (dataset)
                ↓
        trainTestSplit()
                ↓
        Train Set → KNN/KNNOptimized
                         ↓
                   buildKDTree() [optional]
                         ↓
                   Trained Classifier
```

### 2. Prediction Flow
```
Query Point → predict(query, k)
                    ↓
        [KD-Tree Search] OR [Linear Scan]
                    ↓
        K-Nearest Neighbors (sorted by distance)
                    ↓
        Vote Counting (weighted/uniform)
                    ↓
        Predicted Label + Confidence
```

### 3. Visualization Flow
```
Trained Classifier → Visualizer Config
                          ↓
        Grid Generation (NxN cells)
                          ↓
        For each cell: predict(cell_center, k)
                          ↓
        ANSI Color Mapping
                          ↓
        Console Rendering
```

---

## Threading Model

### Parallel Batch Prediction

```
Main Thread
    │
    ├─ Split queries into chunks
    │
    ├─ Spawn Worker Threads
    │   ├─ Thread 1: queries[0...n/4]
    │   ├─ Thread 2: queries[n/4...n/2]
    │   ├─ Thread 3: queries[n/2...3n/4]
    │   └─ Thread 4: queries[3n/4...n]
    │
    └─ Join threads
        │
        └─ Aggregate results
```

**Thread Safety**:
- Each thread has read-only access to classifier
- Each thread writes to separate result indices
- No shared mutable state
- Join synchronization ensures completion

---

## Memory Layout

### KNN Classifier
```
KNN
├─ trainingData: vector<Point>     [n * sizeof(Point)]
│                                   ≈ n * 24 bytes
│
└─ KNNOptimized (additional)
   └─ kdTree: unique_ptr<KDNode>   [n * sizeof(KDNode)]
                                    ≈ n * 64 bytes
```

### KD-Tree Structure
```
KDNode
├─ point: Point (24 bytes)
├─ left: unique_ptr<KDNode> (8 bytes)
├─ right: unique_ptr<KDNode> (8 bytes)
└─ axis: int (4 bytes)
Total: ≈ 64 bytes per node
```

**Memory Efficiency**:
- Compact point representation
- Smart pointers for automatic cleanup
- No memory leaks (RAII pattern)
- Vector reserve() to minimize reallocations

---

## Performance Characteristics

### Time Complexity

| Operation | Standard KNN | Optimized KNN (KD-tree) |
|-----------|--------------|-------------------------|
| Build/Train | O(1) | O(n log n) |
| Single Prediction | O(n) | O(log n) avg, O(n) worst |
| Batch (m queries) | O(mn) | O(m log n) avg |
| Parallel Batch | O(mn/p) | O(m log n / p) |
| Range Query | O(n) | O(√n + k) |

Where:
- n = training set size
- m = number of queries
- k = number of neighbors
- p = number of threads

### Space Complexity

| Component | Space |
|-----------|-------|
| Standard KNN | O(n) |
| KD-Tree | O(n) additional |
| Batch Prediction | O(m) for results |
| Visualization Grid | O(grid²) ≈ O(1600) for 40x40 |

---

## Design Patterns

### 1. Strategy Pattern
**Where**: Weight strategies in KNNOptimized

```cpp
enum class WeightStrategy {
    UNIFORM,
    INVERSE_DISTANCE,
    SQUARED_INVERSE,
    GAUSSIAN
};
```

### 2. Template Method
**Where**: Benchmark::measure<Func>

Generic benchmark function accepting any callable.

### 3. Factory Pattern
**Where**: Dataset cluster generation

```cpp
ClusterConfig(centerX, centerY, stdDev, label, numPoints)
```

### 4. RAII (Resource Acquisition Is Initialization)
**Where**: KD-Tree nodes, file handles

Automatic resource management via smart pointers and destructors.

### 5. Singleton (conceptual)
**Where**: Namespaces (Dataset, Visualizer, CrossValidation)

Stateless utility functions grouped logically.

---

## Extension Points

### Adding New Distance Metrics
1. Modify `Point::distanceTo()` or create `DistanceMetric` interface
2. Update KNN prediction logic

### Adding New Visualizations
1. Add function to `Visualizer` namespace
2. Implement grid mapping and rendering

### Adding New Weight Strategies
1. Extend `WeightStrategy` enum
2. Add case in `predictWeighted()`

### Adding New Benchmark Tests
1. Implement test function in `Benchmark` namespace
2. Call from `runFullSuite()`

---

## Build System

### CMake Structure
```
CMakeLists.txt (root)
├─ src/CMakeLists.txt → lib
├─ bin/CMakeLists.txt → executable
└─ tests/CMakeLists.txt → unit tests
```

### Compilation Flags
- C++20 standard
- `-Wall -Wextra -Wpedantic -Werror`
- Optimization: `-O2` (default)
- Threading: `-pthread`

### Dependencies
- Standard C++20 library
- POSIX threads (pthread)
- Google Test (optional, for testing)

---

## Testing Strategy

### Test Coverage
- **Unit Tests**: 70+ tests via Google Test
  - Point operations
  - Dataset generation
  - KNN predictions
  - Visualizer rendering
  - KD-Tree operations
  
- **Integration Tests**: Full workflow demos in main
  
- **Performance Tests**: Benchmark suite

### Test Categories
1. **Correctness**: Algorithm accuracy
2. **Edge Cases**: Empty data, k=1, k=n
3. **Performance**: Scalability, speedup
4. **Robustness**: Invalid inputs, error handling

---

## Future Architecture Improvements

1. **Plugin System**: Dynamic algorithm loading
2. **Serialization**: Save/load trained models
3. **GPU Acceleration**: CUDA/OpenCL for massive datasets
4. **Distributed**: MPI for cluster computing
5. **Streaming**: Online learning with data streams

---

## References

- KD-Tree: Bentley, J.L. (1975). "Multidimensional binary search trees"
- KNN: Cover & Hart (1967). "Nearest neighbor pattern classification"
- C++20 Standard: ISO/IEC 14882:2020
