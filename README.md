# KNN Classifier with Console Visualization

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-ready K-Nearest Neighbors classifier in C++20 with KD-tree
optimizations, parallel prediction, and ASCII console visualization.

## Features

- **KD-Tree Spatial Indexing**: 30x faster nearest neighbor search
- **Parallel Batch Prediction**: 4x speedup with multi-threading
- **Weighted Voting**: Uniform, Inverse Distance, Gaussian kernel
- **Cross-Validation**: K-fold CV with automatic hyperparameter tuning
- **ASCII Visualization**: Full ANSI color support, multiple modes
- **Interactive Mode**: Real-time query point classification
- **Comprehensive Testing**: 62+ unit tests with Google Test

## Project Structure

```
.
├── CMakeLists.txt
├── .clang-format
├── .gitignore
├── bin/
│   ├── CMakeLists.txt
│   └── main.cpp
├── include/
│   ├── benchmark.hpp
│   ├── dataset.hpp
│   ├── kdtree.hpp
│   ├── knn.hpp
│   ├── knn_optimized.hpp
│   ├── point.hpp
│   └── visualizer.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── benchmark.cpp
│   ├── dataset.cpp
│   ├── kdtree.cpp
│   ├── knn.cpp
│   ├── knn_optimized.cpp
│   └── visualizer.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── dataset_ut.cpp
│   ├── kdtree_ut.cpp
│   ├── knn_optimized_ut.cpp
│   ├── knn_ut.cpp
│   ├── point_ut.cpp
│   └── visualizer_ut.cpp
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   ├── EXAMPLES.md
│   ├── PERFORMANCE.md
│   └── VISUALIZATION_GUIDE.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Build

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

## Usage

```bash
./bin/knn-classifier
```

## Testing

62+ unit tests covering all components.

## Design Highlights

The KD-tree partitions points by median on alternating axes (x → y → x...).
Search prunes branches when the splitting plane is farther than the current
k-th neighbor — this gives O(log n) average lookup.

The visualizer maps the 2D plane onto an ASCII grid. Each cell runs
`predict(cell_center, k)` and gets rendered as a colored background block.
Training points overlay with `●■▲♦★◆` symbols. Query points glow green `Q`,
and k-neighbors are highlighted yellow `★`. Up to 8 classes get distinct
ANSI colors; beyond that the palette cycles.

Parallel prediction splits the query set into chunks, one per thread.
Each thread reads the shared KD-tree without locking — the tree is
immutable after construction, and every thread writes to its own slice
of the result vector.

The benchmark suite uses a `measure<Func>` template that runs any
callable N times, then computes mean, min, max, and standard deviation
in milliseconds.

## Project Links

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Performance**: [PERFORMANCE.md](PERFORMANCE.md)
- **Examples**: [EXAMPLES.md](EXAMPLES.md)
- **Visualization Guide**: [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **CI**: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## License

MIT
