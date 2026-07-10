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

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Performance Benchmarks](PERFORMANCE.md)
- [Code Examples](EXAMPLES.md)
- [Visualization Guide](VISUALIZATION_GUIDE.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT
