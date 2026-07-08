# KNN Classifier with Console Visualization

A K-Nearest Neighbors classifier implementation in C++20 with ASCII-based console visualization.

## Features

- K-Nearest Neighbors classification algorithm
- Synthetic dataset generation
- Console-based visualization using ASCII graphics and ANSI colors
- Interactive mode for testing predictions
- Comprehensive unit tests with Google Test

## Requirements

- CMake 3.20 or higher
- C++20 compatible compiler (Clang recommended)
- Linux/macOS/Windows with terminal supporting ANSI colors

## Project Structure

```
.
├── CMakeLists.txt          # Root build configuration
├── bin/
│   ├── CMakeLists.txt      # Executable build config
│   └── main.cpp            # Entry point
├── include/
│   ├── point.hpp           # Point class definition
│   ├── knn.hpp             # KNN classifier header
│   └── visualizer.hpp      # Console visualization header
├── src/
│   ├── CMakeLists.txt      # Library build config
│   └── knn.cpp             # KNN classifier implementation
├── tests/
│   ├── CMakeLists.txt      # Test build config
│   └── test_point.cpp      # Point class unit tests
└── README.md               # This file
```

## Build Instructions

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the executable
./bin/knn-classifier
```

### Build with tests

```bash
# Configure with tests enabled
cmake -DBUILD_TESTS=ON ..

# Build
cmake --build .

# Run tests
ctest --output-on-failure
# or directly
./tests/tests
```

### Build with specific compiler

```bash
# Use Clang
cmake -DCMAKE_CXX_COMPILER=clang++ ..

# Use GCC
cmake -DCMAKE_CXX_COMPILER=g++ ..
```

## Usage

*(Will be updated as features are implemented)*

```bash
./bin/knn-classifier
```

## Development Progress

- [x] Step 1: Project setup and structure
- [x] Step 2: Point data structure implementation
  - Point class with x, y coordinates and label
  - Euclidean distance calculation
  - Equality operators and stream output
  - Comprehensive unit tests (Google Test)
  - Simple assertion tests in main
- [ ] Step 3: Synthetic dataset generation
- [ ] Step 4: KNN algorithm implementation
- [ ] Step 5: Console visualization
- [ ] Step 6: Interactive mode
- [ ] Step 7: Optimizations
- [ ] Step 8: Final documentation

## Testing

The project includes two levels of testing:

1. **Unit Tests (Google Test)**: Run comprehensive tests with `ctest` or `./tests/tests`
2. **Integration Tests**: Simple assert-based tests in `main.cpp`

## License

MIT License

## Author

Created as part of a C++ machine learning project series.
