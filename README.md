# KNN Classifier with Console Visualization

A K-Nearest Neighbors classifier implementation in C++20 with ASCII-based console visualization.

## Features

- K-Nearest Neighbors classification algorithm
- Synthetic dataset generation
- Console-based visualization using ASCII graphics and ANSI colors
- Interactive mode for testing predictions

## Requirements

- CMake 3.20 or higher
- C++20 compatible compiler (Clang recommended)
- Linux/macOS/Windows with terminal supporting ANSI colors

## Project Structure

```
.
├── CMakeLists.txt      # Build configuration
├── main.cpp            # Entry point
├── point.hpp           # Point class definition
├── knn.hpp             # KNN classifier header
├── knn.cpp             # KNN classifier implementation
├── visualizer.hpp      # Console visualization header
└── README.md           # This file
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
./knn_classifier
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
./knn_classifier
```

## Development Progress

- [x] Step 1: Project setup and structure
- [ ] Step 2: Point data structure implementation
- [ ] Step 3: Synthetic dataset generation
- [ ] Step 4: KNN algorithm implementation
- [ ] Step 5: Console visualization
- [ ] Step 6: Interactive mode
- [ ] Step 7: Optimizations
- [ ] Step 8: Final documentation

## License

MIT License

## Author

Created as part of a C++ machine learning project series.
