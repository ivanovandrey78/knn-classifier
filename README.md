# KNN Classifier with Console Visualization

A K-Nearest Neighbors classifier implementation in C++20 with ASCII-based console visualization.

## Features

- **Production-Ready KNN Classifier**
  - Optimized algorithm with partial sorting
  - Batch prediction support
  - Confidence scoring
  - K-nearest neighbors retrieval
  
- **Advanced Dataset Generation**
  - Configurable multi-class clusters
  - Normal distribution with custom parameters
  - Train/test splitting with shuffling
  - CSV serialization
  - Data normalization
  
- **Comprehensive Utilities**
  - Accuracy calculation
  - Confusion matrix generation
  - Dataset statistics
  - Performance benchmarking
  
- **Beautiful Console Visualization**
  - ASCII graphics with full ANSI color support
  - Multiple visualization modes:
    - Decision boundaries
    - Training data scatter plots
    - Query point predictions
    - K-nearest neighbors highlighting
  - Interactive mode with real-time updates
  - Support for multi-class datasets (up to 8+ classes)
  - Configurable grid size and parameters
  
- **Testing & Quality**
  - 70+ unit tests with Google Test
  - Edge case coverage
  - Performance tests
  - Visualization regression tests

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

The program offers multiple demo modes:

```bash
./bin/knn-classifier
```

You'll be presented with options:
1. **Full demo** - Complete walkthrough of all features
2. **Visualization demo** - Focus on visualization capabilities
3. **Interactive mode** - Enter custom query points for real-time classification
4. **Quick test** - Fast demonstration

### Interactive Mode

In interactive mode, you can enter query points and see:
- Real-time classification
- K-nearest neighbors highlighted
- Decision boundaries
- Prediction confidence

Example:
```
Enter X Y coordinates (or 'exit'): 10 10
```

The system will display a color-coded map showing:
- 🟦🟥 **Decision boundaries** (background colors)
- **●■▲** Training data points (colored symbols)
- **Q** Your query point (green)
- **★** K-nearest neighbors (yellow highlight)

### Code Examples

See [EXAMPLES.md](EXAMPLES.md) for detailed usage examples including:
- Basic classification
- Custom visualization configs
- Multi-class datasets
- Batch prediction
- And more...

### Visualization Guide

See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for:
- Detailed explanation of visualization modes
- Color reference guide
- Configuration tips
- Interpretation guidelines
- Troubleshooting

## Development Progress

- [x] Step 1: Project setup and structure
- [x] Step 2: Point data structure implementation
  - Point class with x, y coordinates and label
  - Euclidean distance calculation
  - Equality operators and stream output
  - Comprehensive unit tests (Google Test)
  - Simple assertion tests in main
- [x] Step 3: Synthetic dataset generation (Production version)
  - Configurable cluster generation with normal distribution
  - Multi-class dataset support
  - Train/test split with optional shuffling
  - Dataset statistics and bounds calculation
  - CSV save/load functionality
  - Data normalization
  - Reproducible with seed parameter
- [x] Step 4: KNN algorithm implementation (Production version)
  - Optimized prediction with partial_sort
  - Batch prediction support
  - Confidence scoring for predictions
  - K-nearest neighbors retrieval
  - Robust error handling and validation
  - Performance benchmarking
  - Accuracy calculation and confusion matrix
  - Comprehensive unit tests
- [x] Step 5: Console visualization (Production version)
  - ASCII graphics with full ANSI color support
  - Multiple visualization modes:
    - Decision boundary visualization
    - Training data scatter plots
    - Combined boundary + data view
    - Query point highlighting
    - K-nearest neighbors display
  - Configurable grid size and parameters
  - Legend and axis labeling
  - Support for multi-class datasets
  - Interactive visualization mode
  - Beautiful color schemes for up to 8 classes
- [x] Step 6: Interactive mode (Integrated in Step 5)
  - Real-time query point input
  - Dynamic visualization updates
  - K-nearest neighbors highlighting
  - Prediction confidence display
- [ ] Step 7: Optimizations
- [ ] Step 8: Final documentation

## Testing

The project includes comprehensive testing:

1. **Unit Tests (Google Test)**: 70+ tests covering all components
   ```bash
   cmake -DBUILD_TESTS=ON ..
   cmake --build .
   ctest --output-on-failure
   ```

2. **Integration Tests**: Real-world demonstrations in `main.cpp`

3. **Test Coverage**:
   - Point class: Distance calculations, operators
   - Dataset: Generation, splitting, metrics
   - KNN: Predictions, edge cases, accuracy
   - Visualizer: All visualization modes, configs

## What's Next

- [x] Steps 1-6: Core functionality complete ✓
- [ ] Step 7: Performance optimizations
  - Spatial indexing (KD-tree)
  - Parallel batch prediction
  - Memory optimizations
- [ ] Step 8: Final documentation
  - Architecture diagrams
  - API documentation
  - Performance benchmarks

## License

MIT License