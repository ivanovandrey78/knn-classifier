# KNN Classifier - Usage Examples

## Quick Start

### 1. Basic Classification

```cpp
#include <dataset.hpp>
#include <knn.hpp>

// Generate dataset
auto data = Dataset::generateSimpleDataset(100, 42);

// Split into train/test
auto [train, test] = Dataset::trainTestSplit(data, 0.8);

// Train classifier
KNN classifier(train);

// Predict
Point query(10.0, 10.0);
int prediction = classifier.predict(query, 5);
```

### 2. Visualization

```cpp
#include <visualizer.hpp>

// Generate and train
auto data = Dataset::generateSimpleDataset(50, 42);
KNN classifier(data);

// Configure visualization
VisualizerConfig config;
config.gridSize = 40;
config.k = 5;
config.useColors = true;

// Draw decision boundary
Visualizer::drawMap(classifier, config);

// Draw with training data
Visualizer::drawMapWithData(classifier, config);

// Visualize query
Point query(10.0, 10.0);
Visualizer::drawMapWithQuery(classifier, query, config);

// Show k-nearest neighbors
Visualizer::drawMapWithNeighbors(classifier, query, config);
```

### 3. Multi-Class Dataset

```cpp
#include <dataset.hpp>

// Create 3-class dataset
std::vector<ClusterConfig> configs = {
    ClusterConfig(5.0, 5.0, 1.0, 0, 100),    // Class 0
    ClusterConfig(15.0, 15.0, 1.2, 1, 100),  // Class 1
    ClusterConfig(10.0, 20.0, 0.8, 2, 80)    // Class 2
};

auto data = Dataset::generateClusters(configs, 42);
KNN classifier(data);

// Visualize
VisualizerConfig config;
config.gridSize = 50;
Visualizer::drawMapWithData(classifier, config);
```

### 4. Batch Prediction & Evaluation

```cpp
#include <dataset.hpp>
#include <knn.hpp>

auto fullData = Dataset::generateSimpleDataset(200, 42);
auto [train, test] = Dataset::trainTestSplit(fullData, 0.7);

KNN classifier(train);

// Batch predict
auto predictions = classifier.predictBatch(test, 5);

// Get ground truth
std::vector<int> groundTruth;
for (const auto& point : test) {
    groundTruth.push_back(point.label);
}

// Calculate accuracy
double accuracy = Dataset::calculateAccuracy(predictions, groundTruth);
std::cout << "Accuracy: " << accuracy << "%" << std::endl;

// Confusion matrix
auto matrix = Dataset::confusionMatrix(predictions, groundTruth, 2);
```

### 5. Custom Visualization Config

```cpp
VisualizerConfig config;
config.gridSize = 60;           // Higher resolution
config.k = 7;                   // More neighbors
config.useColors = true;        // ANSI colors
config.showLegend = true;       // Show legend
config.showAxes = true;         // Show axes
config.showGrid = false;        // No grid lines
config.emptyChar = '.';         // Custom empty char
config.padding = 0.15;          // 15% padding

Visualizer::drawMap(classifier, config);
```

### 6. Prediction with Confidence

```cpp
Point query(12.0, 8.0);
auto [label, confidence] = classifier.predictWithConfidence(query, 5);

std::cout << "Predicted: Class " << label << std::endl;
std::cout << "Confidence: " << (confidence * 100) << "%" << std::endl;
```

### 7. Get K-Nearest Neighbors

```cpp
Point query(10.0, 10.0);
auto neighbors = classifier.getKNearest(query, 5);

for (size_t i = 0; i < neighbors.size(); ++i) {
    std::cout << "Neighbor " << (i+1) << ": " 
              << neighbors[i].point 
              << " (distance: " << neighbors[i].distance << ")"
              << std::endl;
}
```

### 8. Save/Load Dataset

```cpp
// Save
auto data = Dataset::generateSimpleDataset(100, 42);
Dataset::saveToCSV(data, "my_dataset.csv");

// Load
auto loadedData = Dataset::loadFromCSV("my_dataset.csv");
```

### 9. Data Normalization

```cpp
auto data = Dataset::generateSimpleDataset(100, 42);

// Before normalization
auto [minX, maxX, minY, maxY] = Dataset::getBounds(data);
std::cout << "Before: X[" << minX << ", " << maxX << "]" << std::endl;

// Normalize to [0, 1]
Dataset::normalize(data);

// After normalization
auto [minX2, maxX2, minY2, maxY2] = Dataset::getBounds(data);
std::cout << "After: X[" << minX2 << ", " << maxX2 << "]" << std::endl;
```

### 10. Interactive Loop

```cpp
auto data = Dataset::generateSimpleDataset(100, 42);
KNN classifier(data);

VisualizerConfig config;
config.gridSize = 40;
config.k = 5;

while (true) {
    std::cout << "Enter X Y (or 'quit'): ";
    std::string input;
    std::getline(std::cin, input);
    
    if (input == "quit") break;
    
    std::istringstream iss(input);
    double x, y;
    if (iss >> x >> y) {
        Point query(x, y);
        
        Visualizer::clearScreen();
        Visualizer::drawMapWithNeighbors(classifier, query, config);
    }
}
```

## Color Scheme

The visualizer uses different colors for different classes:

- **Class 0**: Bright Blue
- **Class 1**: Bright Red  
- **Class 2**: Bright Green
- **Class 3**: Bright Yellow
- **Class 4**: Bright Magenta
- **Class 5**: Bright Cyan
- **Class 6**: Cyan
- **Class 7**: Magenta

Special markers:
- **Q** (Green): Query point
- **★** (Yellow): K-nearest neighbors
- **●■▲♦**: Different class symbols

## Performance Tips

1. **Grid Size**: Lower grid size (10-30) for faster rendering
2. **K Value**: Smaller k (1-5) for faster prediction
3. **Dataset Size**: Limit to 500-1000 points for interactive use
4. **Disable Colors**: Set `useColors = false` on non-ANSI terminals

## Common Patterns

### Pattern: Cross-validation

```cpp
// Not implemented yet, but you can do manual k-fold
auto data = Dataset::generateSimpleDataset(200, 42);

for (int fold = 0; fold < 5; ++fold) {
    // Split logic here
    // Train and evaluate
}
```

### Pattern: Optimal K Selection

```cpp
auto [train, test] = Dataset::trainTestSplit(data, 0.7);
KNN classifier(train);

for (int k = 1; k <= 20; ++k) {
    auto predictions = classifier.predictBatch(test, k);
    // Calculate accuracy and find best k
}
```

### Pattern: Decision Boundary Animation

```cpp
for (int k = 1; k <= 10; ++k) {
    Visualizer::clearScreen();
    
    VisualizerConfig config;
    config.k = k;
    config.gridSize = 40;
    
    std::cout << "K = " << k << std::endl;
    Visualizer::drawMap(classifier, config);
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
```

## Building & Running

```bash
# Build
mkdir build && cd build
cmake ..
cmake --build .

# Run main program
./bin/knn-classifier

# Run tests
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest --output-on-failure
```

## Troubleshooting

**Issue**: Colors not showing
- **Solution**: Make sure your terminal supports ANSI colors or set `config.useColors = false`

**Issue**: Slow rendering
- **Solution**: Reduce `gridSize` or limit training data size

**Issue**: Crash on empty dataset
- **Solution**: Check `classifier.isReady()` before visualization