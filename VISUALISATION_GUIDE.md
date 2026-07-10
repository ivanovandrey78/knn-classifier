# Visualization Guide

## Overview

The KNN Classifier includes a powerful ASCII-based visualization system with full ANSI color support. This guide explains the different visualization modes and how to interpret them.

## Visualization Modes

### 1. Decision Boundary (`drawMap`)

Shows the predicted class for every point in the 2D space.

```
==================================================
Decision Boundary Visualization
--------------------------------------------------

  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████
                          ████████████████████████

Bounds: X[2.50, 17.50], Y[2.50, 17.50]
```

**What you see:**
- Blue background = Class 0 regions
- Red background = Class 1 regions
- The boundary shows where the classifier switches between classes

### 2. Decision Boundary + Data (`drawMapWithData`)

Overlays training data points on the decision boundary.

```
==================================================
Decision Boundary + Training Data
--------------------------------------------------

  ████●██●████●███████████                        
  ████████●███●███●████████                       
  ███●████████████████●████                       
  ████████████████████████                        
  █████●██████████●████████                       
  ████████████████████████                        
  ████████████████████████                        
  ████████████████████████                        
                      ███■████■████████■███████████
                      ████████████■████████████████
                      ██████■█████████████████■████
                      ████████████████████████████
                      ████████████████████████████
                      ███████■████████████████████
                      ████████████████████████████
```

**What you see:**
- Background colors = Decision regions
- **●** = Class 0 training points
- **■** = Class 1 training points
- **▲** = Class 2 training points (if present)

### 3. Scatter Plot (`drawDataPoints`)

Shows only the training data points without decision boundaries.

```
==================================================
Training Data Scatter Plot
--------------------------------------------------

    ● ●  ●                                        
   ●    ●●  ●                                     
    ●  ●  ●  ●                                    
    ● ●   ●                                       
   ●  ●     ●                                     
                                                  
                                                  
                                                  
                       ■   ■  ■                   
                        ■  ■ ■  ■                 
                       ■ ■   ■ ■                  
                         ■ ■   ■                  
                        ■  ■  ■                   
                                                  

Total points: 100
Bounds: X[2.50, 17.50], Y[2.50, 17.50]
```

**What you see:**
- Clean visualization of data clusters
- Different symbols for different classes
- No classification information

### 4. Query Point Prediction (`drawMapWithQuery`)

Highlights a specific query point and shows its prediction.

```
==================================================
Query Point Prediction
--------------------------------------------------

Query: Point(10.0, 10.0, label=0)
Predicted Class: 1 (confidence: 80.0%)

  ████●██●████●███████████                        
  ████████●███●███●████████                       
  ███●████████████████●████                       
  ████████████████████████                        
  █████●██████████●████████                       
  ████████Q███████████████                        ← Query point here
  ████████████████████████                        
  ████████████████████████                        
                      ███■████■████████■███████████
                      ████████████■████████████████
                      ██████■█████████████████■████
```

**What you see:**
- **Q** (green) = Your query point
- Background shows which region it falls into
- Confidence score tells you how certain the prediction is

### 5. K-Nearest Neighbors (`drawMapWithNeighbors`)

Shows the k-nearest neighbors used for classification.

```
==================================================
K-Nearest Neighbors Visualization (k=5)
--------------------------------------------------

Query: Point(10.0, 10.0, label=0)
Predicted Class: 1 (confidence: 60.0%)

  ████●██●████●███████████                        
  ████████●███●███●████████                       
  ███●████████████████●████                       
  ████████████████████████                        
  █████●██★███████●████████                       ← Neighbor 1
  ████████Q███★███████████                        ← Query + Neighbor 2
  ████████████★███████████                        ← Neighbor 3
  ████████████████████████                        
                      ███★████■████████■███████████  ← Neighbor 4
                      ████████████★████████████████  ← Neighbor 5
                      ██████■█████████████████■████

Q = Query Point, ★ = K-Nearest Neighbors

Nearest Neighbors:
  1. Point(9.5, 11.2, label=1) - distance: 1.500
  2. Point(11.3, 10.8, label=1) - distance: 1.623
  3. Point(10.7, 8.9, label=1) - distance: 1.822
  4. Point(8.2, 9.5, label=0) - distance: 1.944
  5. Point(11.8, 11.5, label=0) - distance: 2.245
```

**What you see:**
- **Q** (green) = Query point
- **★** (yellow) = The k neighbors used for voting
- Gray points = Other training data
- List of neighbors with distances

## Color Reference

### Class Colors (Foreground)
- Class 0: **Bright Blue** (●)
- Class 1: **Bright Red** (■)
- Class 2: **Bright Green** (▲)
- Class 3: **Bright Yellow** (♦)
- Class 4: **Bright Magenta** (★)
- Class 5: **Bright Cyan** (◆)

### Background Colors
- Class 0 regions: Blue background
- Class 1 regions: Red background
- Class 2 regions: Green background
- Class 3 regions: Yellow background

### Special Markers
- **Q** in bright green: Query/test point
- **★** in yellow: Nearest neighbors
- Gray symbols: Non-neighbor training points

## Configuration Options

### Grid Size
```cpp
config.gridSize = 40;  // Default
// Higher = more detail, slower rendering
// Lower = less detail, faster rendering
// Range: 10-80 recommended
```

### K Parameter
```cpp
config.k = 5;  // Default
// Number of neighbors to consider
// Affects both prediction and neighbor highlighting
```

### Colors
```cpp
config.useColors = true;   // Enable ANSI colors
config.useColors = false;  // ASCII-only mode (for logs/non-color terminals)
```

### Legend & Axes
```cpp
config.showLegend = true;  // Show class legend
config.showAxes = true;    // Show X/Y axis markers
config.showGrid = false;   // Grid lines (not implemented)
```

### Padding
```cpp
config.padding = 0.1;  // 10% padding around data
// Higher = more empty space around clusters
// Lower = tighter fit to data
```

### Empty Character
```cpp
config.emptyChar = ' ';  // Space (default)
config.emptyChar = '.';  // Dots for scatter plots
config.emptyChar = '·';  // Middle dots
```

## Interpreting Visualizations

### Decision Boundaries

The boundary between different colored regions shows where the classifier changes its prediction. 

- **Smooth boundaries** = Stable classification
- **Jagged boundaries** = May indicate overfitting or small k
- **Overlapping regions** = Difficult classification area

### Confidence Scores

```
Predicted Class: 1 (confidence: 100.0%)  ← All neighbors agree
Predicted Class: 0 (confidence: 60.0%)   ← 3/5 neighbors agree
Predicted Class: 2 (confidence: 40.0%)   ← 2/5 neighbors agree (tie-break)
```

High confidence (>80%): Strong prediction
Medium confidence (60-80%): Moderate certainty
Low confidence (<60%): Uncertain region, close to boundary

### Cluster Quality

Good clusters:
- Tight groupings of same-class points
- Clear separation between classes
- Smooth decision boundaries

Poor clusters:
- Scattered points
- Overlapping classes
- Irregular boundaries

## Tips for Effective Visualization

### 1. Start Simple
```cpp
// Begin with default config
VisualizerConfig config;
Visualizer::drawMap(classifier, config);
```

### 2. Adjust Resolution
```cpp
config.gridSize = 60;  // High detail for publication
config.gridSize = 30;  // Fast preview
config.gridSize = 20;  // Very fast, low detail
```

### 3. Compare Different K Values
```cpp
for (int k : {1, 3, 5, 7, 10}) {
    config.k = k;
    std::cout << "K = " << k << std::endl;
    Visualizer::drawMap(classifier, config);
}
```

### 4. Use Right Mode for Task

| Task | Mode | Function |
|------|------|----------|
| Understand data | Scatter plot | `drawDataPoints()` |
| See classification regions | Decision boundary | `drawMap()` |
| Verify predictions | Query point | `drawMapWithQuery()` |
| Debug classification | K-neighbors | `drawMapWithNeighbors()` |
| Full picture | Combined | `drawMapWithData()` |

### 5. Terminal Compatibility

Most modern terminals support ANSI colors:
- ✅ Linux: Gnome Terminal, Konsole, xterm
- ✅ macOS: Terminal.app, iTerm2
- ✅ Windows: Windows Terminal, ConEmu
- ⚠️ Windows CMD: Limited support (use Windows Terminal)
- ❌ Logs/files: Use `config.useColors = false`

## Advanced Patterns

### Animation
```cpp
for (double x = 0; x <= 20; x += 0.5) {
    Point query(x, 10.0);
    Visualizer::clearScreen();
    Visualizer::drawMapWithQuery(classifier, query, config);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}
```

### Side-by-Side Comparison
```cpp
std::cout << "With k=1:" << std::endl;
config.k = 1;
Visualizer::drawMap(classifier, config);

std::cout << "\nWith k=10:" << std::endl;
config.k = 10;
Visualizer::drawMap(classifier, config);
```

### Grid Search Visualization
```cpp
for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 5; ++col) {
        double x = minX + col * (maxX - minX) / 4;
        double y = minY + row * (maxY - minY) / 4;
        Point query(x, y);
        
        auto [label, conf] = classifier.predictWithConfidence(query, 5);
        std::cout << "(" << x << "," << y << "): Class " 
                  << label << " (" << (conf*100) << "%)" << std::endl;
    }
}
```

## Troubleshooting

### Colors not displaying
```cpp
// Force disable colors
config.useColors = false;
```

### Too slow
```cpp
// Reduce grid size
config.gridSize = 20;
// Or reduce dataset
auto [train, _] = Dataset::trainTestSplit(data, 0.1);
```

### Hard to read
```cpp
// Increase grid size
config.gridSize = 60;
// Increase terminal font size
// Use high contrast terminal theme
```

### Garbled output
```cpp
// Check terminal encoding is UTF-8
// Some symbols may not work on all terminals
// Fall back to ASCII mode if needed
```

## Examples Gallery

See the main program for live examples:
- 2-class classification
- 4-class classification  
- Interactive mode
- Neighbor visualization
- Decision boundary analysis