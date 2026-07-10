#include <visualizer.hpp>

#include <dataset.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace Visualizer {

std::string getColorForLabel(int label, bool useBackground) {
    if (useBackground) {
        switch (label % 6) {
            case 0: return AnsiColors::BG_BLUE;
            case 1: return AnsiColors::BG_RED;
            case 2: return AnsiColors::BG_GREEN;
            case 3: return AnsiColors::BG_YELLOW;
            case 4: return AnsiColors::BG_MAGENTA;
            case 5: return AnsiColors::BG_CYAN;
            default: return AnsiColors::RESET;
        }
    } else {
        switch (label % 8) {
            case 0: return AnsiColors::BRIGHT_BLUE;
            case 1: return AnsiColors::BRIGHT_RED;
            case 2: return AnsiColors::BRIGHT_GREEN;
            case 3: return AnsiColors::BRIGHT_YELLOW;
            case 4: return AnsiColors::BRIGHT_MAGENTA;
            case 5: return AnsiColors::BRIGHT_CYAN;
            case 6: return AnsiColors::CYAN;
            case 7: return AnsiColors::MAGENTA;
            default: return AnsiColors::WHITE;
        }
    }
}

const char* getCharForLabel(int label) {
#ifdef _WIN32
    static const char* symbols[] = {
        "O", "#", "^", "*", "+", "X", "V", "<", ">", "@"
    };
#else
    static const char* symbols[] = {
        "●", "■", "▲", "♦", "★", "◆", "▼", "◄", "►", "♥"
    };
#endif
    return symbols[label % 10];
}

void drawMap(const KNN& classifier, const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!" << std::endl;
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = Dataset::getBounds(training);

    // Add padding
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * config.padding;
    double padY = rangeY * config.padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    rangeX = maxX - minX;
    rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << AnsiColors::BOLD << "Decision Boundary Visualization" << AnsiColors::RESET
              << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    // Draw grid from top to bottom (Y decreases)
    for (int row = 0; row < config.gridSize; ++row) {
        if (config.showAxes && row == config.gridSize / 2) {
            std::cout << "Y ";
        } else {
            std::cout << "  ";
        }

        for (int col = 0; col < config.gridSize; ++col) {
            // Map grid position to data coordinates
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;

            Point query(x, y);
            int predictedLabel = classifier.predict(query, config.k);

            if (config.useColors) {
                std::cout << getColorForLabel(predictedLabel, true) << " "
                          << AnsiColors::RESET;
            } else {
                std::cout << getCharForLabel(predictedLabel);
            }
        }
        std::cout << std::endl;
    }

    if (config.showAxes) {
        std::cout << "  ";
        for (int i = 0; i < config.gridSize; ++i) {
            if (i == config.gridSize / 2) {
                std::cout << "X";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\nBounds: X[" << std::fixed << std::setprecision(2) << minX << ", " << maxX
              << "], Y[" << minY << ", " << maxY << "]" << std::endl;

    if (config.showLegend) {
        // Count unique labels
        std::set<int> uniqueLabels;
        for (const auto& p : training) {
            uniqueLabels.insert(p.label);
        }
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawMapWithData(const KNN& classifier, const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!" << std::endl;
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = Dataset::getBounds(training);

    // Add padding
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * config.padding;
    double padY = rangeY * config.padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    rangeX = maxX - minX;
    rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    // Create grid with decision boundaries
    std::vector<std::vector<int>> grid(config.gridSize, std::vector<int>(config.gridSize, -1));

    for (int row = 0; row < config.gridSize; ++row) {
        for (int col = 0; col < config.gridSize; ++col) {
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;
            Point query(x, y);
            grid[row][col] = classifier.predict(query, config.k);
        }
    }

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << AnsiColors::BOLD << "Decision Boundary + Training Data" << AnsiColors::RESET
              << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    // Draw with training points overlaid
    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;

            // Check if any training point is close to this grid cell
            bool hasPoint = false;
            int pointLabel = -1;
            double threshold = std::min(rangeX, rangeY) / config.gridSize * 0.5;

            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                // Draw training point
                if (config.useColors) {
                    std::cout << AnsiColors::BOLD << getColorForLabel(pointLabel)
                              << getCharForLabel(pointLabel) << AnsiColors::RESET;
                } else {
                    std::cout << getCharForLabel(pointLabel);
                }
            } else {
                // Draw background (decision boundary)
                if (config.useColors) {
                    std::cout << getColorForLabel(grid[row][col], true) << " "
                              << AnsiColors::RESET;
                } else {
                    std::cout << config.emptyChar;
                }
            }
        }
        std::cout << std::endl;
    }

    if (config.showLegend) {
        std::set<int> uniqueLabels;
        for (const auto& p : training) {
            uniqueLabels.insert(p.label);
        }
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawMapWithQuery(const KNN& classifier, const Point& query,
                      const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!" << std::endl;
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = Dataset::getBounds(training);

    // Include query point in bounds
    minX = std::min(minX, query.x);
    maxX = std::max(maxX, query.x);
    minY = std::min(minY, query.y);
    maxY = std::max(maxY, query.y);

    // Add padding
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * config.padding;
    double padY = rangeY * config.padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    rangeX = maxX - minX;
    rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << AnsiColors::BOLD << "Query Point Prediction" << AnsiColors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);

    auto [predictedLabel, confidence] = classifier.predictWithConfidence(query, config.k);
    std::cout << "\nQuery: " << query << std::endl;
    std::cout << "Predicted Class: " << predictedLabel << " (confidence: " << std::fixed
              << std::setprecision(1) << (confidence * 100) << "%)" << std::endl;
    std::cout << "\n";

    // Draw grid
    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;

            double threshold = std::min(rangeX, rangeY) / config.gridSize * 0.5;

            // Check for query point
            if (std::abs(query.x - x) < threshold && std::abs(query.y - y) < threshold) {
                if (config.useColors) {
                    std::cout << AnsiColors::BOLD << AnsiColors::BRIGHT_GREEN << "Q"
                              << AnsiColors::RESET;
                } else {
                    std::cout << "Q";
                }
                continue;
            }

            // Check for training points
            bool hasPoint = false;
            int pointLabel = -1;
            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                if (config.useColors) {
                    std::cout << getColorForLabel(pointLabel) << getCharForLabel(pointLabel)
                              << AnsiColors::RESET;
                } else {
                    std::cout << getCharForLabel(pointLabel);
                }
            } else {
                // Background
                Point gridPoint(x, y);
                int bgLabel = classifier.predict(gridPoint, config.k);
                if (config.useColors) {
                    std::cout << getColorForLabel(bgLabel, true) << " " << AnsiColors::RESET;
                } else {
                    std::cout << config.emptyChar;
                }
            }
        }
        std::cout << std::endl;
    }

    if (config.useColors) {
        std::cout << "\n" << AnsiColors::BRIGHT_GREEN << "Q" << AnsiColors::RESET
                  << " = Query Point" << std::endl;
    }
}

void drawMapWithNeighbors(const KNN& classifier, const Point& query,
                          const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!" << std::endl;
        return;
    }

    auto neighbors = classifier.getKNearest(query, config.k);
    const auto& training = classifier.getTrainingData();

    auto [minX, maxX, minY, maxY] = Dataset::getBounds(training);

    // Include query point
    minX = std::min(minX, query.x);
    maxX = std::max(maxX, query.x);
    minY = std::min(minY, query.y);
    maxY = std::max(maxY, query.y);

    // Add padding
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * config.padding;
    double padY = rangeY * config.padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    rangeX = maxX - minX;
    rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << AnsiColors::BOLD << "K-Nearest Neighbors Visualization (k=" << config.k << ")"
              << AnsiColors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);

    auto [predictedLabel, confidence] = classifier.predictWithConfidence(query, config.k);
    std::cout << "\nQuery: " << query << std::endl;
    std::cout << "Predicted Class: " << predictedLabel << " (confidence: " << std::fixed
              << std::setprecision(1) << (confidence * 100) << "%)" << std::endl;
    std::cout << "\n";

    // Create set of neighbor points for quick lookup
    std::set<std::pair<double, double>> neighborSet;
    for (const auto& neighbor : neighbors) {
        neighborSet.insert({neighbor.point.x, neighbor.point.y});
    }

    // Draw grid
    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;

            double threshold = std::min(rangeX, rangeY) / config.gridSize * 0.5;

            // Check for query point
            if (std::abs(query.x - x) < threshold && std::abs(query.y - y) < threshold) {
                if (config.useColors) {
                    std::cout << AnsiColors::BOLD << AnsiColors::BRIGHT_GREEN << "Q"
                              << AnsiColors::RESET;
                } else {
                    std::cout << "Q";
                }
                continue;
            }

            // Check for neighbor or other training points
            bool isNeighbor = false;
            bool hasPoint = false;
            int pointLabel = -1;

            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    if (neighborSet.count({point.x, point.y})) {
                        isNeighbor = true;
                    }
                    break;
                }
            }

            if (hasPoint) {
                if (isNeighbor) {
                    // Highlight neighbors
                    if (config.useColors) {
                        std::cout << AnsiColors::BOLD << AnsiColors::BRIGHT_YELLOW
                                  << getCharForLabel(pointLabel) << AnsiColors::RESET;
                    } else {
                        std::cout << "*";
                    }
                } else {
                    // Regular training point
                    if (config.useColors) {
                        std::cout << AnsiColors::GRAY << getCharForLabel(pointLabel)
                                  << AnsiColors::RESET;
                    } else {
                        std::cout << getCharForLabel(pointLabel);
                    }
                }
            } else {
                // Background
                Point gridPoint(x, y);
                int bgLabel = classifier.predict(gridPoint, config.k);
                if (config.useColors) {
                    std::cout << getColorForLabel(bgLabel, true) << " " << AnsiColors::RESET;
                } else {
                    std::cout << config.emptyChar;
                }
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n" << AnsiColors::BRIGHT_GREEN << "Q" << AnsiColors::RESET
              << " = Query Point, " << AnsiColors::BRIGHT_YELLOW << "★" << AnsiColors::RESET
              << " = K-Nearest Neighbors" << std::endl;

    std::cout << "\nNearest Neighbors:" << std::endl;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << neighbors[i].point << " - distance: "
                  << std::setprecision(3) << neighbors[i].distance << std::endl;
    }
}

void drawDataPoints(const std::vector<Point>& training, const VisualizerConfig& config) {
    if (training.empty()) {
        std::cout << "No data points to visualize!" << std::endl;
        return;
    }

    auto [minX, maxX, minY, maxY] = Dataset::getBounds(training);

    // Add padding
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * config.padding;
    double padY = rangeY * config.padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    rangeX = maxX - minX;
    rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << AnsiColors::BOLD << "Training Data Scatter Plot" << AnsiColors::RESET
              << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    // Draw grid
    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = minX + (col / static_cast<double>(config.gridSize - 1)) * rangeX;
            double y = maxY - (row / static_cast<double>(config.gridSize - 1)) * rangeY;

            double threshold = std::min(rangeX, rangeY) / config.gridSize * 0.5;

            // Check for training points
            bool hasPoint = false;
            int pointLabel = -1;
            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                if (config.useColors) {
                    std::cout << AnsiColors::BOLD << getColorForLabel(pointLabel)
                              << getCharForLabel(pointLabel) << AnsiColors::RESET;
                } else {
                    std::cout << getCharForLabel(pointLabel);
                }
            } else {
                std::cout << config.emptyChar;
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\nTotal points: " << training.size() << std::endl;
    std::cout << "Bounds: X[" << std::fixed << std::setprecision(2) << minX << ", " << maxX
              << "], Y[" << minY << ", " << maxY << "]" << std::endl;

    if (config.showLegend) {
        std::set<int> uniqueLabels;
        for (const auto& p : training) {
            uniqueLabels.insert(p.label);
        }
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawLegend(int numClasses, bool useColors) {
    std::cout << "\nLegend:" << std::endl;
    for (int i = 0; i < numClasses; ++i) {
        std::cout << "  Class " << i << ": ";
        if (useColors) {
            std::cout << getColorForLabel(i) << getCharForLabel(i) << AnsiColors::RESET;
        } else {
            std::cout << getCharForLabel(i);
        }
        std::cout << std::endl;
    }
}

void drawSeparator(int width, bool thick) {
    char ch = thick ? '=' : '-';
    for (int i = 0; i < width; ++i) {
        std::cout << ch;
    }
    std::cout << std::endl;
}

void clearScreen() {
    // ANSI escape code to clear screen
    std::cout << "\033[2J\033[1;1H";
}

std::string progressBar(double progress, int width) {
    int filledWidth = static_cast<int>(progress * width);
    std::ostringstream oss;

    oss << "[";
    for (int i = 0; i < width; ++i) {
        if (i < filledWidth) {
            oss << "█";
        } else {
            oss << "░";
        }
    }
    oss << "] " << std::fixed << std::setprecision(1) << (progress * 100) << "%";

    return oss.str();
}

}  // namespace Visualizer