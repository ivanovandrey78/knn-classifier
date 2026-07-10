#include <visualizer.hpp>

#include <dataset.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace visualizer {

std::string getColorForLabel(int label, bool useBackground) {
    if (useBackground) {
        switch (label % 6) {
            case 0: return ansi_colors::BG_BLUE;
            case 1: return ansi_colors::BG_RED;
            case 2: return ansi_colors::BG_GREEN;
            case 3: return ansi_colors::BG_YELLOW;
            case 4: return ansi_colors::BG_MAGENTA;
            case 5: return ansi_colors::BG_CYAN;
            default: return ansi_colors::RESET;
        }
    } else {
        switch (label % 8) {
            case 0: return ansi_colors::BRIGHT_BLUE;
            case 1: return ansi_colors::BRIGHT_RED;
            case 2: return ansi_colors::BRIGHT_GREEN;
            case 3: return ansi_colors::BRIGHT_YELLOW;
            case 4: return ansi_colors::BRIGHT_MAGENTA;
            case 5: return ansi_colors::BRIGHT_CYAN;
            case 6: return ansi_colors::CYAN;
            case 7: return ansi_colors::MAGENTA;
            default: return ansi_colors::WHITE;
        }
    }
}

char getCharForLabel(int label) {
    const char chars[] = {'O', '#', '^', '*', '+',
                          'X', 'V', '<', '>', '@'};
    return chars[label % 10];
}

static void applyPadding(
    double& minX, double& maxX, double& minY, double& maxY,
    double padding) {
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double padX = rangeX * padding;
    double padY = rangeY * padding;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
}

static double normalizeRange(double range) {
    return range == 0.0 ? 1.0 : range;
}

static double gridToX(
    int col, int gridSize, double minX, double rangeX) {
    return minX
        + (col / static_cast<double>(gridSize - 1)) * rangeX;
}

static double gridToY(
    int row, int gridSize, double maxY, double rangeY) {
    return maxY
        - (row / static_cast<double>(gridSize - 1)) * rangeY;
}

void drawMap(const KNN& classifier, const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!\n";
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = dataset::getBounds(training);

    applyPadding(minX, maxX, minY, maxY, config.padding);
    double rangeX = normalizeRange(maxX - minX);
    double rangeY = normalizeRange(maxY - minY);

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << ansi_colors::BOLD
              << "Decision Boundary Visualization"
              << ansi_colors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    for (int row = 0; row < config.gridSize; ++row) {
        if (config.showAxes && row == config.gridSize / 2)
            std::cout << "Y ";
        else
            std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);

            Point query(x, y);
            int predictedLabel = classifier.predict(query, config.k);

            if (config.useColors)
                std::cout << getColorForLabel(predictedLabel, true)
                          << " " << ansi_colors::RESET;
            else
                std::cout << getCharForLabel(predictedLabel);
        }
        std::cout << std::endl;
    }

    if (config.showAxes) {
        std::cout << "  ";
        for (int i = 0; i < config.gridSize; ++i)
            std::cout << (i == config.gridSize / 2 ? "X" : " ");
        std::cout << std::endl;
    }

    std::cout << "\nBounds: X[" << std::fixed << std::setprecision(2)
              << minX << ", " << maxX << "], Y[" << minY << ", "
              << maxY << "]" << std::endl;

    if (config.showLegend) {
        std::set<int> uniqueLabels;
        for (const auto& p : training) uniqueLabels.insert(p.label);
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawMapWithData(
    const KNN& classifier, const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!\n";
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = dataset::getBounds(training);

    applyPadding(minX, maxX, minY, maxY, config.padding);
    double rangeX = normalizeRange(maxX - minX);
    double rangeY = normalizeRange(maxY - minY);

    std::vector<std::vector<int>> grid(
        config.gridSize, std::vector<int>(config.gridSize, -1));

    for (int row = 0; row < config.gridSize; ++row) {
        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);
            grid[row][col] = classifier.predict(Point(x, y), config.k);
        }
    }

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << ansi_colors::BOLD
              << "Decision Boundary + Training Data"
              << ansi_colors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    double threshold
        = std::min(rangeX, rangeY) / config.gridSize * 0.5;

    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);

            bool hasPoint = false;
            int pointLabel = -1;
            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold
                    && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                if (config.useColors)
                    std::cout << ansi_colors::BOLD
                              << getColorForLabel(pointLabel)
                              << getCharForLabel(pointLabel)
                              << ansi_colors::RESET;
                else
                    std::cout << getCharForLabel(pointLabel);
            } else {
                if (config.useColors)
                    std::cout << getColorForLabel(grid[row][col], true)
                              << " " << ansi_colors::RESET;
                else
                    std::cout << config.emptyChar;
            }
        }
        std::cout << std::endl;
    }

    if (config.showLegend) {
        std::set<int> uniqueLabels;
        for (const auto& p : training) uniqueLabels.insert(p.label);
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawMapWithQuery(const KNN& classifier, const Point& query,
                      const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!\n";
        return;
    }

    const auto& training = classifier.getTrainingData();
    auto [minX, maxX, minY, maxY] = dataset::getBounds(training);

    minX = std::min(minX, query.x);
    maxX = std::max(maxX, query.x);
    minY = std::min(minY, query.y);
    maxY = std::max(maxY, query.y);

    applyPadding(minX, maxX, minY, maxY, config.padding);
    double rangeX = normalizeRange(maxX - minX);
    double rangeY = normalizeRange(maxY - minY);

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << ansi_colors::BOLD
              << "Query Point Prediction"
              << ansi_colors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);

    auto [predictedLabel, confidence]
        = classifier.predictWithConfidence(query, config.k);
    std::cout << "\nQuery: " << query << std::endl;
    std::cout << "Predicted Class: " << predictedLabel
              << " (confidence: " << std::fixed
              << std::setprecision(1) << (confidence * 100)
              << "%)" << std::endl;
    std::cout << "\n";

    double threshold
        = std::min(rangeX, rangeY) / config.gridSize * 0.5;

    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);

            if (std::abs(query.x - x) < threshold
                && std::abs(query.y - y) < threshold) {
                if (config.useColors)
                    std::cout << ansi_colors::BOLD
                              << ansi_colors::BRIGHT_GREEN << "Q"
                              << ansi_colors::RESET;
                else
                    std::cout << "Q";
                continue;
            }

            bool hasPoint = false;
            int pointLabel = -1;
            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold
                    && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                if (config.useColors)
                    std::cout << getColorForLabel(pointLabel)
                              << getCharForLabel(pointLabel)
                              << ansi_colors::RESET;
                else
                    std::cout << getCharForLabel(pointLabel);
            } else {
                Point gridPoint(x, y);
                int bgLabel = classifier.predict(gridPoint, config.k);
                if (config.useColors)
                    std::cout << getColorForLabel(bgLabel, true)
                              << " " << ansi_colors::RESET;
                else
                    std::cout << config.emptyChar;
            }
        }
        std::cout << std::endl;
    }

    if (config.useColors)
        std::cout << "\n" << ansi_colors::BRIGHT_GREEN << "Q"
                  << ansi_colors::RESET << " = Query Point"
                  << std::endl;
}

void drawMapWithNeighbors(const KNN& classifier, const Point& query,
                          const VisualizerConfig& config) {
    if (!classifier.isReady()) {
        std::cout << "Classifier has no training data!\n";
        return;
    }

    auto neighbors = classifier.getKNearest(query, config.k);
    const auto& training = classifier.getTrainingData();

    auto [minX, maxX, minY, maxY] = dataset::getBounds(training);

    minX = std::min(minX, query.x);
    maxX = std::max(maxX, query.x);
    minY = std::min(minY, query.y);
    maxY = std::max(maxY, query.y);

    applyPadding(minX, maxX, minY, maxY, config.padding);
    double rangeX = normalizeRange(maxX - minX);
    double rangeY = normalizeRange(maxY - minY);

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << ansi_colors::BOLD
              << "K-Nearest Neighbors Visualization (k="
              << config.k << ")" << ansi_colors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);

    auto [predictedLabel, confidence]
        = classifier.predictWithConfidence(query, config.k);
    std::cout << "\nQuery: " << query << std::endl;
    std::cout << "Predicted Class: " << predictedLabel
              << " (confidence: " << std::fixed
              << std::setprecision(1) << (confidence * 100)
              << "%)" << std::endl;
    std::cout << "\n";

    std::set<std::pair<double, double>> neighborSet;
    for (const auto& neighbor : neighbors)
        neighborSet.insert({neighbor.point.x, neighbor.point.y});

    double threshold
        = std::min(rangeX, rangeY) / config.gridSize * 0.5;

    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);

            if (std::abs(query.x - x) < threshold
                && std::abs(query.y - y) < threshold) {
                if (config.useColors)
                    std::cout << ansi_colors::BOLD
                              << ansi_colors::BRIGHT_GREEN << "Q"
                              << ansi_colors::RESET;
                else
                    std::cout << "Q";
                continue;
            }

            bool isNeighbor = false;
            bool hasPoint = false;
            int pointLabel = -1;

            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold
                    && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    if (neighborSet.count({point.x, point.y}))
                        isNeighbor = true;
                    break;
                }
            }

            if (hasPoint) {
                if (isNeighbor) {
                    if (config.useColors)
                        std::cout << ansi_colors::BOLD
                                  << ansi_colors::BRIGHT_YELLOW
                                  << getCharForLabel(pointLabel)
                                  << ansi_colors::RESET;
                    else
                        std::cout << "*";
                } else {
                    if (config.useColors)
                        std::cout << ansi_colors::GRAY
                                  << getCharForLabel(pointLabel)
                                  << ansi_colors::RESET;
                    else
                        std::cout << getCharForLabel(pointLabel);
                }
            } else {
                Point gridPoint(x, y);
                int bgLabel = classifier.predict(gridPoint, config.k);
                if (config.useColors)
                    std::cout << getColorForLabel(bgLabel, true)
                              << " " << ansi_colors::RESET;
                else
                    std::cout << config.emptyChar;
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\n" << ansi_colors::BRIGHT_GREEN << "Q"
              << ansi_colors::RESET << " = Query Point, "
              << ansi_colors::BRIGHT_YELLOW << "★"
              << ansi_colors::RESET << " = K-Nearest Neighbors"
              << std::endl;

    std::cout << "\nNearest Neighbors:" << std::endl;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << neighbors[i].point
                  << " - distance: " << std::setprecision(3)
                  << neighbors[i].distance << std::endl;
    }
}

void drawDataPoints(const std::vector<Point>& training,
                    const VisualizerConfig& config) {
    if (training.empty()) {
        std::cout << "No data points to visualize!\n";
        return;
    }

    auto [minX, maxX, minY, maxY] = dataset::getBounds(training);

    applyPadding(minX, maxX, minY, maxY, config.padding);
    double rangeX = normalizeRange(maxX - minX);
    double rangeY = normalizeRange(maxY - minY);

    std::cout << "\n";
    drawSeparator(config.gridSize + 2, true);
    std::cout << ansi_colors::BOLD
              << "Training Data Scatter Plot"
              << ansi_colors::RESET << std::endl;
    drawSeparator(config.gridSize + 2, false);
    std::cout << "\n";

    double threshold
        = std::min(rangeX, rangeY) / config.gridSize * 0.5;

    for (int row = 0; row < config.gridSize; ++row) {
        std::cout << "  ";

        for (int col = 0; col < config.gridSize; ++col) {
            double x = gridToX(col, config.gridSize, minX, rangeX);
            double y = gridToY(row, config.gridSize, maxY, rangeY);

            bool hasPoint = false;
            int pointLabel = -1;
            for (const auto& point : training) {
                if (std::abs(point.x - x) < threshold
                    && std::abs(point.y - y) < threshold) {
                    hasPoint = true;
                    pointLabel = point.label;
                    break;
                }
            }

            if (hasPoint) {
                if (config.useColors)
                    std::cout << ansi_colors::BOLD
                              << getColorForLabel(pointLabel)
                              << getCharForLabel(pointLabel)
                              << ansi_colors::RESET;
                else
                    std::cout << getCharForLabel(pointLabel);
            } else {
                std::cout << config.emptyChar;
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\nTotal points: " << training.size() << std::endl;
    std::cout << "Bounds: X[" << std::fixed << std::setprecision(2)
              << minX << ", " << maxX << "], Y[" << minY << ", "
              << maxY << "]" << std::endl;

    if (config.showLegend) {
        std::set<int> uniqueLabels;
        for (const auto& p : training) uniqueLabels.insert(p.label);
        drawLegend(uniqueLabels.size(), config.useColors);
    }
}

void drawLegend(int numClasses, bool useColors) {
    std::cout << "\nLegend:" << std::endl;
    for (int i = 0; i < numClasses; ++i) {
        std::cout << "  Class " << i << ": ";
        if (useColors)
            std::cout << getColorForLabel(i) << getCharForLabel(i)
                      << ansi_colors::RESET;
        else
            std::cout << getCharForLabel(i);
        std::cout << std::endl;
    }
}

void drawSeparator(int width, bool thick) {
    for (int i = 0; i < width; ++i)
        std::cout << (thick ? '=' : '-');
    std::cout << std::endl;
}

void clearScreen() {
    std::cout << "\033[2J\033[1;1H";
}

std::string progressBar(double progress, int width) {
    int filledWidth = static_cast<int>(progress * width);
    std::ostringstream oss;

    oss << "[";
    for (int i = 0; i < width; ++i)
        oss << (i < filledWidth ? "█" : "░");
    oss << "] " << std::fixed << std::setprecision(1)
        << (progress * 100) << "%";

    return oss.str();
}

}  // namespace Visualizer
