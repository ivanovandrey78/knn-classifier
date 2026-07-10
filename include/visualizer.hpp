#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include <knn.hpp>
#include <point.hpp>
#include <string>
#include <vector>

/**
 * @brief ANSI color codes for terminal output
 */
namespace ansi_colors {
const std::string RESET = "\033[0m";
const std::string BOLD = "\033[1m";

// Foreground colors
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string WHITE = "\033[37m";
const std::string GRAY = "\033[90m";

// Background colors
const std::string BG_RED = "\033[41m";
const std::string BG_GREEN = "\033[42m";
const std::string BG_YELLOW = "\033[43m";
const std::string BG_BLUE = "\033[44m";
const std::string BG_MAGENTA = "\033[45m";
const std::string BG_CYAN = "\033[46m";

// Bright colors
const std::string BRIGHT_RED = "\033[91m";
const std::string BRIGHT_GREEN = "\033[92m";
const std::string BRIGHT_YELLOW = "\033[93m";
const std::string BRIGHT_BLUE = "\033[94m";
const std::string BRIGHT_MAGENTA = "\033[95m";
const std::string BRIGHT_CYAN = "\033[96m";
}  // namespace ansi_colors

/**
 * @brief Visualization options
 */
struct VisualizerConfig {
    int gridSize = 40;         // Grid resolution
    int k = 5;                 // K for KNN predictions
    bool useColors = true;     // Use ANSI colors
    bool showLegend = true;    // Show legend
    bool showAxes = true;      // Show axis labels
    bool showGrid = false;     // Show grid lines
    char emptyChar = ' ';      // Character for empty space
    double padding = 0.1;      // Padding around data (0.0-1.0)
};

/**
 * @brief ASCII console visualization for KNN classifier
 */
namespace visualizer {

/**
 * @brief Get color for a class label
 * @param label Class label
 * @param useBackground Use background color instead of foreground
 * @return ANSI color code
 */
std::string getColorForLabel(int label, bool useBackground = false);

/**
 * @brief Get character symbol for a class label
 * @param label Class label
 * @return Character to represent the class
 */
char getCharForLabel(int label);

/**
 * @brief Draw a map of the classification space (decision boundary)
 * @param classifier Trained KNN classifier
 * @param config Visualization configuration
 */
void drawMap(const KNN& classifier, const VisualizerConfig& config = VisualizerConfig());

/**
 * @brief Draw map with training data points overlaid
 * @param classifier Trained KNN classifier
 * @param config Visualization configuration
 */
void drawMapWithData(const KNN& classifier, const VisualizerConfig& config = VisualizerConfig());

/**
 * @brief Draw map with a highlighted query point
 * @param classifier Trained KNN classifier
 * @param query Query point to highlight
 * @param config Visualization configuration
 */
void drawMapWithQuery(const KNN& classifier, const Point& query,
                      const VisualizerConfig& config = VisualizerConfig());

/**
 * @brief Draw map with query point and its k-nearest neighbors
 * @param classifier Trained KNN classifier
 * @param query Query point
 * @param config Visualization configuration
 */
void drawMapWithNeighbors(const KNN& classifier, const Point& query,
                          const VisualizerConfig& config = VisualizerConfig());

/**
 * @brief Draw only the training data points (scatter plot)
 * @param training Training dataset
 * @param config Visualization configuration
 */
void drawDataPoints(const std::vector<Point>& training,
                    const VisualizerConfig& config = VisualizerConfig());

/**
 * @brief Draw a legend explaining the visualization
 * @param numClasses Number of different classes
 * @param useColors Whether colors are enabled
 */
void drawLegend(int numClasses, bool useColors = true);

/**
 * @brief Print a horizontal separator line
 * @param width Width of the line
 * @param thick Use thick line
 */
void drawSeparator(int width = 50, bool thick = false);

/**
 * @brief Clear the console screen
 */
void clearScreen();

/**
 * @brief Create a simple ASCII progress bar
 * @param progress Current progress (0.0 to 1.0)
 * @param width Width of the bar
 * @return String representing the progress bar
 */
std::string progressBar(double progress, int width = 40);

}  // namespace Visualizer

#endif  // VISUALIZER_HPP