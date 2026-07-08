#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include <vector>
#include "point.hpp"

/**
 * @brief ASCII console visualization for KNN classifier
 */
namespace Visualizer {

/**
 * @brief Draw a map of the classification space
 * @param training Training dataset
 * @param gridSize Size of the grid (default 20x20)
 */
void drawMap(const std::vector<Point>& training, int gridSize = 20);

}  // namespace Visualizer

#endif  // VISUALIZER_HPP
