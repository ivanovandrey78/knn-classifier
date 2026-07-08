#ifndef POINT_HPP
#define POINT_HPP

#include <cmath>

/**
 * @brief Represents a 2D point with coordinates and class label
 */
class Point {
public:
    double x;
    double y;
    int label;

    Point() : x(0.0), y(0.0), label(0) {}
    Point(double x, double y, int label = 0) : x(x), y(y), label(label) {}

    /**
     * @brief Calculate Euclidean distance to another point
     * @param other The other point
     * @return Distance as double
     */
    double distanceTo(const Point& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

#endif  // POINT_HPP
