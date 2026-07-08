#ifndef POINT_HPP
#define POINT_HPP

#include <cmath>
#include <ostream>

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

    /**
     * @brief Equality operator with epsilon comparison for floating point
     * @param other The other point
     * @return true if points are equal within epsilon
     */
    bool operator==(const Point& other) const {
        const double epsilon = 1e-9;
        return std::abs(x - other.x) < epsilon && 
               std::abs(y - other.y) < epsilon && 
               label == other.label;
    }

    /**
     * @brief Inequality operator
     */
    bool operator!=(const Point& other) const {
        return !(*this == other);
    }

    /**
     * @brief Stream output operator for easy printing
     */
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "Point(" << p.x << ", " << p.y << ", label=" << p.label << ")";
        return os;
    }
};

#endif  // POINT_HPP
