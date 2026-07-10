#include <gtest/gtest.h>
#include <point.hpp>
#include <cmath>

/**
 * @brief Test suite for Point class
 */
class PointTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

// Test default constructor
TEST_F(PointTest, DefaultConstructor) {
    Point p;
    EXPECT_DOUBLE_EQ(p.x, 0.0);
    EXPECT_DOUBLE_EQ(p.y, 0.0);
    EXPECT_EQ(p.label, 0);
}

// Test parameterized constructor
TEST_F(PointTest, ParameterizedConstructor) {
    Point p1(3.5, 4.2, 1);
    EXPECT_DOUBLE_EQ(p1.x, 3.5);
    EXPECT_DOUBLE_EQ(p1.y, 4.2);
    EXPECT_EQ(p1.label, 1);

    // Test with default label
    Point p2(1.0, 2.0);
    EXPECT_DOUBLE_EQ(p2.x, 1.0);
    EXPECT_DOUBLE_EQ(p2.y, 2.0);
    EXPECT_EQ(p2.label, 0);
}

// Test distance calculation - basic case
TEST_F(PointTest, DistanceToBasic) {
    Point p1(0.0, 0.0);
    Point p2(3.0, 4.0);
    
    // 3-4-5 triangle, distance should be 5.0
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 5.0);
}

// Test distance calculation - same point
TEST_F(PointTest, DistanceToSamePoint) {
    Point p1(5.5, 7.3);
    Point p2(5.5, 7.3);
    
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 0.0);
}

// Test distance calculation - horizontal distance
TEST_F(PointTest, DistanceToHorizontal) {
    Point p1(0.0, 5.0);
    Point p2(10.0, 5.0);
    
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 10.0);
}

// Test distance calculation - vertical distance
TEST_F(PointTest, DistanceToVertical) {
    Point p1(3.0, 0.0);
    Point p2(3.0, 8.0);
    
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 8.0);
}

// Test distance calculation - negative coordinates
TEST_F(PointTest, DistanceToNegativeCoordinates) {
    Point p1(-3.0, -4.0);
    Point p2(0.0, 0.0);
    
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 5.0);
}

// Test distance calculation - symmetry
TEST_F(PointTest, DistanceSymmetry) {
    Point p1(1.5, 2.5);
    Point p2(4.5, 6.5);
    
    double distance1 = p1.distanceTo(p2);
    double distance2 = p2.distanceTo(p1);
    
    EXPECT_DOUBLE_EQ(distance1, distance2);
}

// Test distance calculation - known value
TEST_F(PointTest, DistanceToKnownValue) {
    Point p1(1.0, 1.0);
    Point p2(4.0, 5.0);
    
    // sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5
    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 5.0);
}

// Test distance calculation - floating point precision
TEST_F(PointTest, DistanceToFloatingPoint) {
    Point p1(0.0, 0.0);
    Point p2(1.0, 1.0);
    
    double distance = p1.distanceTo(p2);
    double expected = std::sqrt(2.0);
    
    EXPECT_NEAR(distance, expected, 1e-9);
}

// Test equality operator
TEST_F(PointTest, EqualityOperator) {
    Point p1(3.0, 4.0, 1);
    Point p2(3.0, 4.0, 1);
    Point p3(3.0, 4.0, 2);  // Different label
    Point p4(3.1, 4.0, 1);  // Different x
    
    EXPECT_TRUE(p1 == p2);
    EXPECT_FALSE(p1 == p3);
    EXPECT_FALSE(p1 == p4);
}

// Test inequality operator
TEST_F(PointTest, InequalityOperator) {
    Point p1(1.0, 2.0, 0);
    Point p2(1.0, 2.0, 0);
    Point p3(1.0, 2.0, 1);
    
    EXPECT_FALSE(p1 != p2);
    EXPECT_TRUE(p1 != p3);
}

// Test with different labels (distance should not depend on label)
TEST_F(PointTest, DistanceIgnoresLabel) {
    Point p1(0.0, 0.0, 0);
    Point p2(3.0, 4.0, 1);
    Point p3(3.0, 4.0, 99);
    
    double distance1 = p1.distanceTo(p2);
    double distance2 = p1.distanceTo(p3);
    
    EXPECT_DOUBLE_EQ(distance1, distance2);
    EXPECT_DOUBLE_EQ(distance1, 5.0);
}