# Contributing to KNN Classifier

Thank you for your interest in contributing! This document provides guidelines and best practices.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Keep discussions technical and on-topic

---

## Getting Started

### Prerequisites
- C++20 compatible compiler (Clang 12+, GCC 10+, MSVC 2019+)
- CMake 3.20+
- Git
- Google Test (for testing, fetched automatically)

### Fork and Clone
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/knn-classifier.git
cd knn-classifier
```

### Build
```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest
```

---

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/my-awesome-feature
# or
git checkout -b fix/bug-description
```

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation only
- `test/` - Test improvements
- `perf/` - Performance improvements

### 2. Make Changes
- Write code following our standards (see below)
- Add tests for new functionality
- Update documentation
- Run formatters and linters

### 3. Commit
```bash
# Format code
clang-format -i src/*.cpp include/*.hpp

# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add weighted KNN support"
```

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `build`: Build system changes
- `ci`: CI/CD changes

**Examples**:
```
feat(knn): add Gaussian weight strategy

Implement Gaussian kernel weighting for KNN predictions.
Weight = exp(-distance^2 / (2*sigma^2))

Closes #42
```

```
fix(visualizer): correct color mapping for >8 classes

Previous implementation would crash with >8 classes.
Now cycles through color palette.

Fixes #57
```

### 4. Push and Create PR
```bash
git push origin feature/my-awesome-feature
# Create Pull Request on GitHub
```

---

## Coding Standards

### C++ Style Guide
We follow a modified **Google C++ Style Guide**.

#### Naming Conventions
```cpp
// Classes: PascalCase
class KNNOptimized { };

// Functions: camelCase
void calculateDistance();

// Variables: camelCase
int numNeighbors;

// Constants: UPPER_CASE
const int MAX_ITERATIONS = 100;

// Namespaces: PascalCase
namespace CrossValidation { }

// Files: snake_case
knn_optimized.cpp
knn_optimized.hpp
```

#### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line Length**: 100 characters max
- **Braces**: Attached (K&R style)
  ```cpp
  if (condition) {
      // code
  }
  ```

- **Pointers/References**: Left-aligned
  ```cpp
  int* ptr;
  const Point& ref;
  ```

#### Code Example
```cpp
/**
 * @brief Calculate weighted vote for KNN prediction
 * @param query The query point
 * @param k Number of neighbors
 * @return Predicted class label
 */
int KNNOptimized::predictWeighted(const Point& query, int k) const {
    auto neighbors = getKNearest(query, k);
    if (neighbors.empty()) {
        return 0;
    }

    std::map<int, double> votes;
    for (const auto& neighbor : neighbors) {
        double weight = calculateWeight(neighbor.distance);
        votes[neighbor.label] += weight;
    }

    return findMaxVote(votes);
}
```

### Modern C++ Features
✅ **Use**:
- `auto` for complex types
- Range-based for loops
- Smart pointers (`unique_ptr`, `shared_ptr`)
- `std::optional` for nullable returns
- Structured bindings
- `constexpr` where applicable
- `= default` / `= delete` for special members

❌ **Avoid**:
- Raw `new` / `delete`
- C-style casts
- `NULL` (use `nullptr`)
- Manual memory management
- Global variables

### Header Guards
```cpp
#ifndef MODULE_NAME_HPP
#define MODULE_NAME_HPP

// Content

#endif  // MODULE_NAME_HPP
```

### Includes Order
```cpp
// 1. Corresponding header (for .cpp files)
#include <module_name.hpp>

// 2. C++ standard library
#include <vector>
#include <algorithm>

// 3. Third-party libraries
#include <gtest/gtest.h>

// 4. Project headers
#include <point.hpp>
#include <knn.hpp>
```

---

## Testing Requirements

### Unit Tests
All new features must have unit tests.

#### Example Test
```cpp
#include <gtest/gtest.h>
#include <knn_optimized.hpp>

class KNNOptimizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
        data = Dataset::generateSimpleDataset(50, 42);
        classifier.setTrainingData(data);
    }

    std::vector<Point> data;
    KNNOptimized classifier;
};

TEST_F(KNNOptimizedTest, WeightedPrediction) {
    classifier.setWeightStrategy(WeightStrategy::INVERSE_DISTANCE);
    
    Point query(5.0, 5.0);
    int prediction = classifier.predict(query, 5);
    
    EXPECT_EQ(prediction, 0);
}
```

### Test Coverage
Aim for:
- **Statements**: >80%
- **Branches**: >70%
- **Functions**: >90%

### Running Tests
```bash
cd build
ctest --output-on-failure
# or
./tests/tests --gtest_filter=KNN*
```

### Performance Tests
Add benchmarks for performance-critical code:
```cpp
auto result = Benchmark::measure("New Algorithm", [&]() {
    // Your code
}, 100);
```

---

## Documentation

### Code Documentation
Use Doxygen-style comments:

```cpp
/**
 * @brief Short description
 * 
 * Longer description explaining the function's purpose,
 * algorithm, and usage.
 * 
 * @param param1 Description of param1
 * @param param2 Description of param2
 * @return Description of return value
 * @throws std::exception What exceptions can be thrown
 * 
 * @note Any special notes
 * @warning Important warnings
 * 
 * Example:
 * @code
 * KNN classifier(data);
 * int prediction = classifier.predict(query, 5);
 * @endcode
 */
int functionName(Type param1, Type param2);
```

### README Updates
Update README.md for:
- New features
- API changes
- New dependencies
- Breaking changes

### EXAMPLES.md
Add usage examples for new features.

### ARCHITECTURE.md
Update architecture docs for:
- New components
- Modified data flows
- New design patterns

---

## Pull Request Process

### Before Submitting
1. ✅ Code compiles without warnings
2. ✅ All tests pass
3. ✅ Code is formatted (`clang-format`)
4. ✅ Documentation is updated
5. ✅ Benchmarks run (if performance-related)
6. ✅ No merge conflicts with main

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass locally
- [ ] Benchmarks run (if applicable)

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No compiler warnings
- [ ] EXAMPLES.md updated (if needed)

## Related Issues
Closes #123
```

### Review Process
1. Automated checks (CI)
2. Code review by maintainer(s)
3. Address feedback
4. Approval and merge

### What We Look For
- **Correctness**: Does it work?
- **Testing**: Is it tested?
- **Clarity**: Is the code readable?
- **Performance**: Does it maintain/improve performance?
- **Documentation**: Is it documented?

---

## Project Structure

```
knn-classifier/
├── include/          # Public headers
│   ├── point.hpp
│   ├── knn.hpp
│   ├── knn_optimized.hpp
│   ├── kdtree.hpp
│   ├── dataset.hpp
│   ├── visualizer.hpp
│   └── benchmark.hpp
├── src/              # Implementation
│   ├── knn.cpp
│   ├── knn_optimized.cpp
│   ├── kdtree.cpp
│   ├── dataset.cpp
│   ├── visualizer.cpp
│   └── benchmark.cpp
├── bin/              # Executable source
│   └── main.cpp
├── tests/            # Unit tests
│   ├── point_ut.cpp
│   ├── knn_ut.cpp
│   ├── dataset_ut.cpp
│   └── visualizer_ut.cpp
├── docs/             # Additional documentation
├── .clang-format     # Code formatting rules
├── CMakeLists.txt    # Build configuration
└── README.md         # Main documentation
```

---

## Code Review Checklist

### For Reviewers
- [ ] Code follows style guide
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] No unnecessary complexity
- [ ] Performance is acceptable
- [ ] Security considerations addressed
- [ ] Error handling is robust

### For Authors
- [ ] Self-reviewed code
- [ ] Tested edge cases
- [ ] Considered performance impact
- [ ] Updated relevant docs
- [ ] Followed commit conventions

---

## Common Tasks

### Adding a New Feature

1. **Create branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Add header declaration** (`include/my_feature.hpp`)
   ```cpp
   #ifndef MY_FEATURE_HPP
   #define MY_FEATURE_HPP
   
   class MyFeature {
   public:
       void doSomething();
   };
   
   #endif
   ```

3. **Add implementation** (`src/my_feature.cpp`)
   ```cpp
   #include <my_feature.hpp>
   
   void MyFeature::doSomething() {
       // Implementation
   }
   ```

4. **Add tests** (`tests/my_feature_ut.cpp`)
   ```cpp
   #include <gtest/gtest.h>
   #include <my_feature.hpp>
   
   TEST(MyFeatureTest, BasicTest) {
       MyFeature feature;
       EXPECT_NO_THROW(feature.doSomething());
   }
   ```

5. **Update CMake** (`src/CMakeLists.txt`)
   ```cmake
   add_library(${PROJECT_NAME}_lib STATIC
       ...
       my_feature.cpp
   )
   ```

6. **Document** (README, EXAMPLES, etc.)

### Fixing a Bug

1. **Write failing test** that reproduces the bug
2. **Fix the bug**
3. **Verify test passes**
4. **Add regression test**
5. **Document fix** in commit message

---

## Performance Optimization

### Profiling
```bash
# Build with profiling
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build .

# Run with perf (Linux)
perf record ./bin/knn-classifier
perf report
```

### Benchmarking
```cpp
#include <benchmark.hpp>

// Before optimization
auto before = Benchmark::measure("Before", oldFunction, 100);

// After optimization
auto after = Benchmark::measure("After", newFunction, 100);

// Compare
double speedup = before.averageTimeMs / after.averageTimeMs;
std::cout << "Speedup: " << speedup << "x\n";
```

---

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [your-email@example.com]

## Recognition

Contributors will be listed in:
- README.md Contributors section
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
