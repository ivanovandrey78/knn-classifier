# Contributing

Thanks for considering contributing! This project follows common open-source
practices used in major C++ repositories like
[fmtlib/fmt](https://github.com/fmtlib/fmt),
[pybind/pybind11](https://github.com/pybind/pybind11), and
[nlohmann/json](https://github.com/nlohmann/json).

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/knn-classifier.git
cd knn-classifier
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest
```

## Branch Naming

| Prefix       | Description                        |
|-------------|------------------------------------|
| `feature/`  | New feature                        |
| `fix/`      | Bug fix                            |
| `refactor/` | Code restructuring                 |
| `docs/`     | Documentation only                 |
| `test/`     | Test improvements                  |
| `perf/`     | Performance improvements           |
| `codestyle/`| Formatting, naming, style fixes    |

Examples:
```
feature/kd-tree-weighted-voting
fix/visualizer-color-overflow
codestyle/snake-case-variables
```

## Cherry-Picking

We use cherry-pick for hotfixes and selective backports:

```bash
# Apply a specific commit to current branch
git cherry-pick <commit-hash>

# Cherry-pick with original author preserved
git cherry-pick -x <commit-hash>

# Abort on conflict
git cherry-pick --abort
```

Rules:
- Only cherry-pick **compilable, tested commits** (no partial features)
- Always use `-x` to record the source commit hash
- Prefer merge over cherry-pick for feature branches
- For critical fixes, cherry-pick to release branches after testing on `main`

## Coding Standards

We follow a **Google C++ Style Guide** with minor adjustments.

### Naming
```cpp
class KnnOptimized { };          // PascalCase for types
void calculateDistance();        // lowerCamelCase for functions
int num_neighbors;               // snake_case for variables
constexpr int kMaxIter = 100;    // kPrefixed for constexpr
namespace cross_validation { }   // lowercase for namespaces
```

### Modern C++ Features — Use
- `auto` for complex types
- Range-based for loops
- Smart pointers (`unique_ptr`, `shared_ptr`)
- Structured bindings
- `constexpr` / `if constexpr`
- `= default` / `= delete`
- **`<=>` (spaceship operator)** for multi-way comparisons
- `std::optional`, `std::variant`

### Modern C++ — Avoid
- Raw `new` / `delete`
- C-style casts (`(int)x`)
- `NULL` (use `nullptr`)
- Manual memory management
- Global variables

### Formatting
- 4 spaces (no tabs)
- K&R braces
- Left-aligned pointers: `int* p`

## Testing

Run tests:
```bash
cd build && ctest --output-on-failure
```

All new features must include tests. We aim for:
- **Statements**: >80%
- **Functions**: >90%

## Pull Requests

Before submitting:
- [ ] Compiles without warnings
- [ ] All tests pass
- [ ] Formatted with `clang-format`
- [ ] Tests added for new code
- [ ] Docs updated if needed

## License

By contributing you agree to the MIT License.
