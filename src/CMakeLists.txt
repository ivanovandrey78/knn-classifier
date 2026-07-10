add_library(${PROJECT_NAME}_lib STATIC 
    knn.cpp
    dataset.cpp
    visualizer.cpp
    kdtree.cpp
    knn_optimized.cpp
    benchmark.cpp
)

target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link pthread for threading support
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME}_lib
    PUBLIC
        Threads::Threads
)