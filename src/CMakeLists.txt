add_library(${PROJECT_NAME}_lib STATIC 
    knn.cpp
    dataset.cpp
    visualizer.cpp
)

target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)