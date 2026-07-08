include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0  
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tests 
    point_ut.cpp
)

target_link_libraries(tests
    PRIVATE
        gtest_main  
        ${PROJECT_NAME}_lib
)

include(GoogleTest)
gtest_discover_tests(tests)
