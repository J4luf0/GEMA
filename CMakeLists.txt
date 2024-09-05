cmake_minimum_required(VERSION 3.10)

project(GEMA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/0953a17a4281fc26831da647ad3fcd5e21e6473b.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*")
list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/CMakeFiles/.*")

enable_testing()

add_executable(GEMA ${SOURCES})

target_link_libraries(GEMA GTest::gtest_main)
include(googletest)
gtest_discover_tests(GEMA EXTRA_ARGS --gtest_color=yes)

target_include_directories(GEMA PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(GEMA PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_include_directories(GEMA PRIVATE ${PROJECT_SOURCE_DIR}/lib/googletest/include)


