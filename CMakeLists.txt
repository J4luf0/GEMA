cmake_minimum_required(VERSION 3.31) #default 3.10

project(GEMA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    #URL https://github.com/google/googletest/archive/0953a17a4281fc26831da647ad3fcd5e21e6473b.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
    FetchContent_MakeAvailable(googletest)
endif()

#file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*")
#list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/CMakeFiles/.*")

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/test/*.cpp")

enable_testing()

add_executable(GEMA ${SOURCES})

target_link_libraries(GEMA gtest_main) #target_link_libraries(GEMA GTest::gtest_main)
include(googletest)

gtest_discover_tests(GEMA EXTRA_ARGS --gtest_color=yes)

target_include_directories(GEMA PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(GEMA PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_include_directories(GEMA PRIVATE ${GTEST_DIR}/include/gtest)


