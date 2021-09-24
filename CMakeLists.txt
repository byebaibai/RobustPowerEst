cmake_minimum_required(VERSION 3.0)
project(generic_pse)

set(CMAKE_CXX_STANDARD 17)
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")

# Find Eigen3
SET( EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/eigen3/include" )
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )
# Find Eigen3 Done

# Find GTest
SET(GTEST_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/gtest")
find_package(GTest REQUIRED)
# Find GTest Done

# Find yaml-cpp
SET(YAML_CPP_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/yaml-cpp")
SET(YAML_CPP_INCLUDE_DIRS "${YAML_CPP_ROOT}/include")
FIND_LIBRARY(YAML_CPP_LIBRARIES libyaml-cpp.a
        "${YAML_CPP_ROOT}/lib"
        )
# Find yaml-cpp Done

# Find fmt
SET(FMT_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/fmt")
SET(FMT_INCLUDE_DIRS "${FMT_ROOT}/include")
FIND_LIBRARY(FMT_LIBRARIES libfmt.a
        "${FMT_ROOT}/lib"
        )
# Find fmt Done

# Find kfr
SET(KFR_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/kfr")
SET(KFR_INCLUDE_DIRS "${KFR_ROOT}/include")
add_subdirectory(3rdparty/kfr kfr-bin)
# Find kfr Done

# Find Lapack
find_package(LAPACK REQUIRED)
# Find Lapack Done

INCLUDE_DIRECTORIES(
        "${CMAKE_SOURCE_DIR}/include"
        ${GTEST_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${FMT_INCLUDE_DIRS}
        ${KFR_INCLUDE_DIRS}
)
add_executable(generic_pse main.cpp src/dpss.cpp include/dpss.h src/utils.cpp include/utils.h)

target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES} ${FMT_LIBRARIES} kfr kfr_io kfr_dft ${LAPACK_LIBRARIES})
