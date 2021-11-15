cmake_minimum_required(VERSION 3.4)
project(generic_pse)

set(CMAKE_CXX_STANDARD 17)
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

SET (CMAKE_CXX_COMPILER "/usr/bin/g++")

# Find yaml-cpp
find_package (PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)
# Find yaml-cpp Done

# Find Armadillo
find_package(Armadillo REQUIRED)
# Find Armadillo Done

# Find Lapack
find_package(LAPACK REQUIRED)
# Find Lapack Done

# Find HDF5
find_package(HDF5  REQUIRED)
# Find HDF5 Done

# Find pybind11
find_package(pybind11  REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS} ${pybind11_LIBRARIES}")
# Find pybind11 Done

INCLUDE_DIRECTORIES(
        "${CMAKE_SOURCE_DIR}/include"
        ${ARMADILLO_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        ${YAMLCPP_INCLUDE_DIRS}
)
LINK_DIRECTORIES(
        ${YAMLCPP_LIBRARY_DIRS}
)

add_executable(generic_pse main.cpp src/utils.cpp include/utils.h src/spectrum_estimator.cpp include/spectrum_estimator.h)

target_link_libraries(${PROJECT_NAME} ${YAMLCPP_LIBRARIES}
        ${LAPACK_LIBRARIES} ${ARMADILLO_LIBRARIES}
        ${HDF5_LIBRARIES} python3.6m pybind11::module)
