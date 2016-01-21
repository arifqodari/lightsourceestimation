cmake_minimum_required(VERSION 2.8.7)
project(lightsource)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_PREFIX_PATH "/usr/local")
set(CMAKE_INCLUDE_PATH "${CMAKE_PREFIX_PATH}/include")
set(CMAKE_LIBRARY_PATH "${CMAKE_PREFIX_PATH}/lib")

# C++11 supports
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# OpenCV
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect ml)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

# Boost
find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# add executable file
add_executable(main main.cpp tree.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES})
