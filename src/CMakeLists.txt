cmake_minimum_required(VERSION 3.10)
project(JPEGCompression)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to OpenCV
set(OpenCV_DIR "C:/opencv/build") # Adjust if needed

find_package(OpenCV REQUIRED)

add_executable(JPEGCompression main.cpp)
target_link_libraries(JPEGCompression ${OpenCV_LIBS})