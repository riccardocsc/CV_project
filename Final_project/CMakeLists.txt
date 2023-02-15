cmake_minimum_required(VERSION 2.8)

project(final_project)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp src/object_recognition.h src/object_recognition.cpp src/cascade.h src/cascade.cpp src/search.h src/search.cpp src/hog.h src/hog.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})