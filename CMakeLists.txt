cmake_minimum_required(VERSION 3.11)

project(StsDigiSort VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

#add_subdirectory(tests)
add_subdirectory(lib/xpu)

add_library(SortKernel SHARED sorting/SortKernel.cpp)
xpu_attach(SortKernel sorting/SortKernel.cpp)

add_library(JanSergeySort SHARED sorting/JanSergeySort.cpp)
xpu_attach(JanSergeySort sorting/JanSergeySort.cpp)

add_library(JanSergeySortSided SHARED sorting/JanSergeySortSided.cpp)
xpu_attach(JanSergeySortSided sorting/JanSergeySortSided.cpp)

# add the executable
add_executable(stsdigisort main.cpp)
target_link_libraries(stsdigisort xpu SortKernel JanSergeySort JanSergeySortSided)
