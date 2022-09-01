cmake_minimum_required(VERSION 3.11)

project(StsDigiSort VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

#add_subdirectory(tests)
add_subdirectory(lib/xpu)

add_library(BlockSort SHARED sorting/BlockSort.cpp)
xpu_attach(BlockSort sorting/BlockSort.cpp)

add_library(JanSergeySort SHARED sorting/JanSergeySort.cpp)
xpu_attach(JanSergeySort sorting/JanSergeySort.cpp)

add_library(JanSergeySortSingleBlock SHARED sorting/JanSergeySortSingleBlock.cpp)
xpu_attach(JanSergeySortSingleBlock sorting/JanSergeySortSingleBlock.cpp)

add_library(JanSergeySortParInsert SHARED sorting/JanSergeySortParInsert.cpp)
xpu_attach(JanSergeySortParInsert sorting/JanSergeySortParInsert.cpp)

# add the executable
add_executable(stsdigisort main.cpp)
target_link_libraries(stsdigisort xpu BlockSort JanSergeySort JanSergeySortSingleBlock JanSergeySortParInsert)
