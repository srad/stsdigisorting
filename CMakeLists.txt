cmake_minimum_required(VERSION 3.11)

project(StsDigiSort VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/lib/xpu/src)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(xpu SHARED
    lib/xpu/src/xpu/detail/dl_utils.cpp
    lib/xpu/src/xpu/detail/log.cpp
    lib/xpu/src/xpu/detail/runtime.cpp
    lib/xpu/src/xpu/driver/cpu/cpu_driver.cpp
    lib/xpu/src/xpu/driver/cpu/this_thread.cpp
)

add_library(SortKernel SHARED SortKernel.cpp)

# add the executable
add_executable(stsdigisort main.cpp)
target_link_libraries(stsdigisort xpu SortKernel)
