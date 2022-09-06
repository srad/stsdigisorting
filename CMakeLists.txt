cmake_minimum_required(VERSION 3.11)

project(StsDigiSort VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

add_subdirectory(lib/sqlite_orm)

#add_subdirectory(tests)
add_subdirectory(lib/xpu)

add_library(BlockSort SHARED src/sorting/BlockSort.cpp)
xpu_attach(BlockSort src/sorting/BlockSort.cpp)

add_library(JanSergeySort SHARED src/sorting/JanSergeySort.cpp)
xpu_attach(JanSergeySort src/sorting/JanSergeySort.cpp)

add_library(JanSergeySortSingleBlock SHARED src/sorting/JanSergeySortSingleBlock.cpp)
xpu_attach(JanSergeySortSingleBlock src/sorting/JanSergeySortSingleBlock.cpp)

add_library(JanSergeySortSimple SHARED src/sorting/JanSergeySortSimple.cpp)
xpu_attach(JanSergeySortSimple src/sorting/JanSergeySortSimple.cpp)


add_library(JanSergeySortParInsert SHARED src/sorting/JanSergeySortParInsert.cpp)
xpu_attach(JanSergeySortParInsert src/sorting/JanSergeySortParInsert.cpp)

# add the executable
add_executable(stsdigisort src/main.cpp)
target_link_libraries(stsdigisort
    Threads::Threads
    xpu
    BlockSort
    JanSergeySort
    JanSergeySortSimple
    JanSergeySortSingleBlock
    JanSergeySortParInsert
    sqlite_orm::sqlite_orm
    )
