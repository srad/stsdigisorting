git clone https://github.com/fnc12/sqlite_orm.git sqlite_orm
export Catch2_DIR=./sqlite_orm/Catch2/build
cd sqlite_orm
cmake -B build
cmake --build build #--target install