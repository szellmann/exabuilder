cmake_minimum_required(VERSION 3.28)

project(exabuilder LANGUAGES C CXX CUDA)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE main.cu)

set_source_files_properties(main.cu PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr")
