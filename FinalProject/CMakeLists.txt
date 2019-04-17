cmake_minimum_required(VERSION 2.8)
project(cu_CGH LANGUAGES CUDA CXX)

find_package(CUDA QUIET REQUIRED)	

if(NOT DEFINED CMAKE_CUDA_COMPILER)
	set(CMAKE_CUDA_COMPILER nvcc)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
	set(CMAKE_CXX_COMPILER g++)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
# ==============================================
# External Project setup
include_directories(${CMAKE_SOURCE_DIR}/ext/includes)
# ==============================================

set(SRCS ${CMAKE_SOURCE_DIR}/src/main.cpp)


cuda_add_executable(helloworld ${SRCS})
target_link_libraries(helloworld X11 png)
