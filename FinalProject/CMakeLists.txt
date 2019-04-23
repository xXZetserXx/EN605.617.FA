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
include_directories(SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/ext/includes)
# ==============================================

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRCS 
	${SRC_DIR}/main.cpp
	${SRC_DIR}/filterGen.cu)


include_directories(${CMAKE_SOURCE_DIR}/src)

cuda_add_executable(cu_CGH ${SRCS})
target_link_libraries(cu_CGH X11 png cufft)
