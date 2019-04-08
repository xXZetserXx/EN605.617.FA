CUDA_PATH?=/usr/local/cuda

# Common includes and libraries for CUDA
INCLUDES  := -I../common/inc
LIBRARIES := -L/usr/local/cuda/lib -lcudart


NPPLIBS := -lnppc_static -lnppial_static -lnppicc_static -lnppicom_static -lnppidei_static -lnppif_static -lnppig_static -lnppim_static -lnppist_static -lnppisu_static -lnppitc_static -lnpps_static
INCLUDES += -I./
LIBRARIES += -lculibos ${NPPLIBS} -lX11

target1: nppAssignment.cpp
	nvcc nppAssignment.cpp -o nppExec ${INCLUDES} ${LIBRARIES} -lpng -std=c++11

target2: thrustAssignment.cu
	nvcc thrustAssignment.cu -o thrustExec ${INCLUDES} ${LIBRARIES} -std=c++11
	
all: target1 target2