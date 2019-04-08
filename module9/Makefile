CUDA_PATH?=/usr/local/cuda

# Common includes and libraries for CUDA
INCLUDES  := -I../common/inc
LIBRARIES := -L/usr/local/cuda/lib -lcudart


NPPLIBS := -lnppc_static -lnppial_static -lnppicc_static -lnppicom_static -lnppidei_static -lnppif_static -lnppig_static -lnppim_static -lnppist_static -lnppisu_static -lnppitc_static -lnpps_static
INCLUDES += -I./
LIBRARIES += -lculibos ${NPPLIBS} -lX11

all: nppAssignment.cpp
	nvcc nppAssignment.cpp -o nppExec ${INCLUDES} ${LIBRARIES} -std=c++11
