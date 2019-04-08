/*
 * nvgraphAssignment.cpp
 *
 *  Created on: Apr 8, 2019
 *  Author: Mikhail Wolff
 *  Description: This will be a Single-Source shortest path examkple of using nvgraph.
 *  It will calculate the shortest path from one vertex in the graph to all other vertices.
 *  See ExampleGraph.png for a visual of the graph this is being performed on.
 *
 *  We have...
 *  Vertices: 5
 *  Edges: 9
 *
 *  I don't understand what the weight refers to...
 *  Edges		W
 *  A->B		0.50
 *  A->E		0.50
 *  B->E		0.50
 *  B->C		0.50
 *  C->D		1.0
 *  D->C		1.0
 *  E->B		0.33
 *  E->C		0.33
 *  E->D		0.33
 *
 * Source oriented representation (CSC):
 * destination_offsets{0, 1, 3, 4, 6, 8}
 * source_indices {2, 2, 1, 1, 3}
 * W0 = {0.50, 0.50, 0.50, 0.50, 1.0, 1.0, 0.33, 0.33, 0.33}
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nvgraph.h"

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

int main(int argc, char **argv) {

	const size_t  n = 6, nnz = 10, vertex_numsets = 2, edge_numsets = 1;
	int i, *destination_offsets_h, *source_indices_h;
	float *weights_h, *sssp_1_h, *sssp_2_h;
	void** vertex_dim;

	// nvgraph variables
	nvgraphStatus_t status;
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t CSC_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;
}
