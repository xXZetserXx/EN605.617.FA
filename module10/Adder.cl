// OpenCL kernel for element-by-element addition
__kernel void elemAdd(__global float *a, __global float *b, __global float *out, __global int *numElements)
{
	int gid = get_global_id(0);
	
	if(gid >= numElements) {
		return;
	}
	
	out[gid] = a[gid] + b[gid];
}
