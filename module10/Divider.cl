__kernel void elemDiv(__global float *a, __global *b, __global float *out, __global int *numElements)
{
	int gid = get_global_id(0);
	
	if(gid >= numElements) {
		return;
	}
	
	out[gid] = a[gid] / b[gid];
}
