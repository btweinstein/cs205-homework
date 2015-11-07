__kernel void sum_coalesced(__global float* x,
                            __global float* partial,
                            __local  float* fast,
                            long N)
{
    size_t local_id = get_local_id(0);
    int k = get_global_size(0);
    int i = get_global_id(0);

    // thread i (i.e., with i = get_global_id()) should add x[i],
    // x[i + get_global_size()], ... up to N-1, and store in sum.
    int zmax = (N-i)/k;
    float sum = 0;
    for (int z=0; z < zmax; z++) { // YOUR CODE HERE
        sum += x[i + z*k];
    }

    fast[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // binary reduction
    //
    // thread i should sum fast[i] and fast[i + offset] and store back
    // in fast[i], for offset = (local_size >> j) for j from 1 to
    // log_2(local_size) WTF lol
    //
    // You can assume get_local_size(0) is a power of 2.
    //
    // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
    uint local_size = get_local_size(0);
    uint max_j = (uint) log2((double) local_size);
    for (uint j=1;j <= max_j; j++) { // YOUR CODE HERE
        fast[i] += fast[i + (local_size >> j)];
    }

    if (local_id == 0) partial[get_group_id(0)] = fast[0];
}


__kernel void sum_blocked(__global float* x,
                          __global float* partial,
			              __local  float* fast,
			              long N)
{
        float sum = 0;
        size_t local_id = get_local_id(0);
        int k = ceil((float)N / get_global_size(0));

        // thread with global_id 0 should add 0..k-1
        // thread with global_id 1 should add k..2k-1
        // thread with global_id 2 should add 2k..3k-1
        // ...
        //     with k = ceil(N / get_global_size()).
        // 
        // Be careful that each thread stays in bounds, both relative to
        // size of x (i.e., N), and the range it's assigned to sum.
        for (;;) { // YOUR CODE HERE
            ; // YOUR CODE HERE
        }

        fast[local_id] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // binary reduction
        //
        // thread i should sum fast[i] and fast[i + offset] and store back
        // in fast[i], for offset = (local_size >> j) for j from 1 to
        // log_2(local_size)
        //
        // You can assume get_local_size(0) is a power of 2.
        //
        // See http://www.nehalemlabs.net/prototype/blog/2014/06/16/parallel-programming-with-opencl-and-python-parallel-reduce/
        for (;;) { // YOUR CODE HERE
            ; // YOUR CODE HERE
        }

        if (local_id == 0) partial[get_group_id(0)] = fast[0];
    }
