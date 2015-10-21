import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cimport openmp

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef AVX.float8 magnitude_squared_float8(AVX.float8 z_real_float8, AVX.float8 z_imag_float8) nogil:
    cdef AVX.float8 real_mag = AVX.mul(z_real_float8, z_real_float8)
    cdef AVX.float8 imag_mag = AVX.mul(z_imag_float8, z_imag_float8)

    return AVX.add(real_mag, imag_mag)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j

       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    cdef int num_cols_to_iterate = in_coords.shape[1]/8
    cdef int j_start, j_end

    cdef float[:, :] real_in_coords = np.real(in_coords)
    cdef float[:, :] imag_in_coords = np.imag(in_coords)

    cdef float *real_c
    cdef float *imag_c

    cdef AVX.float8 real_c_float8, imag_c_float8

    cdef AVX.float8 real_z_float8, imag_z_float8
    cdef AVX.float8 mag_squared
    cdef AVX.float8 go_mask

    cdef int go=1
    cdef AVX.float8 iter

    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            for j in range(num_cols_to_iterate): # Parallelize via AVX here...do 8 at a time
                j_start = 8*j
                j_end = 8*(j+1)
                if j_end >= in_coords.shape[1]:
                    j_end = in_coords.shape[1] - 1

                real_c = &real_in_coords[i][0] # get pointers to the arrays
                imag_c = &imag_in_coords[i][0]

                real_c_float8 = array_to_float8(real_c, j_start, j_end)
                imag_c_float8 = array_to_float8(imag_c, j_start, j_end)

                real_z_float8 = AVX.float_to_float8(0)
                imag_z_float8 = AVX.float_to_float8(0)

                # Need to iterate over 8 values at once...blahhhhh
                iter = AVX.float_to_float8(0)
                while go==1:
                    mag_squared = magnitude_squared_float8(real_z_float8, imag_z_float8)
                    go_mask = AVX.less_than(mag_squared, AVX.float_to_float8(4))
                    # If go, calculate z*z + c

                    # Now need to do procedural updates of the float8...bleh
                    #if magnitude_squared(z) > 4:
                    #    break
                    #z = z * z + c
                #out_counts[i, j] = iter

cdef AVX.float8 array_to_float8(float *c, int j_start, int j_end) nogil:
    cdef float *filled_array = <float *> malloc(8*sizeof(c))

    cdef int count = 0
    cdef int j
    for j in range(j_start, j_end):
        filled_array[count] = c[j]
        count += 1
    # Fill in the rest of the zeros
    while count < 8:
        filled_array[count] = 0
        count += 1

    cdef AVX.float8 f8 = AVX.make_float8(filled_array[7],
                           filled_array[6],
                           filled_array[5],
                           filled_array[4],
                           filled_array[3],
                           filled_array[2],
                           filled_array[1],
                           filled_array[0])
    free(filled_array)
    return f8

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        float out_vals[8]
        float [:] out_view = out_vals

    assert values.shape[0] == 8

    # Note that the order of the arguments here is opposite the direction when
    # we retrieve them into memory.
    avxval = AVX.make_float8(values[7],
                             values[6],
                             values[5],
                             values[4],
                             values[3],
                             values[2],
                             values[1],
                             values[0])

    avxval = AVX.sqrt(avxval)

    # mask will be true where 2.0 < avxval
    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
