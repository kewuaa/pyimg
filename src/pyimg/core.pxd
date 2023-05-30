cimport numpy as cnp
ctypedef unsigned char uint8
cdef fused ImageC1:
    double[:, :]
    uint8[:, :]
cdef fused ImageC3:
    double[:, :, :]
    uint8[:, :, :]
ctypedef double[:, :] Kernel
