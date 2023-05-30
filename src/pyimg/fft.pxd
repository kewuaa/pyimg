ctypedef double complex cdouble
cdef fused Array1d:
    const double[:]
    const cdouble[:]
cdef fused Array2d:
    const double[:, :]
    const cdouble[:, :]


cdef cdouble[:] _fft(Array1d arr, cdouble[:] out) noexcept nogil
cdef cdouble[:] _ifft(Array1d arr, cdouble[:] out) noexcept nogil
cdef void _fft2(Array2d arr, cdouble[:, :] out, cdouble[:, ::1] temp) noexcept nogil
cdef void _ifft2(Array2d arr, cdouble[:, :] out, cdouble[:, ::1] temp) noexcept nogil
cdef void _shift(cdouble[:, :] arr, cdouble[:, :] temp) noexcept nogil
