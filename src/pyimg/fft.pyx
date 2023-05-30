# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: extra_link_args=-fopenmp
# distutils: extra_compile_args=-fopenmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cpython cimport PyMem_Malloc, PyMem_Free
from libc.complex cimport cexp
from libc.math cimport log2f, pi
from cython.parallel cimport prange
cimport numpy as cnp


cdef void _dft(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维离散傅里叶变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    cdef cdouble jw = -2j * pi / N
    cdef cdouble W
    out[:] = 0.
    cdef size_t i, j
    for i in prange(N, nogil=True):
        W = jw * i
        for j in range(N):
            out[i] += cexp(W * j) * arr[j]


cdef void _idft(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维离散傅里叶逆变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    cdef cdouble jw = 2j * pi / N
    cdef cdouble W
    out[:] = 0.
    cdef size_t i, j
    for i in prange(N, nogil=True):
        W = jw * i
        for j in range(N):
            out[i] += cexp(W * j) * arr[j]
        out[i] /= N


cdef cdouble[:] _fft(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维快速傅里叶变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    cdef size_t bit = <size_t>log2f(<float>N) - 1
    if bit % 1 != 0.:
        return _fft_by_recusive[Array1d](arr, out)
    cdef size_t new_pos, origin_pos, b
    # 位逆序置换，例如 001 -> 100; 0101 -> 1010
    for origin_pos in range(N):
        new_pos = 0
        for b in range(bit + 1):
            new_pos |= (origin_pos >> b & 1) << (bit - b)
        out[new_pos] = arr[origin_pos]
    cdef cdouble jw = -1j * pi
    cdef cdouble e, o
    cdef size_t step = 2
    cdef size_t half_step = 1
    cdef size_t i, j
    for _ in range(bit + 1):
        i = 0
        while i < N:
            for j in range(half_step):
                e = out[i + j]
                o = out[i + j + half_step] * cexp(jw * j)
                out[i + j] = e + o
                out[i + j + half_step] = e - o
            i += step
        step *= 2
        half_step *= 2
        jw /= 2
    return out


cdef cdouble[:] _fft_by_recusive(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维快速傅里叶变换（递归版）。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    if N % 2 != 0:
        if N < 2:
            out[0] = <cdouble>arr[0]
            return out
        else:
            _dft[Array1d](arr, out)
            return out
    cdef size_t half = N / 2
    cdef cdouble jw = -2j * pi / N
    cdef cdouble[:] arr_e = _fft_by_recusive(arr[::2], out[:half])
    cdef cdouble[:] arr_o = _fft_by_recusive(arr[1::2], out[half:])
    cdef size_t i
    cdef cdouble e, o
    for i in range(half):
        e = arr_e[i]
        o = arr_o[i] * cexp(jw * i)
        out[i] = e + o
        out[i + half] = e - o
    return out


cdef void _fft2(Array2d arr, cdouble[:, :] out, cdouble[:, ::1] temp) noexcept nogil:
    """2 维快速傅里叶变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    :param temp: 临时内存视图，用于保存中间结果
    """

    cdef size_t rows = arr.shape[0]
    cdef size_t cols = arr.shape[1]
    cdef size_t i
    if Array2d is cdouble[:, :]:
        for i in prange(rows, nogil=True):
            _fft[cdouble[:]](arr[i, ...], temp[i, ...])
    else:
        for i in prange(rows, nogil=True):
            _fft[double[:]](arr[i, ...], temp[i, ...])
    for i in prange(cols, nogil=True):
        _fft[cdouble[:]](temp[..., i], out[..., i])


cdef cdouble[:] _ifft(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维快速傅里叶逆变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    cdef size_t bit = <size_t>log2f(<float>N) - 1
    if bit % 1 != 0.:
        return _fft_by_recusive[Array1d](arr, out)
    cdef size_t new_pos, origin_pos, b
    # 位逆序置换，例如 001 -> 100; 0101 -> 1010
    for origin_pos in range(N):
        new_pos = 0
        for b in range(bit + 1):
            new_pos |= (origin_pos >> b & 1) << (bit - b)
        out[new_pos] = arr[origin_pos]
    cdef cdouble jw = 1j * pi
    cdef cdouble e, o
    cdef size_t step = 2
    cdef size_t half_step = 1
    cdef size_t i, j
    for _ in range(bit + 1):
        i = 0
        while i < N:
            for j in range(half_step):
                e = out[i + j] / 2
                o = out[i + j + half_step] * cexp(jw * j) / 2
                out[i + j] = e + o
                out[i + j + half_step] = e - o
            i += step
        step *= 2
        half_step *= 2
        jw /= 2
    return out


cdef cdouble[:] _ifft_by_recusive(Array1d arr, cdouble[:] out) noexcept nogil:
    """1 维快速傅里叶逆变换（递归版）。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    """

    cdef size_t N = arr.shape[0]
    if N % 2 != 0:
        if N < 2:
            out[0] = <cdouble>arr[0]
            return out
        else:
            _idft[Array1d](arr, out)
            return out
    cdef size_t half = N / 2
    cdef cdouble jw = 2j * pi / N
    cdef cdouble[:] arr_e = _fft_by_recusive(arr[::2], out[:half])
    cdef cdouble[:] arr_o = _fft_by_recusive(arr[1::2], out[half:])
    cdef size_t i
    cdef cdouble e, o
    for i in range(half):
        e = arr_e[i] / 2
        o = arr_o[i] * cexp(jw * i) / 2
        out[i] = e + o
        out[i + half] = e - o
    return out


cdef void _ifft2(Array2d arr, cdouble[:, :] out, cdouble[:, ::1] temp) noexcept nogil:
    """2 维快速傅里叶逆变换。

    :param arr: 待变换序列
    :param out: 保存结果的内存视图
    :param temp: 临时内存视图，用于保存中间结果
    """

    cdef size_t rows = arr.shape[0]
    cdef size_t cols = arr.shape[1]
    cdef size_t i
    if Array2d is cdouble[:, :]:
        for i in prange(rows, nogil=True):
            _ifft[cdouble[:]](arr[i, ...], temp[i, ...])
    else:
        for i in prange(rows, nogil=True):
            _ifft[double[:]](arr[i, ...], temp[i, ...])
    for i in prange(cols, nogil=True):
        _ifft[cdouble[:]](temp[..., i], out[..., i])


cdef void _shift(cdouble[:, :] arr, cdouble[:, :] temp) noexcept nogil:
    """二维频谱高低频转移。

    :param arr: 输入二维频谱
    """

    cdef size_t half_rows = arr.shape[0] / 2
    cdef size_t half_cols = arr.shape[1] / 2
    temp[:half_rows, :][:] = arr[-half_rows:, :]
    temp[half_rows:, :][:] = arr[:-half_rows, :]
    arr[:, :half_cols][:] = temp[:, -half_cols:]
    arr[:, half_cols:][:] = temp[:, :-half_cols]


# def dft(Array1d arr) -> cnp.ndarray:
#     """用于外部调用的 1 维离散傅里叶变换。

#     :param arr: 待变换序列
#     :return: 以 numpy 数组形式返回变换结果
#     """

#     cdef size_t N = arr.shape[0]
#     cdef cdouble* data = <cdouble*>PyMem_Malloc(sizeof(cdouble) * N)
#     cdef cdouble[::1] view = <cdouble[:N]>data
#     with nogil:
#         _dft[Array1d](arr, view)
#     cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
#         1, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
#     )
#     cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
#     return out


# def dft2(Array2d arr) -> cnp.ndarray:
#     """用于外部调用的 2 维离散傅里叶变换。

#     :param arr: 待变换序列
#     :return: 以 numpy 数组形式返回变换结果
#     """

#     cdef size_t rows = arr.shape[0]
#     cdef size_t cols = arr.shape[1]
#     cdef size_t size = sizeof(cdouble) * rows * cols
#     cdef cdouble* data = <cdouble*>PyMem_Malloc(size)
#     cdef cdouble* temp = <cdouble*>PyMem_Malloc(size)
#     cdef cdouble[:, ::1] view = <cdouble[:rows, :cols]>data
#     cdef cdouble[:, ::1] temp_view = <cdouble[:rows, :cols]>temp
#     cdef size_t i
#     with nogil:
#         if Array2d is cdouble[:, :]:
#             for i in prange(rows):
#                 _dft[cdouble[:]](arr[i, ...], temp_view[i, ...])
#         else:
#             for i in prange(rows):
#                 _dft[double[:]](arr[i, ...], temp_view[i, ...])
#         for i in prange(cols):
#             _dft[cdouble[:]](temp_view[..., i], view[..., i])
#     cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
#         2, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
#     )
#     cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
#     PyMem_Free(temp)
#     return out


# def idft(Array1d arr) -> cnp.ndarray:
#     """用于外部调用的 1 维离散傅里叶逆变换。

#     :param arr: 待变换序列
#     :return: 以 numpy 数组形式返回变换结果
#     """

#     cdef size_t N = arr.shape[0]
#     cdef cdouble* data = <cdouble*>PyMem_Malloc(sizeof(cdouble) * N)
#     cdef cdouble[::1] view = <cdouble[:N]>data
#     with nogil:
#         _idft[Array1d](arr, view)
#     cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
#         1, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
#     )
#     cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
#     return out


# def idft2(Array2d arr) -> cnp.ndarray:
#     """用于外部调用的 2 维离散傅里叶逆变换。

#     :param arr: 待变换序列
#     :return: 以 numpy 数组形式返回变换结果
#     """

#     cdef size_t rows = arr.shape[0]
#     cdef size_t cols = arr.shape[1]
#     cdef size_t size = sizeof(cdouble) * rows * cols
#     cdef cdouble* data = <cdouble*>PyMem_Malloc(size)
#     cdef cdouble* temp = <cdouble*>PyMem_Malloc(size)
#     cdef cdouble[:, ::1] view = <cdouble[:rows, :cols]>data
#     cdef cdouble[:, ::1] temp_view = <cdouble[:rows, :cols]>temp
#     cdef unsigned i
#     with nogil:
#         if Array2d is cdouble[:, :]:
#             for i in prange(rows):
#                 _idft[cdouble[:]](arr[i, ...], temp_view[i, ...])
#         else:
#             for i in prange(rows):
#                 _idft[double[:]](arr[i, ...], temp_view[i, ...])
#         for i in prange(cols):
#             _idft[cdouble[:]](temp_view[..., i], view[..., i])
#     cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
#         2, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
#     )
#     cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
#     PyMem_Free(temp)
#     return out


def fft(Array1d arr) -> cnp.ndarray:
    """用于外部调用的 1 维快速傅里叶变换。

    :param arr: 待变换序列
    :return: 以 numpy 数组形式返回变换结果
    """

    cdef size_t N = arr.shape[0]
    cdef cdouble* data = <cdouble*>PyMem_Malloc(sizeof(cdouble) * N)
    cdef cdouble[::1] view = <cdouble[:N]>data
    with nogil:
        _fft[Array1d](arr, view)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        1, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def fft2(Array2d arr) -> cnp.ndarray:
    """用于外部调用的 2 维快速傅里叶变换。

    :param arr: 待变换序列
    :return: 以 numpy 数组形式返回变换结果
    """
    cdef size_t rows = arr.shape[0]
    cdef size_t cols = arr.shape[1]
    cdef size_t size = sizeof(cdouble) * rows * cols
    cdef cdouble* data = <cdouble*>PyMem_Malloc(size)
    cdef cdouble* temp = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] view = <cdouble[:rows, :cols]>data
    cdef cdouble[:, ::1] temp_view = <cdouble[:rows, :cols]>temp
    with nogil:
        _fft2[Array2d](arr, view, temp_view)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    PyMem_Free(temp)
    return out


def ifft(Array1d arr) -> cnp.ndarray:
    """用于外部调用的 1 维快速傅里叶逆变换。

    :param arr: 待变换序列
    :return: 以 numpy 数组形式返回变换结果
    """

    cdef size_t N = arr.shape[0]
    cdef cdouble* data = <cdouble*>PyMem_Malloc(sizeof(cdouble) * N)
    cdef cdouble[::1] view = <cdouble[:N]>data
    with nogil:
        _ifft[Array1d](arr, view)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        1, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def ifft2(Array2d arr) -> cnp.ndarray:
    """用于外部调用的 2 维快速傅里叶逆变换。

    :param arr: 待变换序列
    :return: 以 numpy 数组形式返回变换结果
    """

    cdef size_t rows = arr.shape[0]
    cdef size_t cols = arr.shape[1]
    cdef size_t size = sizeof(cdouble) * rows * cols
    cdef cdouble* data = <cdouble*>PyMem_Malloc(size)
    cdef cdouble* temp = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] view = <cdouble[:rows, :cols]>data
    cdef cdouble[:, ::1] temp_view = <cdouble[:rows, :cols]>temp
    with nogil:
        _ifft2[Array2d](arr, view, temp_view)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &arr.shape[0], cnp.NPY_CDOUBLE, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    PyMem_Free(temp)
    return out


def shift(cdouble[:, :] arr) -> None:
    cdef size_t rows = arr.shape[0]
    cdef size_t cols = arr.shape[1]
    cdef cdouble* temp_data = <cdouble*>PyMem_Malloc(sizeof(cdouble) * rows * cols)
    cdef cdouble[:, ::1] temp = <cdouble[:rows, :cols]>temp_data
    _shift(arr, temp)
