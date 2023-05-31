# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cpython cimport PyMem_Malloc, PyMem_Free
from libc.math cimport (
    log2f, pow as cpow,
    exp as cexp, fmaxf, sin as csin, cos as ccos,
    fminf, lround, fdim, hypot, pi
)
from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport printf
from libc.string cimport memcpy, memset
from cython.parallel cimport prange
from numpy.math cimport NPY_1_PI
cimport numpy as cnp

from .fft cimport cdouble, _fft2, _ifft2, _shift
cpdef enum FilterType:
    FILTER_MEAN
    FILTER_MEDIAN
    FILTER_MIN
    FILTER_MAX
cpdef enum MorphologyType:
    MORPH_ERODE
    MORPH_DILATE
    MORPH_OPEN
    MORPH_CLOSE
    MORPH_TOPHAT
    MORPH_BLACKHAT
cpdef enum ThresholdType:
    THRESHOLD_GLOBAL
    THRESHOLD_OTSU


cdef void _rgb2gray(ImageC3 img, double[:, ::1] gray) noexcept nogil:
    """RGB 图像灰度转换。

    :param img: RGB 图像
    :return: 灰度图像
    """

    cdef float[3] RGB2GRAY_WEIGHT
    RGB2GRAY_WEIGHT[0] = 0.299
    RGB2GRAY_WEIGHT[1] = 0.587
    RGB2GRAY_WEIGHT[2] = 0.114
    gray[:] = 0.
    cdef size_t i, j, k
    for i in prange(img.shape[0], nogil=True):
        for j in range(img.shape[1]):
            for k in range(3):
                gray[i, j] += RGB2GRAY_WEIGHT[k] * img[i, j, k]


cdef void _special_filter(
    ImageC1 img,
    double[:, ::1] out,
    Kernel kernel,
    double* range_,
) noexcept:
    """空域滤波器。

    :param img: 待滤波图像
    :param out: 保存结果的内存视图
    :param kernel: 滤波核
    :param range_: 归一化范围
    """

    cdef size_t rows = img.shape[0]
    cdef size_t cols = img.shape[1]
    cdef float log_value
    cdef size_t shape[2]
    log_value = log2f(<float>rows)
    shape[0] = rows if log_value % 1 == 0. else 1 << (<int>log_value + 1)
    log_value = log2f(<float>cols)
    shape[1] = cols if log_value % 1 == 0. else 1 << (<int>log_value + 1)
    cdef size_t size = sizeof(double) * shape[0] * shape[1]
    cdef double* img_data = <double*>PyMem_Malloc(size)
    cdef double[:, ::1] pad_img = <double[:shape[0], :shape[1]]>img_data
    cdef double* kernel_data = <double*>PyMem_Malloc(size)
    cdef double[:, ::1] pad_kernel = <double[:shape[0], :shape[1]]>kernel_data
    size = sizeof(cdouble) * shape[0] * shape[1]
    cdef cdouble* fft_img_data = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] fft_img = <cdouble[:shape[0], :shape[1]]>fft_img_data
    cdef cdouble* fft_kernel_data = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] fft_kernel = <cdouble[:shape[0], :shape[1]]>fft_kernel_data
    cdef cdouble* temp = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] temp_view = <cdouble[:shape[0], :shape[1]]>temp
    cdef double max_, min_, ptp
    cdef size_t i, j
    with nogil:
        # pad img
        pad_img[:] = 0.
        for i in prange(rows):
            for j in range(cols):
                pad_img[i, j] = img[i, j]
        # pad kernel
        pad_kernel[:] = 0.
        pad_kernel[:kernel.shape[0], :kernel.shape[1]][:] = kernel[::-1, ::-1]
        # fft
        _fft2[double[:, :]](pad_img, fft_img, temp_view)
        _fft2[double[:, :]](pad_kernel, fft_kernel, temp_view)
        # mutiply
        for i in prange(shape[0]):
            for j in range(shape[1]):
                temp_view[i, j] = fft_img[i, j] * fft_kernel[i, j]
        # ifft
        _ifft2[cdouble[:, :]](temp_view, fft_img, fft_kernel)
        # slice useful part
        temp_view = fft_img[kernel.shape[0] - 1:rows, kernel.shape[1] - 1:cols]
        if range_ is not NULL:
            max_ = min_ = temp_view[0, 0].real
            for i in range(rows - kernel.shape[0] + 1):
                for j in range(cols - kernel.shape[1] + 1):
                    out[i, j] = temp_view[i, j].real
                    if out[i, j] > max_:
                        max_ = out[i, j]
                    if out[i, j] < min_:
                        min_ = out[i, j]
            ptp = (max_ - min_) / (range_[1] - range_[0])
            for i in prange(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j] = (out[i, j] - min_) / ptp + range_[0]
        else:
            for i in prange(rows - kernel.shape[0] + 1):
                for j in range(cols - kernel.shape[1] + 1):
                    out[i, j] = temp_view[i, j].real
    PyMem_Free(temp)
    PyMem_Free(fft_kernel_data)
    PyMem_Free(fft_img_data)
    PyMem_Free(kernel_data)
    PyMem_Free(img_data)


cdef void _init_guassion_kernel(Kernel kernel, float sigma, bint norm) noexcept nogil:
    """初始化一个高斯核。

    :param kernel: 需要初始化的高斯核
    :param sigma: 高斯系数
    :param norm: 是否归一化
    """

    cdef float center[2]
    center[0] = (kernel.shape[0] - 1) / 2.
    center[1] = (kernel.shape[1] - 1) / 2.
    cdef double pow_sigma = 2 * cpow(sigma, 2.)
    cdef double s = 0.
    cdef size_t i, j
    for i in prange(kernel.shape[0], nogil=True):
        for j in range(kernel.shape[1]):
            kernel[i, j] = cexp(
                -(cpow(i - center[0], 2.) + cpow(j - center[1], 2.))
                / pow_sigma
            ) * NPY_1_PI / pow_sigma
            s += kernel[i, j]
    if norm:
        for i in prange(kernel.shape[0], nogil=True):
            for j in range(kernel.shape[1]):
                kernel[i, j] /= s


cdef void _guassion_filter(
    ImageC1 img,
    double[:, ::1] out,
    size_t* kernel_shape,
    float sigma,
) noexcept:
    """高斯滤波器。

    :param img: 待滤波图像
    :param kernel_shape: 高斯核大小
    :param sigma: 高斯系数
    """

    cdef double* kernel_data = <double*>PyMem_Malloc(
        sizeof(double) * kernel_shape[0] * kernel_shape[1]
    )
    cdef double[:, ::1] kernel = <double[:kernel_shape[0], :kernel_shape[1]]>kernel_data
    _init_guassion_kernel(kernel, sigma, 1)
    _special_filter[ImageC1](img, out, kernel, NULL)
    PyMem_Free(kernel_data)


cdef void _guassion_low_pass(ImageC1 img, double[:, ::1] out, float sigma):
    """高斯低通滤波器。

    :param img: 待滤波图像
    :param out: 保存结果的内存视图
    :param sigma: 高斯系数
    """

    cdef size_t size = img.shape[0] * img.shape[1]
    cdef double[:, :] _img
    cdef double* kernel_data = <double*>PyMem_Malloc(sizeof(double) * size)
    cdef double[:, ::1] kernel = <double[:img.shape[0], :img.shape[1]]>kernel_data
    size *= sizeof(cdouble)
    cdef cdouble* fft_img_data = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] fft_img = <cdouble[:img.shape[0], :img.shape[1]]>fft_img_data
    cdef cdouble* temp_data1 = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] temp1 = <cdouble[:img.shape[0], :img.shape[1]]>temp_data1
    cdef cdouble* temp_data2 = <cdouble*>PyMem_Malloc(size)
    cdef cdouble[:, ::1] temp2 = <cdouble[:img.shape[0], :img.shape[1]]>temp_data2
    cdef size_t i, j
    cdef double min_, max_, ptp
    with nogil:
        if ImageC1 is double[:, :]:
            _img = img
        else:
            # convert type
            for i in prange(img.shape[0]):
                for j in range(img.shape[1]):
                    out[i, j] = img[i, j]
            _img = out
        _init_guassion_kernel(kernel, sigma, 0)
        _fft2[double[:, :]](_img, fft_img, temp1)
        _shift(fft_img, temp1)
        for i in prange(img.shape[0]):
            for j in range(img.shape[1]):
                temp1[i, j] = fft_img[i, j] * kernel[i, j]
        _shift(temp1, temp2)
        _ifft2[cdouble[:, :]](temp1, fft_img, temp2)
        max_ = min_ = fft_img[0, 0].real
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = fft_img[i, j].real
                if out[i, j] > max_:
                    max_ = out[i, j]
                if out[i, j] < min_:
                    min_ = out[i, j]
        ptp = max_ - min_
        for i in prange(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = (out[i, j] - min_) / ptp
    PyMem_Free(temp_data2)
    PyMem_Free(temp_data1)
    PyMem_Free(fft_img_data)
    PyMem_Free(kernel_data)


cdef void _mean_filter(
    ImageC1 img,
    uint8[:, ::1] out,
    size_t* kernel_shape
) noexcept nogil:
    """均值滤波器。

    :param img: 待滤波图像
    :param out: 保存结果的内存视图
    :param kernel_shape: 滤波核大小
    """

    cdef size_t size = kernel_shape[0] * kernel_shape[1]
    cdef double window_sum
    cdef size_t i, j, k, l
    for i in range(0, out.shape[0]):
        for j in range(0, out.shape[1]):
            if j == 0:
                # 计算第一个窗口中的像素之和
                window_sum = 0.
                for k in range(kernel_shape[0]):
                    for l in range(kernel_shape[1]):
                        window_sum += img[i + k, l]
            else:
                # 减去前一列，加上后一列
                for k in range(kernel_shape[0]):
                    window_sum -= img[i + k, j - 1] \
                        - img[i + k, j + kernel_shape[1] - 1]
            out[i, j] = lround(window_sum / size)


cdef void _median_filter(
    ImageC1 img,
    uint8[:, ::1] out,
    size_t* kernel_shape
) noexcept nogil:
    """中值滤波器。

    :param img: 待滤波图像
    :param out: 保存结果的内存视图
    :param kernel_shape: 滤波核大小
    """

    cdef size_t threshold = kernel_shape[0] * kernel_shape[1] / 2
    cdef size_t histongram[256]
    cdef size_t i, j, k, l, m
    cdef uint8 median, left, right
    cdef unsigned int cumulation
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if j == 0:
                cumulation = 0
                # 清空柱状图
                memset(histongram, 0, sizeof(size_t) * 256)
                # 统计每个像素值出现的次数
                for k in range(kernel_shape[0]):
                    for l in range(kernel_shape[1]):
                        histongram[<uint8>img[i + k, l]] += 1
                # 对像素值从小到大累加他们出现的次数，当超过阈值时，得到中值
                for m in range(256):
                    cumulation += histongram[m]
                    if cumulation >= threshold:
                        median = m
                        break
            else:
                for k in range(kernel_shape[0]):
                    left = <uint8>img[i + k, j - 1]
                    histongram[left] -= 1
                    # 前一列小于等于当前中值，则累加和减一
                    if left <= median:
                        cumulation -= 1
                    right = <uint8>img[i + k, j + kernel_shape[1] - 1]
                    histongram[right] += 1
                    # 后一列小于等于当前中值，则累加和加一
                    if right <= median:
                        cumulation += 1
                # 累加和小于阈值时，从当前中值往后逐个累加直到累加和大于阈值，得到新的中值
                if cumulation < threshold:
                    for k in range(median + 1, 256):
                        cumulation += histongram[k]
                        if cumulation >= threshold:
                            median = k
                            break
                # 累加和大于阈值时，从当前中值往前逐个累减直到累加和小于于阈值，得到新的中值
                elif cumulation > threshold:
                    for k in range(median, -1, -1):
                        if cumulation - histongram[k] < threshold:
                            median = k
                            break
                        cumulation -= histongram[k]
            out[i, j] = median


cdef void _best_value_filter(
    ImageC1 img,
    uint8[:, ::1] out,
    size_t* kernel_shape,
    bint flags
) noexcept nogil:
    """最值滤波器。

    :param img: 待滤波图像
    :param out: 保存结果的内存视图
    :param kernel_shape: 滤波核大小
    :param flags: 0 -> 最小; 1 -> 最大
    """

    cdef size_t i, j, k, l
    cdef size_t histongram[256]
    cdef int step
    cdef uint8 base, v, pixel, left, right
    cdef float (*compare)(float, float) nogil
    if flags:
        base = 0
        compare = fmaxf
        step = -1
    else:
        base = 255
        compare = fminf
        step = 1
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if j == 0:
                v = base
                memset(histongram, 0, sizeof(size_t) * 256)
                for k in range(kernel_shape[0]):
                    for l in range(kernel_shape[1]):
                        pixel = <uint8>img[i + k, l]
                        histongram[pixel] += 1
                        v = <uint8>compare(pixel, v)
            else:
                pixel = v
                for k in range(kernel_shape[0]):
                    left = <uint8>img[i + k, j - 1]
                    right = <uint8>img[i + k, j + kernel_shape[1] - 1]
                    histongram[left] -= 1
                    histongram[right] += 1
                    pixel = <uint8>compare(pixel, right)
                v = <uint8>compare(pixel, v)
                if v - pixel != 0.:
                    while 1:
                        if histongram[v] > 0:
                            break
                        v += step
            out[i, j] = v


cdef void _min_filter(
    ImageC1 img,
    uint8[:, ::1] out,
    size_t* kernel_shape
) noexcept nogil:
    _best_value_filter[ImageC1](img, out, kernel_shape, 0)
cdef void _max_filter(
    ImageC1 img,
    uint8[:, ::1] out,
    size_t* kernel_shape
) noexcept nogil:
    _best_value_filter[ImageC1](img, out, kernel_shape, 1)


cdef void _erode_dilate(
    uint8[:, :] img,
    uint8[:, ::1] out,
    size_t* kernel_shape,
    bint flags
) noexcept nogil:
    """形态学处理。

    :param img: 待处理图像
    :param out: 保存结果的内存视图
    :param kernel_shape: 核大小
    :param flags: 0 -> 腐蚀; 1 -> 膨胀
    """

    cdef size_t histongram[2]
    cdef size_t i, j, k, l
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if j == 0:
                histongram[0] = 0
                histongram[1] = 0
                for k in range(kernel_shape[0]):
                    for l in range(kernel_shape[1]):
                        histongram[<bint>img[i + k, l]] += 1
            else:
                for k in range(kernel_shape[0]):
                    histongram[<bint>img[i + k, j - 1]] -= 1
                    histongram[<bint>img[i + k, j + kernel_shape[1] - 1]] += 1
            if histongram[flags] > 0:
                out[i, j] = 255 * flags
            else:
                out[i, j] = 255 * (not flags)


cdef double _global_thresholding(ImageC1 img) noexcept nogil:
    """全局阈值化。

    :param img: 待处理图像
    :return: 阈值
    """

    cdef size_t size = img.shape[0] * img.shape[1]
    cdef size_t count = 0
    cdef double s0 = 0., s1 = 0., t0, t1
    cdef size_t i, j
    for i in prange(img.shape[0], nogil=True):
        for j in range(img.shape[1]):
            s0 += img[i, j]
    t0 = s0 / size
    while 1:
        for i in prange(img.shape[0], nogil=True):
            for j in range(img.shape[1]):
                if img[i, j] > t0:
                    s1 += img[i, j]
                    count += 1
        t1 = (s1 / count + (s0 - s1) / (size - count)) / 2
        if fdim(t0, t1) < 3.:
            break
        t0 = t1
        s1 = 0.
        count = 0
    return t1


cdef double _otsu_thresholding(ImageC1 img) noexcept nogil:
    """大津化阈值。

    :param img: 待处理图像
    :return: 阈值
    """

    cdef size_t size = img.shape[0] * img.shape[1]
    cdef size_t histongram[256]
    cdef uint8 t = 0
    cdef size_t length1 = 0, length2
    cdef double s0 = 0., s1 = 0.
    cdef double classify_var, max_var = 0.
    cdef size_t i, j
    memset(histongram, 0, sizeof(size_t) * 256)
    for i in prange(img.shape[0], nogil=True):
        for j in range(img.shape[1]):
            s0 += img[i, j]
            histongram[<uint8>img[i, j]] += 1
    for i in range(255):
        s1 += i * histongram[i]
        length1 += histongram[i]
        length2 = size - length1
        classify_var = length1 * length2 * cpow(s1 / length1 - (s0 - s1) / length2, 2.)
        if classify_var > max_var:
            max_var = classify_var
            t = i
    return t


cdef void _canny(
    ImageC1 img,
    double[:, ::1] out,
    double low_threshold,
    double high_threshold
) noexcept:
    """canny 边缘检测。

    :param img: 输入图像
    :param out: 保存结果的内存视图
    :param low_threshold, high_threshold: 高低阈值
    """

    cdef double kernel[3][3]
    cdef double[:, ::1] _kernel
    kernel[0] = [-1, 0, 1]
    kernel[1] = [-2, 0, 2]
    kernel[2] = [-1, 0, 1]
    _kernel = kernel
    cdef size_t size = sizeof(double) * out.shape[0] * out.shape[1]
    cdef double* dx_data = <double*>PyMem_Malloc(size)
    cdef double[:, ::1] dx = <double[:out.shape[0], :out.shape[1]]>dx_data
    cdef double* dy_data = <double*>PyMem_Malloc(size)
    cdef double[:, ::1] dy = <double[:out.shape[0], :out.shape[1]]>dy_data
    cdef double* temp_data = <double*>PyMem_Malloc(size)
    cdef double[:, ::1] temp = <double[:out.shape[0], :out.shape[1]]>temp_data
    _special_filter[ImageC1](img, dx, _kernel, NULL)
    _special_filter[ImageC1](img, dy, _kernel.T, NULL)
    cdef double pixel, point1, point2, slope
    cdef size_t i, j, k, l
    with nogil:
        for i in prange(out.shape[0]):
            for j in range(out.shape[1]):
                temp[i, j] = out[i, j] = hypot(dx[i, j], dy[i, j])
        for i in prange(1, out.shape[0] - 1):
            for j in range(1, out.shape[1] - 1):
                pixel = out[i, j]
                if dx[i, j] == 0.:
                    point1 = out[i, j - 1]
                    point2 = out[i, j + 1]
                elif dy[i, j] == 0.:
                    point1 = out[i - 1, j]
                    point2 = out[i + 1, j]
                else:
                    slope = dy[i, j] / dx[i, j]
                    if slope > 0:
                        point1 = out[i + 1, j + 1]
                        point2 = out[i - 1, j - 1]
                    else:
                        point1 = out[i - 1, j + 1]
                        point2 = out[i + 1, j - 1]
                    # if slope > 1.:
                    #     point1 = out[i + 1, j] * (1 + (out[i + 1, j + 1] - out[i + 1, j]) / slope)
                    #     point2 = out[i - 1, j] * (1 + (out[i - 1, j - 1] - out[i - 1, j]) / slope)
                    # elif slope < -1.:
                    #     point1 = out[i - 1, j] * (1 + (out[i - 1, j - 1] - out[i - 1, j]) / slope)
                    #     point2 = out[i + 1, j] * (1 + (out[i + 1, j + 1] - out[i + 1, j]) / slope)
                    # elif 0. < slope <= 1.:
                    #     point1 = out[i, j + 1] * (1 + (out[i + 1, j + 1] - out[i + 1, j]) * slope)
                    #     point2 = out[i, j - 1] * (1 + (out[i - 1, j - 1] - out[i, j - 1]) * slope)
                    # else:
                    #     point1 = out[i, j - 1] * (1 + (out[i + 1, j - 1] - out[i, j - 1]) * slope)
                    #     point2 = out[i, j + 1] * (1 + (out[i - 1, j - 1] - out[i, j + 1]) * slope)
                if point1 > pixel or point2 > pixel:
                    temp[i, j] = 0.
        temp[0, :] = 0.
        temp[:, 0] = 0.
        temp[out.shape[0] - 1, :] = 0.
        temp[:, out.shape[1] - 1] = 0.
        out[:] = temp
        for i in prange(1, out.shape[0] - 1):
            for j in range(1, out.shape[1] - 1):
                pixel = temp[i, j]
                if pixel <= low_threshold:
                    out[i, j] = 0.
                elif pixel > high_threshold:
                    out[i, j] = 255.
                else:
                    out[i, j] = 0.
                    for k in range(3):
                        l = i - 1 + k
                        if temp[l, j - 1] > high_threshold or \
                                temp[l, j] > high_threshold or \
                                temp[l, j + 1] > high_threshold:
                            out[i, j] = 255.
                            break
    PyMem_Free(temp_data)
    PyMem_Free(dy_data)
    PyMem_Free(dx_data)


cdef (double*, size_t) _hough(
    ImageC1 img,
    double r_step,
    double theta_step,
    size_t threshold
) noexcept nogil:
    """hough 直线检测。

    :param img: 输入图像
    :param r_step: 长度步进
    :param theta_step: 角度步进
    :param threshold: 阈值
    :return: 返回保存结果的指针
    """

    cdef double r_max = hypot(img.shape[0], img.shape[1])
    cdef size_t r_num = <size_t>(2 * r_max / r_step) + 1
    cdef size_t theta_num = <size_t>(pi / theta_step)
    cdef size_t size = sizeof(size_t) * theta_num * r_num
    cdef size_t* histongram = <size_t*>malloc(size)
    cdef size_t* temp = <size_t*>malloc(size)
    cdef size_t v
    cdef size_t preset_line_num = 10
    cdef size_t line_count = 0
    cdef double theta, r
    cdef double* lines = <double*>malloc(sizeof(double) * preset_line_num * 2)
    cdef size_t i, j, k, l
    memset(histongram, 0, size)
    for i in prange(theta_num, nogil=True):
        theta = theta_step * i
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j, k] > 0:
                    r = k * csin(theta) + j * ccos(theta)
                    histongram[i * theta_num + <size_t>((r + r_max) / r_step)] += 1
    memcpy(temp, histongram, size)
    for i in prange(1, theta_num - 1, nogil=True):
        for j in range(1, r_num - 1):
            v = temp[i * theta_num + j]
            for k in range(3):
                l = (i - 1 + k) * theta_num + j
                if temp[l - 1] > v or temp[l] > v or temp[l + 1] > v:
                    histongram[i * theta_num + j] = 0
                    break
    free(temp)
    for i in range(theta_num):
        for j in range(r_num):
            if histongram[i * theta_num + j] > threshold:
                lines[line_count * 2] = j * r_step - r_max
                lines[line_count * 2 + 1] = i * theta_step
                line_count += 1
                if not (line_count < preset_line_num):
                    preset_line_num *= 2
                    lines = <double*>realloc(
                        lines,
                        sizeof(double) * preset_line_num * 2
                    )
    free(histongram)
    return lines, line_count


def rgb2gray(ImageC3 img) -> cnp.ndarray:
    """用于外部调用的 RGB 图像灰度转换。

    :param img: RGB 图像
    :return: 以 numpy 数组形式返回灰度图像
    """

    cdef size_t rows = img.shape[0]
    cdef size_t cols = img.shape[1]
    cdef double* gray_data = <double*>PyMem_Malloc(
        sizeof(double) * rows * cols
    )
    cdef double[:, ::1] gray = <double[:rows, :cols]>gray_data
    _rgb2gray[ImageC3](img, gray)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &img.shape[0], cnp.NPY_FLOAT64, <void*>gray_data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def special_filter(ImageC1 img, Kernel kernel, bint norm) -> cnp.ndarray:
    """用于外部调用的空域滤波器。

    :param img: 待滤波图像
    :param kernel: 滤波核
    :param norm: 是否归一化
    :return: 以 numpy 数组形式返回滤波结果
    """

    cdef Py_ssize_t shape[2]
    shape[0] = img.shape[0] - kernel.shape[0] + 1
    shape[1] = img.shape[1] - kernel.shape[1] + 1
    cdef double* data = <double*>PyMem_Malloc(
        sizeof(double) * shape[0] * shape[1]
    )
    cdef double[:, ::1] view = <double[:shape[0], :shape[1]]>data
    cdef double range_[2]
    range_[0] = 0.
    range_[1] = 255.
    _special_filter[ImageC1](img, view, kernel, range_ if norm else NULL)
    cdef cnp.ndarray out = cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_FLOAT64, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def init_guassion_kernel(size_t kernel_rows, size_t kernel_cols, float sigma) -> cnp.ndarray:
    """初始化特定形状高斯核。

    :param kernel_rows, kernel_cols: 高斯核形状
    :param sigma: 高斯系数
    :return: 以 numpy 数组形式返回高斯核
    """

    cdef Py_ssize_t shape[2]
    shape[0] = kernel_rows
    shape[1] = kernel_cols
    cdef double* kernel_data = <double*>PyMem_Malloc(sizeof(double) * kernel_rows * kernel_cols)
    cdef double[:, ::1] kernel = <double[:kernel_rows, :kernel_cols]>kernel_data
    _init_guassion_kernel(kernel, sigma, 1)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_FLOAT64, <void*>kernel_data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def guassion_filter(
    ImageC1 img,
    size_t kernel_rows,
    size_t kernel_cols,
    float sigma=5.,
) -> cnp.ndarray:
    """用于外部调用的高斯滤波器。

    :param img: 待滤波图像
    :param kernel_rows, kernel_cols: 高斯核大小
    :param sigma: 高斯系数
    :return: 以 numpy 数组形式返回滤波结果
    """

    cdef Py_ssize_t shape[2]
    shape[0] = img.shape[0] - kernel_rows + 1
    shape[1] = img.shape[1] - kernel_cols + 1
    cdef double* data = <double*>PyMem_Malloc(sizeof(double) * shape[0] * shape[1])
    cdef double[:, ::1] view = <double[:shape[0], :shape[1]]>data
    cdef size_t kernel_shape[2]
    kernel_shape[0] = kernel_rows
    kernel_shape[1] = kernel_cols
    _guassion_filter[ImageC1](img, view, kernel_shape, sigma)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_FLOAT64, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def guassion_low_pass(ImageC1 img, float sigma=5.) -> cnp.ndarray:
    """用于外部调用的高斯低通滤波器。

    :param img: 待滤波图像
    :param sigma: 高斯系数
    :return: 以 numpy 数组形式返回滤波结果
    """

    cdef size_t rows = img.shape[0]
    cdef size_t cols = img.shape[1]
    cdef double* data = <double*>PyMem_Malloc(sizeof(double) * rows * cols)
    cdef double[:, ::1] view = <double[:rows, :cols]>data
    _guassion_low_pass[ImageC1](img, view, sigma)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &img.shape[0], cnp.NPY_FLOAT64, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def nolinear_filter(
    ImageC1 img,
    size_t kernel_rows,
    size_t kernel_cols,
    FilterType T
) -> cnp.ndarray:
    """用于外部调用的非线性滤波器。

    :param img: 待滤波图像
    :param kernel_rows, kernel_cols: 滤波核大小
    :param T: 滤波器类型
        0 -> mean
        1 -> median
        2 -> min
        3 -> max
    :return: 以 numpy 数组形式返回滤波结果
    """

    cdef Py_ssize_t shape[2]
    shape[0] = img.shape[0] - kernel_rows + 1
    shape[1] = img.shape[1] - kernel_cols + 1
    cdef uint8* data = <uint8*>PyMem_Malloc(
        sizeof(uint8) * shape[0] * shape[1]
    )
    cdef uint8[:, ::1] view = <uint8[:shape[0], :shape[1]]>data
    cdef size_t kernel_shape[2]
    kernel_shape[0] = kernel_rows
    kernel_shape[1] = kernel_cols
    cdef void (*filter_)(ImageC1, uint8[:, ::1], size_t*) noexcept nogil
    if T == FILTER_MEAN:
        filter_ = _mean_filter[ImageC1]
    elif T == FILTER_MEDIAN:
        filter_ = _median_filter[ImageC1]
    elif T == FILTER_MIN:
        filter_ = _min_filter[ImageC1]
    elif T == FILTER_MAX:
        filter_ = _max_filter[ImageC1]
    else:
        raise ValueError('got unexpected filter type')
    with nogil:
        filter_(img, view, &kernel_shape[0])
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_UINT8, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def morphology(
    uint8[:, :] img,
    size_t kernel_rows,
    size_t kernel_cols,
    MorphologyType T
) -> cnp.ndarray:
    """用于外部调用的形态学处理。

    :param img: 待处理图像
    :param kernel_rows, kernel_cols: 核大小
    :param T: 形态学处理方法
        0 -> 腐蚀
        1 -> 膨胀
    :return: 以 numpy 数组形式返回结果
    """

    cdef Py_ssize_t shape[2]
    shape[0] = img.shape[0] - kernel_rows + 1
    shape[1] = img.shape[1] - kernel_cols + 1
    cdef uint8* data = <uint8*>PyMem_Malloc(
        sizeof(uint8) * shape[0] * shape[1]
    )
    cdef uint8[:, ::1] view = <uint8[:shape[0], :shape[1]]>data
    cdef size_t kernel_shape[2]
    kernel_shape[0] = kernel_rows
    kernel_shape[1] = kernel_cols
    cdef bint flags
    if T == MORPH_ERODE:
        flags = 0
    elif T == MORPH_DILATE:
        flags = 1
    else:
        raise ValueError('got unexpected type')
    with nogil:
        _erode_dilate(img, view, kernel_shape, flags)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_UINT8, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def thresholding(ImageC1 img, ThresholdType T) -> tuple[float, cnp.ndarray]:
    """用于外部调用的阈值化处理。

    :param img: 待处理图像
    :param T: 阈值化方法
    :return: 以 numpy 数组形式返回结果
    """

    cdef uint8* data = <uint8*>PyMem_Malloc(
        sizeof(uint8) * img.shape[0] * img.shape[1]
    )
    cdef uint8[:, ::1] view = <uint8[:img.shape[0], :img.shape[1]]>data
    cdef double (*get_threshold)(ImageC1) noexcept nogil
    cdef double threshold
    cdef size_t i, j
    if T == THRESHOLD_GLOBAL:
        get_threshold = _global_thresholding[ImageC1]
    elif T == THRESHOLD_OTSU:
        get_threshold = _otsu_thresholding[ImageC1]
    else:
        raise ValueError('got unexpected type')
    with nogil:
        threshold = get_threshold(img)
        for i in prange(img.shape[0]):
            for j in range(img.shape[1]):
                view[i, j] = 255 if img[i, j] > threshold else 0
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &img.shape[0], cnp.NPY_UINT8, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return threshold, out


def canny(ImageC1 img, double low_threshold, double high_threshold) -> cnp.ndarray:
    """用于外部调用的 canny 边缘检测。

    :param img: 待处理图像
    :param low_threshold, high_threshold: 高低阈值
    :return: 以 numpy 数组形式返回结果
    """

    cdef Py_ssize_t shape[2]
    shape[0] = img.shape[0] - 2
    shape[1] = img.shape[1] - 2
    cdef double* data = <double*>PyMem_Malloc(
        sizeof(double) * shape[0] * shape[1]
    )
    cdef double[:, ::1] view = <double[:shape[0], :shape[1]]>data
    _canny[ImageC1](img, view, low_threshold, high_threshold)
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_FLOAT64, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def hough(
    ImageC1 img,
    double r_step,
    double theta_step,
    size_t threshold
) -> cnp.ndarray:
    """用于外部调用的 hough 直线检测。

    :param img: 输入图像
    :param r_step: 长度步进
    :param theta_step: 角度步进
    :param threshold: 阈值
    :return: 以 numpy 数组形式返回结果
    """

    cdef double* lines
    cdef size_t line_num
    cdef Py_ssize_t shape[2]
    with nogil:
        lines, line_num = _hough[ImageC1](img, r_step, theta_step, threshold)
        shape[0] = line_num
        shape[1] = 2
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, &shape[0], cnp.NPY_FLOAT64, <void*>lines
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out
