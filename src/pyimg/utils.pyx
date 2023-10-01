# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cpython.mem cimport PyMem_Malloc
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.stdio cimport printf
from libc.time cimport time
from libc.math cimport (
    log as clog,
    sin as csin,
    cos as ccos,
    tan as ctan,
    sqrt as csqrt,
    pi
)
from cython.parallel cimport prange
cimport numpy as cnp

cdef double RAND_MAX_ = RAND_MAX + 1


ctypedef unsigned char uint8


cdef fused Image:
    uint8[:, :]
    uint8[:, :, :]


cpdef enum NoiseType:
    NOISE_RANDOM
    NOISE_SALT
    NOISE_GUASSION


cdef double _gen_guass(double mean, double std) noexcept nogil:
    """生成高斯随机数。

    :param mean, std: 高斯分布的均值和标准差
    """

    cdef double u[2]
    u[0] = rand() / RAND_MAX_
    u[1] = rand() / RAND_MAX_
    cdef double Z = csqrt(-2 * clog(u[0])) * csin(2 * pi * u[1])
    return mean + Z * std


cdef void _add_random_noise(
    Image img,
    Image out,
    size_t noise_point_num
) noexcept nogil:
    """添加随机噪声。

    :param img: 需要添加噪声的图像
    :param out: 保存结果的内存视图
    :param noise_point_num: 噪声点个数
    """

    srand(<unsigned int>time(NULL))
    cdef size_t i, j
    out[:] = img
    for _ in range(noise_point_num):
        i = rand() % img.shape[0]
        j = rand() % img.shape[1]
        out[i, j] = rand() % 128 + 128


cdef void _add_salt_noise(Image img, Image out, float snr) noexcept nogil:
    """添加椒盐噪声。

    :param img: 需要添加噪声的图像
    :param out: 保存结果的内存视图
    :param snr: 信噪比
    """

    cdef size_t i, j
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if rand() / RAND_MAX_ > snr:
                if rand() % 10 < 5:
                    out[i, j] = 255
                else:
                    out[i, j] = 0
            else:
                out[i, j] = img[i, j]


cdef void _add_guassion_noise(
    Image img,
    Image out,
    double mean,
    double std
) noexcept nogil:
    """添加高斯噪声。

    :param img: 需要添加噪声的图像
    :param out: 保存结果的内存视图
    :param mean: 均值
    :param std: 标准差
    """

    cdef double pixel
    cdef size_t i, j, k
    if Image is uint8[:, :]:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j] + _gen_guass(mean, std) * 255
                if pixel > 255.:
                    out[i, j] = 255
                elif pixel < 0.:
                    out[i, j] = 0
                else:
                    out[i, j] = <uint8>pixel
    else:
        for k in prange(img.shape[2], nogil=True):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    pixel = img[i, j, k] + _gen_guass(mean, std) * 255
                    if pixel > 255.:
                        out[i, j, k] = 255
                    elif pixel < 0.:
                        out[i, j, k] = 0
                    else:
                        out[i, j, k] = <uint8>pixel


def add_noise(
    Image img,
    NoiseType T,
    *,
    size_t noise_point_num=1000,
    float snr=0.8,
    double mean=0.,
    double std=0.5,
) -> cnp.ndarray:
    """添加噪声。

    :param img: 需要添加噪声的图像
    :param T: 噪声类型
        0 -> 随机噪声
        1 -> 椒盐噪声
        2 -> 高斯噪声
    :param noise_point_num: 随机噪声的噪声点个数
    :param snr: 椒盐噪声的信噪比
    :param mean, std: 高斯噪声的均值和标准差
    :return: 以 numpy 数组形式返回加噪图像
    """

    cdef size_t size = 1
    cdef size_t ndim = 0
    while img.shape[ndim] != 0:
        size *= img.shape[ndim]
        ndim += 1
    cdef uint8* data = <uint8*>PyMem_Malloc(sizeof(uint8) * size)
    cdef Image view
    if Image is uint8[:, :]:
        view = <uint8[:img.shape[0], :img.shape[1]]>data
    else:
        view = <uint8[:img.shape[0], :img.shape[1], :img.shape[2]]>data
    with nogil:
        if T == NOISE_RANDOM:
            _add_random_noise[Image](img, view, noise_point_num)
        elif T == NOISE_SALT:
            _add_salt_noise[Image](img, view, snr)
        elif T == NOISE_GUASSION:
            _add_guassion_noise[Image](img, view, mean, std)
        else:
            with gil:
                raise ValueError('got unexpected noise type')
    cdef cnp.ndarray out = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        ndim, &img.shape[0], cnp.NPY_UINT8, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(out, cnp.NPY_ARRAY_OWNDATA)
    return out


def line2points(
    double r,
    double theta,
    tuple image_shape
) -> list[size_t, size_t, size_t, size_t]:
    cdef size_t rows = image_shape[0]
    cdef size_t cols = image_shape[1]
    cdef size_t points[4]
    cdef double x0, y0, x, y, tan_theta
    cdef size_t num = 0
    if theta == 0.:
        points[0] = <size_t>r
        points[1] = 0
        points[2] = <size_t>r
        points[3] = cols - 1
    elif theta == pi / 2:
        points[0] = 0
        points[1] = <size_t>r
        points[2] = rows - 1
        points[3] = <size_t>r
    else:
        x0 = r * ccos(theta)
        y0 = r * csin(theta)
        tan_theta = ctan(theta)
        x = y0 * tan_theta + x0
        if 0 <= x <= rows - 1:
            points[num] = <size_t>x
            points[num + 1] = 0
            num += 2
        x = (y0 - cols + 1) * tan_theta + x0
        if 0 <= x <= rows - 1:
            points[num] = <size_t>x
            points[num + 1] = cols - 1
            num += 2
        y = x0 / tan_theta + y0
        if 0 <= y <= cols - 1:
            points[num] = 0
            points[num + 1] = <size_t>y
            num += 2
        y = (x0 - rows + 1) / tan_theta + y0
        if 0 <= y <= cols - 1:
            points[num] = rows - 1
            points[num + 1] = <size_t>y
            num += 2
        if num != 4:
            printf('unknow error occured\n')
    return points
