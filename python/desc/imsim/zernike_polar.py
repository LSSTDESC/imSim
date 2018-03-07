#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Defines the zernike polynomials in polar coordinates using the Noll
indexing scheme.
"""

from numpy import sqrt, cos, sin, arctan2


def z_1(r, t):
    """The Zernike polynomial for n = 0, m = 0

    This zernike function represents the Piston effect

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 2^(1/2)
    """

    return sqrt(2)


def z_2(r, t):
    """The Zernike polynomial for n = 1, m = 1

    This zernike function represents tilt in the x direction

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 2r cos(t)
    """

    return 2 * r * cos(t)


def z_3(r, t):
    """The Zernike polynomial for n = 1, m = -1

    This zernike function represents tilt in the y direction

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 2r sin(t)
    """

    return 2 * r * sin(t)


def z_4(r, t):
    """The Zernike Polynomial for n = 2, m = 0

    This zernike function represents defocus

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 3^(1/2) * (2r^2 - 1)
    """

    return sqrt(3) * (2 * r ** 2 - 1)


def z_5(r, t):
    """The Zernike Polynomial for n = 2, m = -2

    This zernike function represents oblique astigmatism

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 6^(1/2) * r^2 sin(2t)
    """

    return sqrt(6) * r ** 2 * sin(2 * t)


def z_6(r, t):
    """The Zernike Polynomial for n = 2, m = 2

    This zernike function represents vertical astigmatism

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 6^(1/2) * r^2 cos(2t)
    """

    return sqrt(6) * r ** 2 * cos(2 * t)


def z_7(r, t):
    """The Zernike Polynomial for n = 3, m = -1

    This zernike function represents vertical coma

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (3r^3 - 2r) sin(t)
    """

    return sqrt(8) * (3 * r ** 3 - 2 * r) * sin(t)


def z_8(r, t):
    """The Zernike Polynomial for n = 3, m = 1

    This zernike function represents horizontal coma

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (3r^3 - 2r) sin(t)
    """

    return sqrt(8) * (3 * r ** 3 - 2 * r) * cos(t)


def z_9(r, t):
    """The Zernike Polynomial for n = 3, m = -3

    This zernike function represents vertical trefoil

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (r^3 sin(3t))
    """

    return sqrt(8) * r ** 3 * sin(3 * t)


def z_10(r, t):
    """The Zernike Polynomial for n = 3, m = 3

    This zernike function represents oblique trefoil

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (r^3 cos(3t))
    """

    return sqrt(8) * r ** 3 * cos(3 * t)


def z_11(r, t):
    """The Zernike Polynomial for n = 4, m = 0

    This zernike function represents primary spherical

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 5^(1/2) * (6r^4 - 6r^2 + 1)
    """

    return sqrt(5) * (6 * r ** 4 - 6 * r ** 2 + 1)


def z_12(r, t):
    """The Zernike Polynomial for n = 4, m = 2

    This zernike function represents vertical secondary astigmatism

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (4r^4 - 3r^)cos(2t)
    """

    return sqrt(10) * (4 * r ** 4 - 3 * r ** 2) * cos(2 * t)


def z_13(r, t):
    """The Zernike Polynomial for n = 4, m = -2

    This zernike function represents oblique secondary astigmatism

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (4r^4 - 3r^)sin(2t)
    """

    return sqrt(10) * (4 * r ** 4 - 3 * r ** 2) * sin(2 * t)


def z_14(r, t):
    """The Zernike Polynomial for n = 4, m = 4

    This zernike function represents vertical quadrafoil

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (r^4 cos(4t))
    """

    return sqrt(10) * r ** 4 * cos(4 * t)


def z_15(r, t):
    """The Zernike Polynomial for n = 4, m = -4

    This zernike function represents oblique quadrafoil

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (r^4 sin(4t))
    """

    return sqrt(10) * r ** 4 * sin(4 * t)


def z_16(r, t):
    """The Zernike Polynomial for n = 5, m = 1

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (10r^5 - 12r^3 + 3r)cos(t)
    """

    return sqrt(12) * (10 * r ** 5 - 12 * r ** 3 + 3 * r) * cos(t)


def z_17(r, t):
    """The Zernike Polynomial for n = 5, m = -1

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (10r^5 - 12r^3 + 3r)sin(t)
    """

    return sqrt(12) * (10 * r ** 5 - 12 * r ** 3 + 3 * r) * sin(t)


def z_18(r, t):
    """The Zernike Polynomial for n = 5, m = 3

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (5r^5 - 4r^3)cos(3t)
    """

    return sqrt(12) * (5 * r ** 5 - 4 * r ** 3) * cos(3 * t)


def z_19(r, t):
    """The Zernike Polynomial for n = 5, m = -3

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (5r^5 - 4r^3)sin(3t)
    """

    return sqrt(12) * (5 * r ** 5 - 4 * r ** 3) * sin(3 * t)


def z_20(r, t):
    """The Zernike Polynomial for n = 5, m = 5

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (r^5 cos(5t))
    """

    return sqrt(12) * r ** 5 * cos(5 * t)


def z_21(r, t):
    """The Zernike Polynomial for n = 5, m = -5

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (r^5 sin(5t))
    """

    return sqrt(12) * r ** 5 * sin(5 * t)


def z_22(r, t):
    """The Zernike Polynomial for n = 6, m = 0

    @param [in] r is the radial coordinate to evaluate the polynomial for

    @param [in] t is the angular coordinate to evaluate the polynomial for

    @param [out] 7^(1/2) * (20r^6 - 30r^4 + 12r^2 - 1)
    """

    return sqrt(7) * (20 * r ** 6 - 30 * r ** 4 + 12 * r ** 2 - 1)


def gen_unoptimized(p):
    """
    A generator function for a superposition of zernike polynomials

    Uses given coefficients to generate a function representing a superposition
    of the first 22 Zernike polynomials in the Noll basis:
        f(r, t) = p[i-1] * z_i

    This function does not include any efficiency optimizations

    @param [in] p is an array of 22 polynomial coefficients

    @param [out] a function object
    """

    def fit_func(r, t):
        fit = (p[0] * z_1(r, t) + p[1] * z_2(r, t)
               + p[2] * z_3(r, t) + p[3] * z_4(r, t)
               + p[4] * z_5(r, t) + p[5] * z_6(r, t)
               + p[6] * z_7(r, t) + p[7] * z_8(r, t)
               + p[8] * z_9(r, t) + p[9] * z_10(r, t)
               + p[10] * z_11(r, t) + p[11] * z_12(r, t)
               + p[12] * z_13(r, t) + p[13] * z_14(r, t)
               + p[14] * z_15(r, t) + p[15] * z_16(r, t)
               + p[16] * z_17(r, t) + p[17] * z_18(r, t)
               + p[18] * z_19(r, t) + p[19] * z_20(r, t)
               + p[20] * z_21(r, t) + p[21] * z_22(r, t))

        return fit

    return fit_func


def gen_superposition(p):
    """
    An optimized generator function for a superposition of zernike polynomials

    Uses given coefficients to generate a function representing a superposition
    of the first 22 Zernike polynomials in the Noll basis:
        f(r, t) = p[i-1] * z_i

    @param [in] p is an array of 22 polynomial coefficients

    @param [out] a function object
    """

    s_6 = sqrt(6)
    s_8 = sqrt(8)
    s_10 = sqrt(10)
    s_12 = sqrt(12)

    def out_func(r, t):
        # Store in memory any values that are needed more than once
        r_2 = r ** 2
        r_3 = r ** 3
        r_4 = r ** 4
        r_5 = r ** 5
        r_6 = r ** 6

        s_t, c_t = sin(t), cos(t)
        s_2t, c_2t = sin(2 * t), cos(2 * t)
        s_3t, c_3t = sin(3 * t), cos(3 * t)

        fit = (
            # n = 0
            sqrt(2) * p[0]

            # n = 1
            + 2 * r * (p[1] * c_t + p[2] * s_t)

            # n = 2
            + p[3] * sqrt(3) * (2 * r_2 - 1)
            + s_6 * r_2 * (p[4] * s_2t + p[5] * c_2t)

            # n = 3
            + s_8 * (
                (3 * r_3 - 2 * r) * (p[6] * s_t + p[7] * c_t)
                + r_3 * (p[8] * s_3t + p[9] * c_3t)
            )

            # n = 4
            + sqrt(5) * p[10] * (6 * r_4 - 6 * r_2 + 1)
            + s_10 * (
                (4 * r_4 - 3 * r_2) * (p[11] * c_2t + p[12] * s_2t)
                + r_4 * (p[13] * cos(4 * t) + p[14] * sin(4 * t))
            )

            # n = 5
            + s_12 * (
                (10 * r_5 - 12 * r_3 + 3 * r) * (p[15] * c_t + p[16] * s_t)
                + (5 * r_5 - 4 * r_3) * (p[17] * c_3t + p[18] * s_3t)
                + r_5 * (p[19] * cos(5 * t) + p[20] * sin(5 * t))
            )

            # n = 6 (only goes as far as n = 0)
            + p[21] * sqrt(7) * (20 * r_6 - 30 * r_4 + 12 * r_2 - 1)
        )
        return fit

    return out_func


if __name__ == '__main__':
    # Test for algebraic mistakes that may have occurred in gen_superposition
    import numpy as np

    p = np.ones(22)
    assert np.isclose(gen_superposition(p)(1.5, 0.7),
                      gen_unoptimized(p)(1.5, 0.7))
