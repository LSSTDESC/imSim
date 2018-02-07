#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Defines the zernike polynomials in cartesian coordinates using the Noll
indexing scheme.
"""

from numpy import sqrt


def z_1(x, y):
    """The Zernike polynomial for n = 0, m = 0

    This zernike function represents the Piston effect

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 1
    """

    return 1


def z_2(x, y):
    """The Zernike polynomial for n = 1, m = 1

    This zernike function represents tilt in the y direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 2y
    """

    return 2 * y


def z_3(x, y):
    """The Zernike polynomial for n = 1, m = -1

    This zernike function represents tilt in the x direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 2x
    """

    return 2 * x


def z_4(x, y):
    """The Zernike Polynomial for n = 2, m = 0

    This zernike function represents defocus

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 3^(1/2) * 2x^2 + 2y^2 - 1
    """

    return sqrt(3) * (2 * x ** 2 + 2 * y ** 2 - 1)


def z_5(x, y):
    """The Zernike Polynomial for n = 2, m = -2

    This zernike function represents oblique astigmatism

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 6^(1/2) * 2xy
    """

    return sqrt(24) * x * y


def z_6(x, y):
    """The Zernike Polynomial for n = 2, m = 2

    This zernike function represents vertical astigmatism

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 6^(1/2) * (y^2 - x^2)
    """

    return sqrt(6) * (y ** 2 - x ** 2)


def z_7(x, y):
    """The Zernike Polynomial for n = 3, m = -1

    This zernike function represents coma in the y direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (3x^3 + 3xy^2 - 2x)
    """

    return sqrt(8) * (3 * x ** 3 + 3 * x * y ** 2 - 2 * x)


def z_8(x, y):
    """The Zernike Polynomial for n = 3, m = 1

    This zernike function represents coma in the x direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (3y^3 + 3yx^2 - 2y)
    """

    return sqrt(8) * (3 * y ** 3 + 3 * y * x ** 2 - 2 * y)


def z_9(x, y):
    """The Zernike Polynomial for n = 3, m = -3

    This zernike function represents trefoil in the y direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (3xy^2 - x^3)
    """

    return sqrt(8) * (3 * x * y ** 2 - x ** 3)


def z_10(x, y):
    """The Zernike Polynomial for n = 3, m = 3

    This zernike function represents trefoil in the x direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 8^(1/2) * (y^3 - 3yx^2)
    """

    return sqrt(8) * (y ** 3 - 3 * y * x ** 2)


def z_11(x, y):
    """The Zernike Polynomial for n = 4, m = 0

    This zernike function represents primary spherical

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 5^(1/2) * (1 - 6x^2 - 6y^2 + 6x^4 + 12x^2 y^2 + 6y^4)
    """

    return sqrt(5) * (1 - 6 * x ** 2 - 6 * y ** 2 + 6 * x ** 4
                      + 12 * x ** 2 * y ** 2 + 6 * y ** 4)


def z_12(x, y):
    """The Zernike Polynomial for n = 4, m = 2

    This zernike function represents secondary astigmatism in the y direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (3x^2 - 3y^2 - 4x^4 + 4y^4)
    """

    return sqrt(10) * (3 * x ** 2 - 3 * y ** 2 - 4 * x ** 4 + 4 * y ** 4)


def z_13(x, y):
    """The Zernike Polynomial for n = 4, m = -2

    This zernike function represents secondary astigmatism in the x direction

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (8x^3 y + 8xy^3 - 6xy)
    """

    return sqrt(10) * (8 * x ** 3 * y + 8 * x * y ** 3 - 6 * x * y)


def z_14(x, y):
    """The Zernike Polynomial for n = 4, m = 4

    This zernike function represents vertical quadrafoil

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (x^4 - 6x^2 y^2 + y^4)
    """

    return sqrt(10) * (x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4)


def z_15(x, y):
    """The Zernike Polynomial for n = 4, m = -4

    This zernike function represents oblique quadrafoil

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 10^(1/2) * (4xy^3 - 4x^3 y)
    """

    return sqrt(10) * (4 * x * y ** 3 - 4 * x ** 3 * y)


def z_16(x, y):
    """The Zernike Polynomial for n = 5, m = 1

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (3y - 12y^3 - 12x^2y + 10y^5 + 20x^2y^3 + 10x^4y)
    """

    return sqrt(12) * (3 * y - 12 * y ** 3 - 12 * x ** 2 * y
                       + 10 * y ** 5 + 20 * x ** 2 * y ** 3
                       + 10 * x ** 4 * y)


def z_17(x, y):
    """The Zernike Polynomial for n = 5, m = -1

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (3x - 12x^3 - 12xy^2 + 10X^5 + 20x^3y^2 + 10xy^4)
    """

    return sqrt(12) * (3 * x - 12 * x ** 3 - 12 * y ** 2 * x
                       + 10 * x ** 5 + 20 * y ** 2 * x ** 3
                       + 10 * y ** 4 * x)


def z_18(x, y):
    """The Zernike Polynomial for n = 5, m = 3

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (-4y^3 + 12x^2y + 5y^5 - 10x^2y^3 - 15x^4y)
    """

    return sqrt(12) * (-4 * y ** 3 + 12 * x ** 2 * y + 5 * y ** 5
                       - 10 * x ** 2 * y ** 3 - 15 * x ** 4 * y)


def z_19(x, y):
    """The Zernike Polynomial for n = 5, m = -3

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (-4x^3 + 12y^2 x + 5x^5 - 10y^2 x^3 - 15y^4 x)
    """

    return sqrt(12) * (-4 * x ** 3 + 12 * y ** 2 * x + 5 * x ** 5
                       - 10 * y ** 2 * x ** 3 - 15 * y ** 4 * x)


def z_20(x, y):
    """The Zernike Polynomial for n = 5, m = 5

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (y^5 - 10x^2 y^3+ 5x^4 y)
    """

    return sqrt(12) * (y ** 5 - 10 * x ** 2 * y ** 3 + 5 * x ** 4 * y)


def z_21(x, y):
    """The Zernike Polynomial for n = 5, m = -5

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 12^(1/2) * (x^5 - 10y^2 x^3+ 5y^4 x)
    """

    return sqrt(12) * (x ** 5 - 10 * y ** 2 * x ** 3 + 5 * y ** 4 * x)


def z_22(x, y):
    """The Zernike Polynomial for n = 6, m = 0

    @param [in] x is the x coordinate to evaluate the polynomial for

    @param [in] y is the y coordinate to evaluate the polynomial for

    @param [out] 7^(1/2) * (-1 + 12x^2 + 12y^2 - 30x^4 - 60x^2 y^2
                            - 30y^4 + 20x^6 + 60x^4 y^2 + 60x^2 y^4 + 20y^6)
    """

    return sqrt(7) * (-1 + 12 * x ** 2 + 12 * y ** 2 - 30 * x ** 4
                      - 60 * x ** 2 * y ** 2 - 30 * y ** 4 + 20 * x ** 6
                      + 60 * x ** 4 * y ** 2 + 60 * x ** 2 * y ** 4
                      + 20 * y ** 6)
