import numpy as np
import unittest


def k_SS(T):
    """
    STAINLESS STEEL THERMAL CONDUCTIVITY MODEL
    -------------------------------------------------------------------------
    Model developed after a second-order polynomial fit of the AISI 316
    Stainless Steel data from
    Source: Table 4 in C.Y. Ho and T.K. Chu, "Electrical resistivity and
    thermal conductivity of nine selected AISI Stainless Steel," CINDAS
    report 45 (1977)
    -------------------------------------------------------------------------

    :param T: Stainless Steel temperature in K
    :return: Stainless Steel thermal conductivity in W.m^-1.K^-1
    """
    return -1.5809e-06 * T**2 + 0.0169 * T + 8.8025


def k_fuel(T):
    """
    FUEL PELLET THERMAL CONDUCTIVITY MODEL
    -------------------------------------------------------------------------
    Source: Eqs. (6.1) and (6.3)-(6.7) in S.G. Popov et al., "Themophysical
    properties of MOX and UO2 fules including the effects of irradiation,"
    Report ORNL/TM-2000/351, ORNL, TN, USA (2000)
    -------------------------------------------------------------------------

    :param T: Fuel pellet temperature in K
    :return: Fuel pellet thermal conductivity in W.m^-1.K^-1
    """

    B = 3  # Fuel burnup[at. %]
    p = 0.174  # Porosity [1]
    OM = 1.98  # Stoichiometric ratio[1]

    omega = 1.09 / B ** 3.265 + 0.0643 * np.sqrt(T / B)
    FD = omega / np.arctan(1 / omega)
    FP = 1 + 0.019 * B / (3 - 0.019 * B) / (1 + np.exp(- (T - 1200) / 100))
    FM = (1 - p) / (1 + 2 * p)
    FR = 1 - 0.2 / (1 + np.exp((T - 900) / 80))
    x = 2 - OM
    A = 2.85 * x + 0.035
    C = (-7.15 * x + 2.86) * 1E-4
    lambda_0 = 1.1579 / (A + C * T) + 2.3434E11 * T ** (-5 / 2) * np.exp(-16350 / T)

    return lambda_0 * FD * FP * FM * FR


