import numpy as np
import unittest


def rho(T):
    """
    SODIUM DENSITY MODEL
    -------------------------------------------------------------------------
    Source: Eq. (3-2) in V. Sobolev, Database of thermophysical properties of
    liquid metal coolants for GEN-IV - Sodium, lead, lead-bismuth eutectic
    (and bismuth). Report SCK.CEN-BLG-1069, SCK.CEN, Mol, Belgium (2011)
    -------------------------------------------------------------------------

    Warning: The correlation is only valid at 10^5 Pa (dependence on pressure
    around 10^5 Pa can nevertheless be neglected).

    :param T: Sodium temperature in K
    :return: Sodium density rho_Na in kg.m^-3
    """
    return 1014 - 0.235 * T


def cp(T):
    """
    SODIUM SPECIFIC HEAT MODEL
    -------------------------------------------------------------------------
    Source: Eq. (3-35) in V. Sobolev, Database of thermophysical properties of
    liquid metal coolants for GEN-IV - Sodium, lead, lead-bismuth eutectic
    (and bismuth). Report SCK.CEN-BLG-1069, SCK.CEN, Mol, Belgium (2011).
    -------------------------------------------------------------------------

    Warning: The correlation is only valid at 10^5 Pa (dependence on pressure
    around 10^5 Pa can nevertheless be neglected).

    :param T: Sodium temperature in K
    :return: Sodium specific heat in J.kg^-1.K^-1
    """
    return -3.001E6*(T**-2) + 1658 - 0.8479*T + 4.454E-4*T**2


def k(T):
    """
    SODIUM THERMAL CONDUCTIVITY MODEL
    -------------------------------------------------------------------------
    Source: Eq. (4-15a) in V. Sobolev, Database of thermophysical properties of
    liquid metal coolants for GEN-IV - Sodium, lead, lead-bismuth eutectic
    (and bismuth). Report SCK.CEN-BLG-1069, SCK.CEN, Mol, Belgium (2011)
    -------------------------------------------------------------------------

    Warning: The correlation is only valid at 10^5 Pa (dependence on pressure
    around 10^5 Pa can nevertheless be neglected).

    :param T: Sodium temperature in K
    :return: Sodium thermal conductivity in W.m^-1.K^-1
    """
    return 104 - 0.047 * T


def mu(T):
    """
    SODIUM DYNAMIC VISCOSITY MODEL
    -------------------------------------------------------------------------
    Source: Eq. (4-5) in V. Sobolev, Database of thermophysical properties of
    liquid metal coolants for GEN-IV - Sodium, lead, lead-bismuth eutectic
    (and bismuth). Report SCK.CEN-BLG-1069, SCK.CEN, Mol, Belgium (2011)
    -------------------------------------------------------------------------

    Warning: The correlation is only valid at 10^5 Pa (dependence on pressure
    around 10^5 Pa can nevertheless be neglected).

    :param T: Sodium temperature K
    :return: Sodium dynamic viscosity in Pa.s
    """
    return np.exp(556.835/T - 0.3958*np.log(T) - 6.4406)


def h(T):
    """
    SODIUM ENTHALPY MODEL
    -------------------------------------------------------------------------
    Enthalpy increment from the melting temperature of sodium at 10^5 Pa,
    which is T_M,Na = 371 K.

    Delta h (T_Na) = integral between T_M,Na and T_Na of c_P (T)
    with for sodium at 10^5 Pa
    c_P (T) = -3.001E6*T^-2 + 1658 - 0.8479*T + 4.454E-4*T^2

    Source: Eq. (3-35) in V. Sobolev, Database of thermophysical properties of
    liquid metal coolants for GEN-IV - Sodium, lead, lead-bismuth eutectic
    (and bismuth). Report SCK.CEN-BLG-1069, SCK.CEN, Mol, Belgium (2011)
    -------------------------------------------------------------------------

    Warnings:
    - The correlation is only valid at 10^5 Pa (dependence on pressure
      around 10^5 Pa can nevertheless be neglected).
    - The model assumes that h_Na (T_M,Na) = 0 (enthalpy being a relative
      quantity).

    :param T: Sodium temperature in K
    :return: Sodium specific enthalpy in J.kg^-1
    """
    TMelt = 371  # Sodium melting temperature in K at 10^5 Pa
    return 3.001E6 * (T**-1 - TMelt**-1)\
           + 1658 * (T - TMelt)\
           - 1/2 * 0.8479 * (T**2 - TMelt**2)\
           + 1/3 * 4.454E-4 * (T**3 - TMelt**3)


class ThermoTest(unittest.TestCase):

    def setUp(self):
        self.T = np.array([600, 650, 700, 750], dtype='float32')

    def test_rho(self):
        reference_rho = np.array([873, 861.25, 849.5, 837.75])
        self.assertTrue(np.allclose(rho(self.T), reference_rho, rtol=1e-5))

    def test_cp(self):
        reference_cp = np.array([1301.27, 1287.94, 1276.59, 1267.28])
        self.assertTrue(np.allclose(cp(self.T), reference_cp, rtol=1e-5))

    def test_k(self):
        reference_k = np.array([75.8, 73.45, 71.1, 68.75])
        self.assertTrue(np.allclose(k(self.T), reference_k, rtol=1e-5))

    def test_mu(self):
        reference_mu = np.array([3.2088e-4, 2.8945e-4, 2.644e-4, 2.4399e-4])
        self.assertTrue(np.allclose(mu(self.T), reference_mu, rtol=1e-5))

    def test_h(self):
        reference_h = np.array([306812, 371535, 435640, 499228])
        self.assertTrue(np.allclose(h(self.T), reference_h, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
