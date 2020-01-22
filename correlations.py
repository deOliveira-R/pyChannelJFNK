from math import tau
import numpy as np
import thermo_Na
import unittest

np.seterr(invalid='ignore')


def fric_factor(geometry, v, T):
    """
    SODIUM FRICTION FACTOR MODEL FOR LIQUID SODIUM
    -------------------------------------------------------------------------
    Engel model (corrected by Todreas) for wire-wrapped fuel pins
    Source: Eq. (9) in A. Chenu et al., "Pressure drop modelling and
    comparisons with experiments for single- and tow-phase sodium flow."
    Nuclear Engineering and Design, 241, pp. 3898-3909 (2011)
    -------------------------------------------------------------------------

    :param geometry: dictionary containing information on channel
    :param v: Sodium velocity in m.s^-1
    :param T: Sodium temperature in K
    :return: Single-phase friction factor (dimensionless)
    """
    De = geometry['De']
    Re = thermo_Na.rho(T)*v*De/thermo_Na.mu(T)

    return (Re < 400)*99/Re\
           + (Re > 5000)*0.48/Re**0.25 \
           + (Re >= 400) * (Re <= 5000) * np.nan_to_num((0.48 / Re ** 0.25 * np.sqrt(Re - 400) / 4600) + 99 / Re * np.sqrt(1 - (Re - 400) / 4600))


def h_gap(qp):
    """
    GAP CONDUCTANCE MODEL
    -------------------------------------------------------------------------
    Source: A. Ponomarev and K. Mikityuk, "Analysis of hypothitical
    unprotected loss of flow in Superphénix start-up core: sensitivity to
    modeling details," ICONE27, May 19-24, 2019, Ibaraki, Japan (2019)
    -------------------------------------------------------------------------

    :param qp: Linear power density in W.m^-1
    :return: Gap conductance in W.m^-2.K^-1
    """
    return np.minimum(3*(1E3 - qp/100 + (qp/(100*10))**2 + (qp/(100*100))**3), 2.3E4)


def h_Na(geometry, v, T):
    """
    SODIUM SINGLE PHASE HEAT TRANSFER COEFFICIENT MODEL
    -------------------------------------------------------------------------
    Source: Eq. (14) in K. Mikityuk, "Heat transfer to liquid metal: Review
    of data and correlations for tube bundles," Nuclear Engineering and
    Design, 239, pp. 680-687 (2009).
    -------------------------------------------------------------------------

    :param geometry: dictionary containing information on channel
    :param v: Sodium velocity in m.s^-1
    :param T: Sodium temperature in K
    :return: Single-phase heat transfer coefficient in W.m^-2.K^-1
    """
    De = geometry['De']
    Rco = geometry['Rco']
    p = geometry['pin_pitch']

    Pe = v*De*thermo_Na.rho(T)*thermo_Na.cp(T)/thermo_Na.k(T)
    Nu = 0.047*(1 - np.exp(-3.8*(p/Rco - 1)))*(Pe**0.77 + 250)

    return thermo_Na.k(T)*Nu/De


class CorrelationTest(unittest.TestCase):

    def setUp(self):
        self.geometry = {'pin_pitch': 9.8E-3,
                         'De': 0.003958735792072682,
                         'Rco': 0.00425}
        self.v = np.array([6, 6.5, 7, 7.5], dtype='float32')
        self.T = np.array([600, 650, 700, 750], dtype='float32')
        self.qp = 1e5 * np.sin((np.arange(5) + 0.5) / 5 * tau / 2)

    def test_fric(self):
        reference_fric = np.array([0.0301055, 0.028856, 0.02778773, 0.02686283])
        self.assertTrue(np.allclose(fric_factor(self.geometry, self.v, self.T), reference_fric, rtol=1e-5))

    def test_h_gap(self):
        reference_h_gap = np.array([5.02622e3, 2.17967e4, 2.30000e4, 2.17967e4, 5.02622e3])
        self.assertTrue(np.allclose(h_gap(self.qp), reference_h_gap, rtol=1e-5))

    def test_h_Na(self):
        reference_h_Na = np.array([305776.6, 301872.1, 297739.2, 293410.5])
        self.assertTrue(np.allclose(h_Na(self.geometry, self.v, self.T), reference_h_Na, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
