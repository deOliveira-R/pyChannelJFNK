import XS_parametrization as xs
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fsolve
import unittest
import matplotlib.pyplot as plt


class nkSolver:
    """
    Neutron diffusion solver

    Solves a neutron diffusion problem in 1D
    """

    def __init__(self, discretization, boundary_conditions, library):
        # this must be eliminated from nk solver.
        # T should be given in the right format so that nk is oblivious to th.
        self.T_in = boundary_conditions['T_in']
        self.qp_ave = boundary_conditions['qp_ave']

        self.xs = xs.XSParametrization(library)

        self.axial_nodes = discretization['axial_nodes']
        self.energy_groups = self.xs.eg
        self.phi_size = self.axial_nodes * self.energy_groups
        self.domain_size = self.phi_size + 1

        self.Dz = discretization['Dz'] * 100

        self.b = np.tile(np.concatenate([[1], np.zeros(self.axial_nodes - 1)]), self.energy_groups)
        self.c = np.tile(np.concatenate([[0], np.ones(self.axial_nodes - 2), [0]]), self.energy_groups)
        self.t = np.tile(np.concatenate([np.zeros(self.axial_nodes - 1), [1]]), self.energy_groups)

    def split_solution(self, sol):
        [phi, k] = np.split(sol, [self.phi_size])
        return phi, k

    def neg_pos_surf(self, T_rel):
        T_tmp = np.concatenate([[self.T_in], T_rel * self.T_in])
        T_m = T_tmp[:-1]
        T_p = T_tmp[1:]
        return T_m, T_p, T_tmp

    def interpolate_T(self, T_rel):
        T_m, T_p = self.neg_pos_surf(T_rel)[0:2]
        return (T_p + T_m) / 2

    def calculate_Ds(self, D):
        Dc = D

        Db = np.concatenate([[0], Dc[:-1]])
        Db[::self.axial_nodes] = np.zeros(self.energy_groups)

        Dt = np.concatenate([Dc[1:], [0]])
        Dt[self.axial_nodes - 1::self.axial_nodes] = np.zeros(self.energy_groups)

        return Db, Dc, Dt

    def calculate_coef(self, D):
        Db, Dc, Dt = self.calculate_Ds(D)

        # for index i
        a = self.c * ((2 * Dt * Dc) / (self.Dz ** 2 * (Dt + Dc)) + (2 * Db * Dc) / (self.Dz ** 2 * (Db + Dc))) \
            + self.t * ((2 * Db * Dc) / (self.Dz ** 2 * (Db + Dc)) + (
                    1 / self.Dz * (1 / 2 / (1 + (self.Dz / (4 * Dc)))))) \
            + self.b * ((2 * Dt * Dc) / (self.Dz ** 2 * (Dt + Dc)) + (
                    1 / self.Dz * (1 / 2 / (1 + (self.Dz / (4 * Dc))))))

        # for index i-1
        b = self.c * (- (2 * Dt * Dc) / (self.Dz ** 2 * (Dt + Dc))) \
            + self.b * (- (2 * Dt * Dc) / (self.Dz ** 2 * (Dt + Dc)))

        # for index i+1
        c = self.c * (- (2 * Db * Dc) / (self.Dz ** 2 * (Db + Dc))) \
            + self.t * (- (2 * Db * Dc) / (self.Dz ** 2 * (Db + Dc)))

        return a, b, c

    def res(self, sol, T_rel, T_fuel_rel):
        """
        NEUTRON DIFFUSION RESIDUAL VECTOR

        :param sol: a vector representing the volume-averaged neutron fluxes and
                    the corresponding eigenvalue. Elements are numbered from the
                    highest neutron energies (inlet to outlet) to lowest energies.
                    The eigenvalue is the last element.
        :param T_rel: a vector representing the surface-averaged coolant temperature
                      from the surface right above the inlet (inlet + 1) to the outlet
                      in relative terms to the inlet coolant temperature T_in.
        :param T_fuel_rel: vector representing the node-averaged fuel temperature from
                           inlet to outlet in relative terms to the inlet sodium
                           temperature T_in.
        :return: the node-wise residuals of the multi-group neutron diffusion
                 equations with the eigenvalue residual last.
        """
        self.xs.update(self.interpolate_T(T_rel), T_fuel_rel * self.T_in)
        scat = self.xs.scat()
        abs = self.xs.abs()
        fis = self.xs.fis()
        nu = self.xs.nu()
        chi = self.xs.chi()
        D = self.xs.diffusion()

        a, b, c = self.calculate_coef(D)

        phi, k = self.split_solution(sol)

        P1 = sp.vstack(
            [sp.hstack([sp.eye(self.axial_nodes, self.axial_nodes)] * self.energy_groups)] * self.energy_groups)
        P2 = sp.diags(nu * fis * phi)
        P3 = np.sum(P1 * P2, axis=1)
        res_P = - chi / k * np.squeeze(np.asarray(P3))

        # basically total XS * phi, or total reaction rate
        res = a * phi + b * np.concatenate([phi[1:], [0]]) + c * np.concatenate([[0], phi[:-1]]) \
            + abs * phi + np.squeeze(np.asarray(scat.sum(axis=0))) * phi \
            - scat * phi \
            + res_P
        # - chi / k * np.sum(np.tile(sp.eye(self.axial_nodes, self.axial_nodes), (self.energy_groups, self.energy_groups)) * sp.spdiags(nu * fis * phi, 0, self.phi_size, self.phi_size), axis=1)

        res = np.concatenate([res, [-np.inner(phi, phi) + self.phi_size]])

        return res

    def calculate_qp(self, phi):
        """
        ESTIMATION OF THE LINEAR POWER DENSITY

        :param phi: vector representing the volume-averaged neutron fluxes in arbitrary units. The numbering of the
                    elements is from the highest neutron energies (from the inlet to the outlet) to the lowest neutron
                    energies (from the inlet to the outlet).
        :return: vector representing the node-wise linear power density (in W/m) from the inlet to the outlet.
        """
        fis = self.xs.fis()
        kappa = self.xs.kappa()

        qp_shape = sp.hstack([sp.eye(self.axial_nodes, self.axial_nodes)] * self.energy_groups) * (kappa * fis * phi)
        qp = self.qp_ave * qp_shape / (np.sum(qp_shape) / self.axial_nodes)

        return qp

class NKTest(unittest.TestCase):

    def setUp(self):
        self.discretization = {'axial_nodes': 25,
                               'Dz': 0.064}
        self.boundary_conditions = {'T_in': 673,
                                    'v_in': 7.5,
                                    'qp_ave': 3e4}
        self.solver = nkSolver(self.discretization, self.boundary_conditions, 'XS_data.hdf5')

        self.T_rel = (self.boundary_conditions['T_in'] + np.arange(self.discretization['axial_nodes']) * 600
                      / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['T_in']
        self.T_fuel_rel = (self.boundary_conditions['T_in'] + np.arange(self.discretization['axial_nodes']) * 600
                           / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['T_in']
        self.sol = np.ones(self.solver.phi_size + 1)

    def test_bct(self):
        reference_b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.allclose(self.solver.b, reference_b, rtol=1e-5))
        reference_c = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        self.assertTrue(np.allclose(self.solver.c, reference_c, rtol=1e-5))
        reference_t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(np.allclose(self.solver.t, reference_t, rtol=1e-5))

    def test_split(self):
        [phi, k] = self.solver.split_solution(self.sol)
        self.assertEqual(len(phi), self.solver.phi_size)

    def test_m_p(self):
        T_m, T_p, T_tmp = self.solver.neg_pos_surf(self.T_rel)
        self.assertEqual(len(T_m), self.discretization['axial_nodes'])
        self.assertEqual(len(T_p), self.discretization['axial_nodes'])
        self.assertEqual(len(T_tmp), self.discretization['axial_nodes'] + 1)
        reference_T = np.array([673.0, 673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0,
                                898.0, 923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0,
                                1148.0, 1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        self.assertTrue(np.allclose(T_tmp, reference_T, rtol=1e-5))
        reference_T_m = np.array([673.0, 673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0,
                                  898.0, 923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0,
                                  1148.0, 1173.0, 1198.0, 1223.0, 1248.0])
        self.assertTrue(np.allclose(T_m, reference_T_m, rtol=1e-5))
        reference_T_p = np.array([673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0, 898.0,
                                  923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0, 1148.0,
                                  1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        self.assertTrue(np.allclose(T_p, reference_T_p, rtol=1e-5))

    def test_interpolate(self):
        T = self.solver.interpolate_T(self.T_rel)
        self.assertEqual(len(T), self.discretization['axial_nodes'])
        reference_T = np.array([673.0, 685.5, 710.5, 735.5, 760.5, 785.5, 810.5, 835.5, 860.5, 885.5,
                                910.5, 935.5, 960.5, 985.5, 1010.5, 1035.5, 1060.5, 1085.5, 1110.5, 1135.5,
                                1160.5, 1185.5, 1210.5, 1235.5, 1260.5])
        self.assertTrue(np.allclose(T, reference_T, rtol=1e-5))

    def test_calculate_Ds(self):
        self.solver.xs.update(self.solver.interpolate_T(self.T_rel), self.T_fuel_rel * self.boundary_conditions['T_in'])
        D = self.solver.xs.diffusion()
        Db, Dc, Dt = self.solver.calculate_Ds(D)
        reference_Dc = [2.99347, 2.99461, 2.99688, 2.99916, 3.00144,
                        3.00373, 3.00602, 3.00831, 3.01061, 3.01291,
                        3.01521, 3.01752, 3.01983, 3.02214, 3.02446,
                        3.02678, 3.02911, 3.03144, 3.03377, 3.03610,
                        3.03844, 3.04078, 3.04313, 3.04548, 3.04783,
                        2.21299, 2.21401, 2.21605, 2.21809, 2.22014,
                        2.22219, 2.22425, 2.22631, 2.22837, 2.23044,
                        2.23251, 2.23459, 2.23666, 2.23875, 2.24083,
                        2.24292, 2.24502, 2.24712, 2.24922, 2.25133,
                        2.25344, 2.25555, 2.25767, 2.25979, 2.26192,
                        1.75429, 1.75528, 1.75726, 1.75925, 1.76124,
                        1.76324, 1.76524, 1.76725, 1.76926, 1.77127,
                        1.77329, 1.77531, 1.77734, 1.77938, 1.78141,
                        1.78346, 1.78550, 1.78756, 1.78961, 1.79167,
                        1.79374, 1.79581, 1.79789, 1.79997, 1.80205,
                        1.41680, 1.41734, 1.41845, 1.41955, 1.42066,
                        1.42177, 1.42288, 1.42400, 1.42511, 1.42623,
                        1.42735, 1.42847, 1.42959, 1.43071, 1.43184,
                        1.43296, 1.43409, 1.43522, 1.43636, 1.43749,
                        1.43863, 1.43976, 1.44090, 1.44205, 1.44319,
                        1.15505, 1.15542, 1.15619, 1.15697, 1.15775,
                        1.15852, 1.15930, 1.16008, 1.16086, 1.16164,
                        1.16243, 1.16321, 1.16400, 1.16478, 1.16557,
                        1.16636, 1.16715, 1.16794, 1.16873, 1.16952,
                        1.17031, 1.17111, 1.17191, 1.17270, 1.17350,
                        9.72721e-1, 9.72955e-1, 9.73525e-1, 9.74096e-1, 9.74668e-1,
                        9.75240e-1, 9.75813e-1, 9.76386e-1, 9.76961e-1, 9.77536e-1,
                        9.78111e-1, 9.78688e-1, 9.79265e-1, 9.79842e-1, 9.80421e-1,
                        9.81000e-1, 9.81580e-1, 9.82160e-1, 9.82741e-1, 9.83323e-1,
                        9.83906e-1, 9.84489e-1, 9.85073e-1, 9.85657e-1, 9.86243e-1,
                        7.08458e-1, 7.08561e-1, 7.08932e-1, 7.09304e-1, 7.09676e-1,
                        7.10049e-1, 7.10422e-1, 7.10795e-1, 7.11169e-1, 7.11543e-1,
                        7.11918e-1, 7.12293e-1, 7.12668e-1, 7.13044e-1, 7.13420e-1,
                        7.13797e-1, 7.14174e-1, 7.14551e-1, 7.14929e-1, 7.15307e-1,
                        7.15685e-1, 7.16064e-1, 7.16444e-1, 7.16823e-1, 7.17204e-1,
                        7.47149e-1, 7.47056e-1, 7.47112e-1, 7.47169e-1, 7.47225e-1,
                        7.47282e-1, 7.47339e-1, 7.47395e-1, 7.47452e-1, 7.47508e-1,
                        7.47565e-1, 7.47622e-1, 7.47678e-1, 7.47735e-1, 7.47792e-1,
                        7.47848e-1, 7.47905e-1, 7.47962e-1, 7.48019e-1, 7.48075e-1,
                        7.48132e-1, 7.48189e-1, 7.48245e-1, 7.48302e-1, 7.48359e-1]
        self.assertTrue(np.allclose(Dc, reference_Dc, rtol=1e-5))
        reference_Db = [0, 2.99347, 2.99461, 2.99688, 2.99916,
                        3.00144, 3.00373, 3.00602, 3.00831, 3.01061,
                        3.01291, 3.01521, 3.01752, 3.01983, 3.02214,
                        3.02446, 3.02678, 3.02911, 3.03144, 3.03377,
                        3.03610, 3.03844, 3.04078, 3.04313, 3.04548,
                        0, 2.21299, 2.21401, 2.21605, 2.21809,
                        2.22014, 2.22219, 2.22425, 2.22631, 2.22837,
                        2.23044, 2.23251, 2.23459, 2.23666, 2.23875,
                        2.24083, 2.24292, 2.24502, 2.24712, 2.24922,
                        2.25133, 2.25344, 2.25555, 2.25767, 2.25979,
                        0, 1.75429, 1.75528, 1.75726, 1.75925,
                        1.76124, 1.76324, 1.76524, 1.76725, 1.76926,
                        1.77127, 1.77329, 1.77531, 1.77734, 1.77938,
                        1.78141, 1.78346, 1.78550, 1.78756, 1.78961,
                        1.79167, 1.79374, 1.79581, 1.79789, 1.79997,
                        0, 1.41680, 1.41734, 1.41845, 1.41955,
                        1.42066, 1.42177, 1.42288, 1.42400, 1.42511,
                        1.42623, 1.42735, 1.42847, 1.42959, 1.43071,
                        1.43184, 1.43296, 1.43409, 1.43522, 1.43636,
                        1.43749, 1.43863, 1.43976, 1.44090, 1.44205,
                        0, 1.15505, 1.15542, 1.15619, 1.15697,
                        1.15775, 1.15852, 1.15930, 1.16008, 1.16086,
                        1.16164, 1.16243, 1.16321, 1.16400, 1.16478,
                        1.16557, 1.16636, 1.16715, 1.16794, 1.16873,
                        1.16952, 1.17031, 1.17111, 1.17191, 1.17270,
                        0, 9.72721e-1, 9.72955e-1, 9.73525e-1, 9.74096e-1,
                        9.74668e-1, 9.75240e-1, 9.75813e-1, 9.76386e-1, 9.76961e-1,
                        9.77536e-1, 9.78111e-1, 9.78688e-1, 9.79265e-1, 9.79842e-1,
                        9.80421e-1, 9.81000e-1, 9.81580e-1, 9.82160e-1, 9.82741e-1,
                        9.83323e-1, 9.83906e-1, 9.84489e-1, 9.85073e-1, 9.85657e-1,
                        0, 7.08458e-1, 7.08561e-1, 7.08932e-1, 7.09304e-1,
                        7.09676e-1, 7.10049e-1, 7.10422e-1, 7.10795e-1, 7.11169e-1,
                        7.11543e-1, 7.11918e-1, 7.12293e-1, 7.12668e-1, 7.13044e-1,
                        7.13420e-1, 7.13797e-1, 7.14174e-1, 7.14551e-1, 7.14929e-1,
                        7.15307e-1, 7.15685e-1, 7.16064e-1, 7.16444e-1, 7.16823e-1,
                        0, 7.47149e-1, 7.47056e-1, 7.47112e-1, 7.47169e-1,
                        7.47225e-1, 7.47282e-1, 7.47339e-1, 7.47395e-1, 7.47452e-1,
                        7.47508e-1, 7.47565e-1, 7.47622e-1, 7.47678e-1, 7.47735e-1,
                        7.47792e-1, 7.47848e-1, 7.47905e-1, 7.47962e-1, 7.48019e-1,
                        7.48075e-1, 7.48132e-1, 7.48189e-1, 7.48245e-1, 7.48302e-1]
        self.assertTrue(np.allclose(Db, reference_Db, rtol=1e-5))
        reference_Dt = [2.99461, 2.99688, 2.99916, 3.00144, 3.00373,
                        3.00602, 3.00831, 3.01061, 3.01291, 3.01521,
                        3.01752, 3.01983, 3.02214, 3.02446, 3.02678,
                        3.02911, 3.03144, 3.03377, 3.03610, 3.03844,
                        3.04078, 3.04313, 3.04548, 3.04783, 0,
                        2.21401, 2.21605, 2.21809, 2.22014, 2.22219,
                        2.22425, 2.22631, 2.22837, 2.23044, 2.23251,
                        2.23459, 2.23666, 2.23875, 2.24083, 2.24292,
                        2.24502, 2.24712, 2.24922, 2.25133, 2.25344,
                        2.25555, 2.25767, 2.25979, 2.26192, 0,
                        1.75528, 1.75726, 1.75925, 1.76124, 1.76324,
                        1.76524, 1.76725, 1.76926, 1.77127, 1.77329,
                        1.77531, 1.77734, 1.77938, 1.78141, 1.78346,
                        1.78550, 1.78756, 1.78961, 1.79167, 1.79374,
                        1.79581, 1.79789, 1.79997, 1.80205, 0,
                        1.41734, 1.41845, 1.41955, 1.42066, 1.42177,
                        1.42288, 1.42400, 1.42511, 1.42623, 1.42735,
                        1.42847, 1.42959, 1.43071, 1.43184, 1.43296,
                        1.43409, 1.43522, 1.43636, 1.43749, 1.43863,
                        1.43976, 1.44090, 1.44205, 1.44319, 0,
                        1.15542, 1.15619, 1.15697, 1.15775, 1.15852,
                        1.15930, 1.16008, 1.16086, 1.16164, 1.16243,
                        1.16321, 1.16400, 1.16478, 1.16557, 1.16636,
                        1.16715, 1.16794, 1.16873, 1.16952, 1.17031,
                        1.17111, 1.17191, 1.17270, 1.17350, 0,
                        9.72955e-1, 9.73525e-1, 9.74096e-1, 9.74668e-1, 9.75240e-1,
                        9.75813e-1, 9.76386e-1, 9.76961e-1, 9.77536e-1, 9.78111e-1,
                        9.78688e-1, 9.79265e-1, 9.79842e-1, 9.80421e-1, 9.81000e-1,
                        9.81580e-1, 9.82160e-1, 9.82741e-1, 9.83323e-1, 9.83906e-1,
                        9.84489e-1, 9.85073e-1, 9.85657e-1, 9.86243e-1, 0,
                        7.08561e-1, 7.08932e-1, 7.09304e-1, 7.09676e-1, 7.10049e-1,
                        7.10422e-1, 7.10795e-1, 7.11169e-1, 7.11543e-1, 7.11918e-1,
                        7.12293e-1, 7.12668e-1, 7.13044e-1, 7.13420e-1, 7.13797e-1,
                        7.14174e-1, 7.14551e-1, 7.14929e-1, 7.15307e-1, 7.15685e-1,
                        7.16064e-1, 7.16444e-1, 7.16823e-1, 7.17204e-1, 0,
                        7.47056e-1, 7.47112e-1, 7.47169e-1, 7.47225e-1, 7.47282e-1,
                        7.47339e-1, 7.47395e-1, 7.47452e-1, 7.47508e-1, 7.47565e-1,
                        7.47622e-1, 7.47678e-1, 7.47735e-1, 7.47792e-1, 7.47848e-1,
                        7.47905e-1, 7.47962e-1, 7.48019e-1, 7.48075e-1, 7.48132e-1,
                        7.48189e-1, 7.48245e-1, 7.48302e-1, 7.48359e-1, 0]
        self.assertTrue(np.allclose(Dt, reference_Dt, rtol=1e-5))

    def test_calculate_coef(self):
        self.solver.xs.update(self.solver.interpolate_T(self.T_rel), self.T_fuel_rel * self.boundary_conditions['T_in'])
        a, b, c, = self.solver.calculate_coef(self.solver.xs.diffusion())
        reference_a = [1.24009e-1, 1.46235e-1, 1.46332e-1, 1.46443e-1, 1.46555e-1,
                       1.46667e-1, 1.46778e-1, 1.46890e-1, 1.47002e-1, 1.47115e-1,
                       1.47227e-1, 1.47340e-1, 1.47453e-1, 1.47566e-1, 1.47679e-1,
                       1.47792e-1, 1.47906e-1, 1.48019e-1, 1.48133e-1, 1.48247e-1,
                       1.48361e-1, 1.48476e-1, 1.48590e-1, 1.48705e-1, 1.25612e-1,
                       9.93829e-2, 1.08118e-1, 1.08206e-1, 1.08305e-1, 1.08405e-1,
                       1.08506e-1, 1.08606e-1, 1.08707e-1, 1.08807e-1, 1.08908e-1,
                       1.09009e-1, 1.09111e-1, 1.09212e-1, 1.09314e-1, 1.09416e-1,
                       1.09518e-1, 1.09620e-1, 1.09723e-1, 1.09825e-1, 1.09928e-1,
                       1.10031e-1, 1.10134e-1, 1.10238e-1, 1.10341e-1, 1.00954e-1,
                       8.37007e-2, 8.57191e-2, 8.58038e-2, 8.59009e-2, 8.59982e-2,
                       8.60956e-2, 8.61934e-2, 8.62913e-2, 8.63895e-2, 8.64878e-2,
                       8.65864e-2, 8.66853e-2, 8.67843e-2, 8.68836e-2, 8.69831e-2,
                       8.70829e-2, 8.71828e-2, 8.72830e-2, 8.73834e-2, 8.74841e-2,
                       8.75850e-2, 8.76861e-2, 8.77875e-2, 8.78891e-2, 8.53524e-2,
                       7.12869e-2, 6.92131e-2, 6.92601e-2, 6.93141e-2, 6.93682e-2,
                       6.94224e-2, 6.94767e-2, 6.95310e-2, 6.95855e-2, 6.96400e-2,
                       6.96946e-2, 6.97493e-2, 6.98041e-2, 6.98589e-2, 6.99139e-2,
                       6.99689e-2, 7.00241e-2, 7.00793e-2, 7.01346e-2, 7.01899e-2,
                       7.02454e-2, 7.03010e-2, 7.03566e-2, 7.04124e-2, 7.22698e-2,
                       6.09579e-2, 5.64219e-2, 5.64547e-2, 5.64926e-2, 5.65305e-2,
                       5.65685e-2, 5.66065e-2, 5.66446e-2, 5.66827e-2, 5.67209e-2,
                       5.67591e-2, 5.67974e-2, 5.68357e-2, 5.68741e-2, 5.69126e-2,
                       5.69511e-2, 5.69896e-2, 5.70282e-2, 5.70668e-2, 5.71055e-2,
                       5.71443e-2, 5.71831e-2, 5.72219e-2, 5.72608e-2, 6.16957e-2,
                       5.32892e-2, 4.75117e-2, 4.75354e-2, 4.75633e-2, 4.75912e-2,
                       4.76191e-2, 4.76471e-2, 4.76751e-2, 4.77032e-2, 4.77312e-2,
                       4.77593e-2, 4.77875e-2, 4.78157e-2, 4.78439e-2, 4.78721e-2,
                       4.79004e-2, 4.79287e-2, 4.79570e-2, 4.79854e-2, 4.80138e-2,
                       4.80423e-2, 4.80707e-2, 4.80993e-2, 4.81278e-2, 5.38634e-2,
                       4.12739e-2, 3.46010e-2, 3.46158e-2, 3.46340e-2, 3.46522e-2,
                       3.46704e-2, 3.46886e-2, 3.47068e-2, 3.47250e-2, 3.47433e-2,
                       3.47616e-2, 3.47799e-2, 3.47982e-2, 3.48166e-2, 3.48350e-2,
                       3.48533e-2, 3.48718e-2, 3.48902e-2, 3.49086e-2, 3.49271e-2,
                       3.49456e-2, 3.49641e-2, 3.49826e-2, 3.50011e-2, 4.16859e-2,
                       4.31087e-2, 3.64792e-2, 3.64801e-2, 3.64828e-2, 3.64856e-2,
                       3.64884e-2, 3.64911e-2, 3.64939e-2, 3.64967e-2, 3.64994e-2,
                       3.65022e-2, 3.65050e-2, 3.65077e-2, 3.65105e-2, 3.65133e-2,
                       3.65160e-2, 3.65188e-2, 3.65216e-2, 3.65243e-2, 3.65271e-2,
                       3.65299e-2, 3.65327e-2, 3.65354e-2, 3.65382e-2, 4.31661e-2]
        self.assertTrue(np.allclose(a, reference_a, rtol=1e-5))
        reference_b = [-7.30966e-2, -7.31383e-2, -7.31939e-2, -7.32496e-2, -7.33053e-2,
                       -7.33612e-2, -7.34171e-2, -7.34731e-2, -7.35293e-2, -7.35854e-2,
                       -7.36417e-2, -7.36981e-2, -7.37545e-2, -7.38111e-2, -7.38677e-2,
                       -7.39244e-2, -7.39812e-2, -7.40381e-2, -7.40951e-2, -7.41521e-2,
                       -7.42093e-2, -7.42665e-2, -7.43239e-2, -7.43813e-2, 0,
                       -5.40405e-2, -5.40778e-2, -5.41277e-2, -5.41777e-2, -5.42277e-2,
                       -5.42779e-2, -5.43281e-2, -5.43784e-2, -5.44289e-2, -5.44794e-2,
                       -5.45300e-2, -5.45807e-2, -5.46315e-2, -5.46824e-2, -5.47334e-2,
                       -5.47845e-2, -5.48356e-2, -5.48869e-2, -5.49383e-2, -5.49898e-2,
                       -5.50413e-2, -5.50930e-2, -5.51447e-2, -5.51966e-2, 0,
                       -4.28414e-2, -4.28777e-2, -4.29262e-2, -4.29747e-2, -4.30234e-2,
                       -4.30722e-2, -4.31211e-2, -4.31702e-2, -4.32193e-2, -4.32685e-2,
                       -4.33179e-2, -4.33674e-2, -4.34170e-2, -4.34667e-2, -4.35165e-2,
                       -4.35664e-2, -4.36164e-2, -4.36666e-2, -4.37169e-2, -4.37672e-2,
                       -4.38177e-2, -4.38684e-2, -4.39191e-2, -4.39700e-2, 0,
                       -3.45965e-2, -3.46166e-2, -3.46436e-2, -3.46706e-2, -3.46977e-2,
                       -3.47248e-2, -3.47519e-2, -3.47791e-2, -3.48064e-2, -3.48336e-2,
                       -3.48610e-2, -3.48883e-2, -3.49157e-2, -3.49432e-2, -3.49707e-2,
                       -3.49982e-2, -3.50258e-2, -3.50534e-2, -3.50811e-2, -3.51088e-2,
                       -3.51366e-2, -3.51644e-2, -3.51922e-2, -3.52201e-2, 0,
                       -2.82040e-2, -2.82179e-2, -2.82368e-2, -2.82558e-2, -2.82748e-2,
                       -2.82938e-2, -2.83128e-2, -2.83318e-2, -2.83509e-2, -2.83700e-2,
                       -2.83891e-2, -2.84083e-2, -2.84275e-2, -2.84467e-2, -2.84659e-2,
                       -2.84852e-2, -2.85044e-2, -2.85237e-2, -2.85431e-2, -2.85624e-2,
                       -2.85818e-2, -2.86012e-2, -2.86207e-2, -2.86402e-2, 0,
                       -2.37509e-2, -2.37607e-2, -2.37747e-2, -2.37886e-2, -2.38026e-2,
                       -2.38166e-2, -2.38306e-2, -2.38446e-2, -2.38586e-2, -2.38726e-2,
                       -2.38867e-2, -2.39008e-2, -2.39149e-2, -2.39290e-2, -2.39431e-2,
                       -2.39573e-2, -2.39714e-2, -2.39856e-2, -2.39998e-2, -2.40140e-2,
                       -2.40283e-2, -2.40425e-2, -2.40568e-2, -2.40710e-2, 0,
                       -1.72976e-2, -1.73034e-2, -1.73124e-2, -1.73215e-2, -1.73306e-2,
                       -1.73397e-2, -1.73488e-2, -1.73580e-2, -1.73671e-2, -1.73762e-2,
                       -1.73854e-2, -1.73945e-2, -1.74037e-2, -1.74129e-2, -1.74221e-2,
                       -1.74313e-2, -1.74405e-2, -1.74497e-2, -1.74589e-2, -1.74682e-2,
                       -1.74774e-2, -1.74867e-2, -1.74959e-2, -1.75052e-2, 0,
                       -1.82398e-2, -1.82394e-2, -1.82407e-2, -1.82421e-2, -1.82435e-2,
                       -1.82449e-2, -1.82463e-2, -1.82476e-2, -1.82490e-2, -1.82504e-2,
                       -1.82518e-2, -1.82532e-2, -1.82546e-2, -1.82560e-2, -1.82573e-2,
                       -1.82587e-2, -1.82601e-2, -1.82615e-2, -1.82629e-2, -1.82642e-2,
                       -1.82656e-2, -1.82670e-2, -1.82684e-2, -1.82698e-2, 0]
        self.assertTrue(np.allclose(b, reference_b, rtol=1e-5))
        reference_c = [0, -7.30966e-2, -7.31383e-2, -7.31939e-2, -7.32496e-2,
                       -7.33053e-2, -7.33612e-2, -7.34171e-2, -7.34731e-2, -7.35293e-2,
                       -7.35854e-2, -7.36417e-2, -7.36981e-2, -7.37545e-2, -7.38111e-2,
                       -7.38677e-2, -7.39244e-2, -7.39812e-2, -7.40381e-2, -7.40951e-2,
                       -7.41521e-2, -7.42093e-2, -7.42665e-2, -7.43239e-2, -7.43813e-2,
                       0, -5.40405e-2, -5.40778e-2, -5.41277e-2, -5.41777e-2,
                       -5.42277e-2, -5.42779e-2, -5.43281e-2, -5.43784e-2, -5.44289e-2,
                       -5.44794e-2, -5.45300e-2, -5.45807e-2, -5.46315e-2, -5.46824e-2,
                       -5.47334e-2, -5.47845e-2, -5.48356e-2, -5.48869e-2, -5.49383e-2,
                       -5.49898e-2, -5.50413e-2, -5.50930e-2, -5.51447e-2, -5.51966e-2,
                       0, -4.28414e-2, -4.28777e-2, -4.29262e-2, -4.29747e-2,
                       -4.30234e-2, -4.30722e-2, -4.31211e-2, -4.31702e-2, -4.32193e-2,
                       -4.32685e-2, -4.33179e-2, -4.33674e-2, -4.34170e-2, -4.34667e-2,
                       -4.35165e-2, -4.35664e-2, -4.36164e-2, -4.36666e-2, -4.37169e-2,
                       -4.37672e-2, -4.38177e-2, -4.38684e-2, -4.39191e-2, -4.39700e-2,
                       0, -3.45965e-2, -3.46166e-2, -3.46436e-2, -3.46706e-2,
                       -3.46977e-2, -3.47248e-2, -3.47519e-2, -3.47791e-2, -3.48064e-2,
                       -3.48336e-2, -3.48610e-2, -3.48883e-2, -3.49157e-2, -3.49432e-2,
                       -3.49707e-2, -3.49982e-2, -3.50258e-2, -3.50534e-2, -3.50811e-2,
                       -3.51088e-2, -3.51366e-2, -3.51644e-2, -3.51922e-2, -3.52201e-2,
                       0, -2.82040e-2, -2.82179e-2, -2.82368e-2, -2.82558e-2,
                       -2.82748e-2, -2.82938e-2, -2.83128e-2, -2.83318e-2, -2.83509e-2,
                       -2.83700e-2, -2.83891e-2, -2.84083e-2, -2.84275e-2, -2.84467e-2,
                       -2.84659e-2, -2.84852e-2, -2.85044e-2, -2.85237e-2, -2.85431e-2,
                       -2.85624e-2, -2.85818e-2, -2.86012e-2, -2.86207e-2, -2.86402e-2,
                       0, -2.37509e-2, -2.37607e-2, -2.37747e-2, -2.37886e-2,
                       -2.38026e-2, -2.38166e-2, -2.38306e-2, -2.38446e-2, -2.38586e-2,
                       -2.38726e-2, -2.38867e-2, -2.39008e-2, -2.39149e-2, -2.39290e-2,
                       -2.39431e-2, -2.39573e-2, -2.39714e-2, -2.39856e-2, -2.39998e-2,
                       -2.40140e-2, -2.40283e-2, -2.40425e-2, -2.40568e-2, -2.40710e-2,
                       0, -1.72976e-2, -1.73034e-2, -1.73124e-2, -1.73215e-2,
                       -1.73306e-2, -1.73397e-2, -1.73488e-2, -1.73580e-2, -1.73671e-2,
                       -1.73762e-2, -1.73854e-2, -1.73945e-2, -1.74037e-2, -1.74129e-2,
                       -1.74221e-2, -1.74313e-2, -1.74405e-2, -1.74497e-2, -1.74589e-2,
                       -1.74682e-2, -1.74774e-2, -1.74867e-2, -1.74960e-2, -1.75052e-2,
                       0, -1.82398e-2, -1.82394e-2, -1.82407e-2, -1.82421e-2,
                       -1.82435e-2, -1.82449e-2, -1.82463e-2, -1.82476e-2, -1.82490e-2,
                       -1.82504e-2, -1.82518e-2, -1.82532e-2, -1.82546e-2, -1.82559e-2,
                       -1.82573e-2, -1.82587e-2, -1.82601e-2, -1.82615e-2, -1.82629e-2,
                       -1.82642e-2, -1.82656e-2, -1.82670e-2, -1.82684e-2, -1.82698e-2]
        self.assertTrue(np.allclose(c, reference_c, rtol=1e-5))

    def test_res(self):
        res = self.solver.res(self.sol, self.T_rel, self.T_fuel_rel)
        self.assertEqual(len(res), self.discretization['axial_nodes'] * self.solver.energy_groups + 1)
        reference_res = [7.24107e-2, 2.14852e-2, 2.14578e-2, 2.14304e-2, 2.14030e-2,
                         2.13755e-2, 2.13481e-2, 2.13207e-2, 2.12933e-2, 2.12659e-2,
                         2.12384e-2, 2.12110e-2, 2.11836e-2, 2.11562e-2, 2.11288e-2,
                         2.11014e-2, 2.10739e-2, 2.10465e-2, 2.10191e-2, 2.09917e-2,
                         2.09643e-2, 2.09368e-2, 2.09094e-2, 2.08820e-2, 7.20853e-2,
                         2.16431e-2, -2.36986e-2, -2.36979e-2, -2.36973e-2, -2.36966e-2,
                         -2.36959e-2, -2.36953e-2, -2.36946e-2, -2.36940e-2, -2.36933e-2,
                         -2.36926e-2, -2.36920e-2, -2.36913e-2, -2.36906e-2, -2.36900e-2,
                         -2.36893e-2, -2.36886e-2, -2.36880e-2, -2.36873e-2, -2.36866e-2,
                         -2.36860e-2, -2.36853e-2, -2.36846e-2, -2.36840e-2, 2.20743e-2,
                         4.94579e-3, -3.59097e-2, -3.59026e-2, -3.58955e-2, -3.58884e-2,
                         -3.58813e-2, -3.58743e-2, -3.58672e-2, -3.58601e-2, -3.58530e-2,
                         -3.58459e-2, -3.58388e-2, -3.58318e-2, -3.58247e-2, -3.58176e-2,
                         -3.58105e-2, -3.58034e-2, -3.57963e-2, -3.57892e-2, -3.57822e-2,
                         -3.57751e-2, -3.57680e-2, -3.57609e-2, -3.57538e-2, 5.63574e-3,
                         2.28780e-2, -1.38073e-2, -1.37974e-2, -1.37875e-2, -1.37776e-2,
                         -1.37677e-2, -1.37577e-2, -1.37478e-2, -1.37379e-2, -1.37280e-2,
                         -1.37181e-2, -1.37082e-2, -1.36983e-2, -1.36884e-2, -1.36785e-2,
                         -1.36685e-2, -1.36586e-2, -1.36487e-2, -1.36388e-2, -1.36289e-2,
                         -1.36190e-2, -1.36091e-2, -1.35992e-2, -1.35892e-2, 2.34703e-2,
                         3.37664e-2, 1.01223e-3, 1.01142e-3, 1.01061e-3, 1.00979e-3,
                         1.00898e-3, 1.00817e-3, 1.00735e-3, 1.00654e-3, 1.00573e-3,
                         1.00491e-3, 1.00410e-3, 1.00329e-3, 1.00247e-3, 1.00166e-3,
                         1.00084e-3, 1.00003e-3, 9.99218e-4, 9.98404e-4, 9.97591e-4,
                         9.96777e-4, 9.95964e-4, 9.95151e-4, 9.94337e-4, 3.40491e-2,
                         3.58533e-2, 6.31612e-3, 6.31736e-3, 6.31859e-3, 6.31983e-3,
                         6.32106e-3, 6.32230e-3, 6.32354e-3, 6.32477e-3, 6.32601e-3,
                         6.32724e-3, 6.32848e-3, 6.32972e-3, 6.33095e-3, 6.33219e-3,
                         6.33342e-3, 6.33466e-3, 6.33590e-3, 6.33713e-3, 6.33837e-3,
                         6.33960e-3, 6.34084e-3, 6.34207e-3, 6.34331e-3, 3.61369e-2,
                         2.32446e-2, -7.23042e-4, -7.12040e-4, -7.01038e-4, -6.90036e-4,
                         -6.79034e-4, -6.68032e-4, -6.57030e-4, -6.46028e-4, -6.35026e-4,
                         -6.24023e-4, -6.13021e-4, -6.02019e-4, -5.91017e-4, -5.80015e-4,
                         -5.69013e-4, -5.58011e-4, -5.47009e-4, -5.36006e-4, -5.25004e-4,
                         -5.14002e-4, -5.03000e-4, -4.91998e-4, -4.80996e-4, 2.37107e-2,
                         5.36297e-2, 2.88283e-2, 2.88985e-2, 2.89687e-2, 2.90389e-2,
                         2.91092e-2, 2.91794e-2, 2.92496e-2, 2.93198e-2, 2.93900e-2,
                         2.94602e-2, 2.95304e-2, 2.96006e-2, 2.96708e-2, 2.97410e-2,
                         2.98113e-2, 2.98815e-2, 2.99517e-2, 3.00219e-2, 3.00921e-2,
                         3.01623e-2, 3.02325e-2, 3.03027e-2, 3.03729e-2, 5.53395e-2,
                         0]
        self.assertTrue(np.allclose(res, reference_res, rtol=1e-5))

    def test_solve(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol, args=(self.T_rel, self.T_fuel_rel),
                                     full_output=True, xtol=1e-6)
        reference_sol = np.array([5.23285e-2, 9.37556e-2, 1.36144e-1, 1.77422e-1, 2.16392e-1,
                                  2.52249e-1, 2.84387e-1, 3.12315e-1, 3.35626e-1, 3.53993e-1,
                                  3.67162e-1, 3.74958e-1, 3.77284e-1, 3.74123e-1, 3.65537e-1,
                                  3.51668e-1, 3.32729e-1, 3.09008e-1, 2.80863e-1, 2.48719e-1,
                                  2.13079e-1, 1.74552e-1, 1.33927e-1, 9.23682e-2, 5.18501e-2,
                                  1.07143e-1, 1.99980e-1, 2.92215e-1, 3.81211e-1, 4.65004e-1,
                                  5.42058e-1, 6.11121e-1, 6.71145e-1, 7.21257e-1, 7.60749e-1,
                                  7.89073e-1, 8.05852e-1, 8.10876e-1, 8.04107e-1, 7.85678e-1,
                                  7.55889e-1, 7.15203e-1, 6.64239e-1, 6.03763e-1, 5.34693e-1,
                                  4.58096e-1, 3.75228e-1, 2.87594e-1, 1.97091e-1, 1.06211e-1,
                                  2.24236e-1, 4.32886e-1, 6.35535e-1, 8.29548e-1, 1.01190,
                                  1.17969, 1.33028, 1.46139, 1.57106, 1.65772,
                                  1.72012, 1.75740, 1.76907, 1.75502, 1.71550,
                                  1.65114, 1.56292, 1.45218, 1.32059, 1.17015,
                                  1.00316, 8.22228e-1, 6.30204e-1, 4.30021e-1, 2.24185e-1,
                                  2.67968e-1, 5.39707e-1, 7.99720e-1, 1.04641, 1.27709,
                                  1.48874, 1.67836, 1.84323, 1.98097, 2.08962,
                                  2.16764, 2.21398, 2.22804, 2.20970, 2.15932,
                                  2.07771, 1.96614, 1.82632, 1.66035, 1.47070,
                                  1.26015, 1.03166, 7.88198e-1, 5.32302e-1, 2.65381e-1,
                                  2.07641e-1, 4.36496e-1, 6.53639e-1, 8.58297e-1, 1.04883,
                                  1.22314, 1.37905, 1.51444, 1.62745, 1.71651,
                                  1.78039, 1.81823, 1.82955, 1.81427, 1.77268,
                                  1.70547, 1.61368, 1.49871, 1.36224, 1.20624,
                                  1.03284, 8.44240e-1, 6.42466e-1, 4.29072e-1, 2.04700e-1,
                                  1.33745e-1, 2.91352e-1, 4.40220e-1, 5.79859e-1, 7.09332e-1,
                                  8.27406e-1, 9.32734e-1, 1.02400, 1.10000, 1.15972,
                                  1.20237, 1.22739, 1.23450, 1.22366, 1.19509,
                                  1.14927, 1.08692, 1.00899, 9.16610e-1, 8.11067e-1,
                                  6.93734e-1, 5.65973e-1, 4.28995e-1, 2.83693e-1, 1.30455e-1,
                                  1.27725e-1, 2.93686e-1, 4.49268e-1, 5.94323e-1, 7.28146e-1,
                                  8.49683e-1, 9.57716e-1, 1.05102, 1.12845, 1.18904,
                                  1.23203, 1.25690, 1.26339, 1.25151, 1.22151,
                                  1.17392, 1.10950, 1.02921, 9.34209e-1, 8.25767e-1,
                                  7.05211e-1, 5.73804e-1, 4.32603e-1, 2.82268e-1, 1.22814e-1,
                                  1.15351e-2, 2.66370e-2, 4.07300e-2, 5.37026e-2, 6.54841e-2,
                                  7.59988e-2, 8.51641e-2, 9.28988e-2, 9.91314e-2, 1.03806e-1,
                                  1.06887e-1, 1.08360e-1, 1.08232e-1, 1.06535e-1, 1.03321e-1,
                                  9.86629e-2, 9.26520e-2, 8.53951e-2, 7.70105e-2, 6.76237e-2,
                                  5.73598e-2, 4.63362e-2, 3.46528e-2, 2.23877e-2, 9.61542e-3,
                                  1.12575154301880])
        self.assertTrue(np.allclose(sol, reference_sol, rtol=1e-5))
        [phi, k] = self.solver.split_solution(sol)
        plt.plot(phi)
        plt.show()

    def test_qp(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol, args=(self.T_rel, self.T_fuel_rel),
                                     full_output=True, xtol=1e-6)
        [phi, k] = self.solver.split_solution(sol)
        qp = self.solver.calculate_qp(phi)
        reference_qp = np.array([5.39601e3, 1.09596e4, 1.63155e4, 2.13934e4, 2.61278e4,
                                 3.04568e4, 3.43222e4, 3.76715e4, 4.04592e4, 4.26475e4,
                                 4.42073e4, 4.51187e4, 4.53713e4, 4.49642e4, 4.39061e4,
                                 4.22149e4, 3.99175e4, 3.70492e4, 3.36529e4, 2.97782e4,
                                 2.54801e4, 2.08178e4, 1.58531e4, 1.06489e4, 5.27012e3])
        self.assertTrue(np.allclose(qp, reference_qp, rtol=5e-5))


if __name__ == '__main__':
    unittest.main()
