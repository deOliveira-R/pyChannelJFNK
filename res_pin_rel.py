import matpro as mat
import correlations as co
from math import tau
import numpy as np
from scipy.optimize import fsolve
import unittest
# import matplotlib.pyplot as plt


class pinSolver:
    """
    Pin heat transfer solver

    Solves a heat conduction problem in 1D
    """

    def __init__(self, geometry, discretization, boundary_conditions):
        # this must be eliminated from ht solver.
        # T and v should be given in the right format so that ht is oblivious to th.
        self.T_in = boundary_conditions['T_in']
        self.v_in = boundary_conditions['v_in']

        self.geometry = geometry
        self.h = geometry['H']
        # self.Rco = geometry['Rco']
        self.Rci = geometry['Rci']
        self.Rfo = geometry['Rfo']

        self.axial_nodes = discretization['axial_nodes']
        self.radial_nodes = discretization['radial_nodes_pin']
        self.fuel_size = self.axial_nodes * self.radial_nodes
        self.domain_size = self.axial_nodes * (self.radial_nodes + 2)

        self.Dz = discretization['Dz']
        self.DR = discretization['DR']
        self.SR = discretization['SR']
        self.VR = discretization['VR']

        self.T_p, self.T_m = None, None
        self.v_p, self.v_m = None, None

        VRrel_tmp = self.VR[:self.radial_nodes] / np.sum(self.VR[:self.radial_nodes])

        DRm_tmp = self.DR[::2]
        DRp_tmp = self.DR[1::2]
        SRm_tmp = np.concatenate([[0], self.SR[1:-1:2]])
        SRp_tmp = self.SR[1::2]
        VRm_tmp = self.VR[::2]
        VRp_tmp = self.VR[1::2]
        VR_tmp = VRp_tmp + VRm_tmp

        self.DRm = np.repeat(DRm_tmp, self.axial_nodes)
        self.DRp = np.repeat(DRp_tmp, self.axial_nodes)
        self.SRm = np.repeat(SRm_tmp, self.axial_nodes)
        self.SRp = np.repeat(SRp_tmp, self.axial_nodes)

        self.VR = np.repeat(VR_tmp, self.axial_nodes)
        self.VRrel = np.repeat(VRrel_tmp, self.axial_nodes)

    def neg_pos_surf(self, T_rel, v_rel):
        T = np.concatenate([[self.T_in], T_rel * self.T_in])
        v = np.concatenate([[self.v_in], v_rel * self.v_in])

        self.T_m = T[:-1]
        self.T_p = T[1:]
        self.v_m = v[:-1]
        self.v_p = v[1:]

    def interpolate_node(self, p, m):
        return (p + m) / 2

    def split_solution(self, sol):
        [T_fuel_rel, T_gap_rel, T_clad_rel] = np.split(sol, [self.fuel_size, self.fuel_size + self.axial_nodes])
        return T_fuel_rel * self.T_in, T_gap_rel * self.T_in, T_clad_rel * self.T_in

    def calculate_k(self, T_fuel, T_clad, qp):
        return np.concatenate([mat.k_fuel(T_fuel),
                               co.h_gap(qp) * (self.Rci - self.Rfo),
                               mat.k_SS(T_clad)])

    def expand_qp(self, qp):
        return np.tile(qp, self.radial_nodes)

    def calculate_coef(self, k, h, qp, T):
        a = np.concatenate([- self.SRp[:self.axial_nodes] / self.VR[:self.axial_nodes] * k[:self.axial_nodes] * k[self.axial_nodes:2 * self.axial_nodes] / (k[:self.axial_nodes] * self.DRm[self.axial_nodes:2 * self.axial_nodes] + k[self.axial_nodes:2 * self.axial_nodes] * self.DRp[:self.axial_nodes] ),
             - self.SRp[self.axial_nodes:self.fuel_size + self.axial_nodes] / self.VR[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / (k[self.axial_nodes:self.fuel_size + self.axial_nodes] * self.DRm[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] + k[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * self.DRp[self.axial_nodes:self.fuel_size + self.axial_nodes] ) \
             - self.SRm[self.axial_nodes:self.fuel_size + self.axial_nodes] / self.VR[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[:self.fuel_size] / (k[:self.fuel_size] * self.DRm[self.axial_nodes:self.fuel_size + self.axial_nodes] + k[self.axial_nodes:self.fuel_size + self.axial_nodes] * self.DRp[:self.fuel_size] ),
             + self.SRp[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / self.VR[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * h / (k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] - h * self.DRp[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] ) \
             - self.SRm[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / self.VR[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size:self.fuel_size + self.axial_nodes] / (k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * self.DRp[self.fuel_size:self.fuel_size + self.axial_nodes] + k[self.fuel_size:self.fuel_size + self.axial_nodes] * self.DRm[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] )])

        b = np.concatenate([self.SRp[:self.axial_nodes] / self.VR[:self.axial_nodes] * k[:self.axial_nodes] * k[self.axial_nodes:2 * self.axial_nodes] / (k[:self.axial_nodes] * self.DRm[self.axial_nodes:2 * self.axial_nodes] + k[self.axial_nodes:2 * self.axial_nodes] * self.DRp[:self.axial_nodes] ),
            self.SRp[self.axial_nodes:self.fuel_size + self.axial_nodes] / self.VR[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / (k[self.axial_nodes:self.fuel_size+self.axial_nodes] * self.DRm[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] + k[2 * self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * self.DRp[self.axial_nodes:self.fuel_size + self.axial_nodes] ),
            np.zeros(self.axial_nodes)])

        c = np.concatenate([np.zeros(self.axial_nodes),
            self.SRm[self.axial_nodes:self.fuel_size + self.axial_nodes] / self.VR[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[self.axial_nodes:self.fuel_size + self.axial_nodes] * k[:self.fuel_size] / (k[:self.fuel_size] * self.DRm[self.axial_nodes:self.fuel_size + self.axial_nodes] + k[self.axial_nodes:self.fuel_size + self.axial_nodes] * self.DRp[:self.fuel_size] ),
            self.SRm[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / self.VR[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size:self.fuel_size + self.axial_nodes] / (k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * self.DRp[self.fuel_size:self.fuel_size + self.axial_nodes] + k[self.fuel_size:self.fuel_size + self.axial_nodes] * self.DRm[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] )])

        q = np.concatenate([self.VRrel[:self.fuel_size] * qp[:self.fuel_size] / self.VR[:self.fuel_size],
            np.zeros(self.axial_nodes),
            - self.SRp[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] / self.VR[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * h * T / (k[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] - h * self.DRp[self.fuel_size + self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] )])

        return a, b, c, q

    def res(self, sol, qp, v_rel, T_rel):
        """
        PIN POWER RESIDUAL VECTOR

        :param sol: a vector representing: the volume-averaged temperature in the
                fuel pin (i.e.pellet, gap, and cladding). The numbering of the
                elements is from the inner fuel pin and all axial nodes (from
                the inlet to the outlet) to the outer fuel pin and all axial
                nodes (from the inlet to the outlet)
        :param qp: a vector representing the node-wise linear power density (from
                   the inlet to the outlet) in W.m^-1
        :param v_rel: a vector representing the surface-averaged velocity from
                      the inlet + 1 node to the outlet in relative terms to the inlet
                      velocity v_in
        :param T_rel: a vector representing the surface-averaged temperature from
                      the inlet + 1 node to the outlet in relative terms to the inlet
                      temperature T_in
        :return: a vector representing the node-wise normalized residuals of the
                 heat conduction equation from the inner fuel pin and all axial nodes
                 to the outer fuel pin and all axial nodes (from the inlet to the outlet)
        """
        self.neg_pos_surf(T_rel, v_rel)
        T = self.interpolate_node(self.T_p, self.T_m)
        v = self.interpolate_node(self.v_p, self.v_m)
        h = co.h_Na(self.geometry, v, T)

        T_fuel, T_gap, T_clad = self.split_solution(sol)
        k = self.calculate_k(T_fuel, T_clad, qp)

        expanded_qp = self.expand_qp(qp)
        a, b, c, q = self.calculate_coef(k, h, expanded_qp, T)

        res = (a * sol * self.T_in
               + b * np.concatenate([sol[self.axial_nodes:self.fuel_size + 2 * self.axial_nodes] * self.T_in, np.zeros(self.axial_nodes)])
               + c * np.concatenate([np.zeros(self.axial_nodes), sol[:self.fuel_size + self.axial_nodes] * self.T_in])
               + q ) \
                / (np.sum(expanded_qp) * self.Dz / self.h / ((tau/2) * self.Rfo**2 / self.radial_nodes))

        return res

    def calculate_T_fuel_rel(self, sol):
        T_fuel_rel = np.sum(
            np.reshape(sol[:self.fuel_size], (self.radial_nodes, self.axial_nodes)),
            axis=0) / self.radial_nodes
        return T_fuel_rel


class pinTest(unittest.TestCase):

    def setUp(self):
        self.geometry = {'H': 1.6,
                         'Rfo': 3.57E-3,
                         'Rci': 3.685E-3,
                         'Rco': 0.00425,
                         'De':0.0039587,
                         'pin_pitch': 9.8E-3}
        self.discretization = {'axial_nodes': 25,
                               'radial_nodes_pin': 5,
                               'Dz': 0.064,
                               'DR': np.array([0.00000000e+00, 4.67619411e-04, 3.58816994e-04, 3.02496719e-04, 2.66504959e-04,
                                               2.40938900e-04, 2.21566186e-04, 2.06228777e-04, 1.93694302e-04, 1.83200626e-04,
                                               5.79556916e-05, 5.70443084e-05, 2.92544783e-04, 2.72455217e-04]),
                               'SR': np.array([0.0070933, 0.01003144, 0.01228595, 0.01418659, 0.01586109,
                                               0.01737496, 0.0187671, 0.02006287, 0.02127989, 0.02243097,
                                               0.02279512, 0.02315354, 0.02499165, 0.02670354]),
                               'VR': np.array([4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06,
                                               4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06,
                                               1.31055465e-06, 1.31055465e-06, 7.04231190e-06, 7.04231190e-06])}
        self.boundary_conditions = {'T_in': 673,
                                    'v_in': 7.5,
                                    'qp_ave': 3e4}

        self.solver = pinSolver(self.geometry, self.discretization, self.boundary_conditions)
        self.fuel_size = self.solver.fuel_size
        self.domain_size = self.solver.domain_size

        self.T_rel = (self.boundary_conditions['T_in'] + np.arange(self.discretization['axial_nodes']) * 600
                 / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['T_in']
        self.v_rel = (self.boundary_conditions['v_in'] + np.arange(self.discretization['axial_nodes']) * 3
                 / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['v_in']
        self.T_v_rel_uniform = np.ones(self.discretization['axial_nodes'])
        self.qp = self.boundary_conditions['qp_ave'] * np.sin((np.arange(self.discretization['axial_nodes']) + 0.5) / self.discretization['axial_nodes'] * (tau/2))
        self.qp_uniform = np.full(self.discretization['axial_nodes'], self.boundary_conditions['qp_ave'])
        self.sol = np.ones(self.discretization['axial_nodes'] * (self.discretization['radial_nodes_pin'] + 2))

    def test_split(self):
        T_fuel, T_gap, T_clad = self.solver.split_solution(self.sol)
        self.assertEqual(len(T_fuel), self.fuel_size)
        self.assertEqual(len(T_gap), self.discretization['axial_nodes'])
        self.assertEqual(len(T_clad), self.discretization['axial_nodes'])

    def test_m_p(self):
        self.solver.neg_pos_surf(self.T_rel, self.v_rel)
        self.assertEqual(len(self.solver.T_m), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.T_p), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.v_m), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.v_p), self.discretization['axial_nodes'])
        reference_T_m = np.array([673.0, 673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0,
                                  898.0, 923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0,
                                  1148.0, 1173.0, 1198.0, 1223.0, 1248.0])
        self.assertTrue(np.allclose(self.solver.T_m, reference_T_m, rtol=1e-5))
        reference_T_p = np.array([673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0, 898.0,
                                  923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0, 1148.0,
                                  1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        self.assertTrue(np.allclose(self.solver.T_p, reference_T_p, rtol=1e-5))
        reference_v_m = np.array([7.500, 7.500, 7.625, 7.750, 7.875, 8, 8.125, 8.250, 8.375, 8.500,
                                  8.625, 8.750, 8.875, 9, 9.125, 9.250, 9.375, 9.500, 9.625, 9.750,
                                  9.875, 10, 10.125, 10.250, 10.375])
        self.assertTrue(np.allclose(self.solver.v_m, reference_v_m, rtol=1e-5))
        reference_v_p = np.array([7.500, 7.625, 7.750, 7.875, 8, 8.125, 8.250, 8.375, 8.500, 8.625,
                                  8.750, 8.875, 9, 9.125, 9.250, 9.375, 9.500, 9.625, 9.750, 9.875,
                                  10, 10.125, 10.250, 10.375, 10.500])
        self.assertTrue(np.allclose(self.solver.v_p, reference_v_p, rtol=1e-5))

    def test_interpolate(self):
        self.solver.neg_pos_surf(self.T_rel, self.v_rel)
        T = self.solver.interpolate_node(self.solver.T_p, self.solver.T_m)
        self.assertEqual(len(T), self.discretization['axial_nodes'])
        reference_T = np.array([673.0, 685.5, 710.5, 735.5, 760.5, 785.5, 810.5, 835.5, 860.5, 885.5,
                                910.5, 935.5, 960.5, 985.5, 1010.5, 1035.5, 1060.5, 1085.5, 1110.5, 1135.5,
                                1160.5, 1185.5, 1210.5, 1235.5, 1260.5])
        self.assertTrue(np.allclose(T, reference_T, rtol=1e-5))
        v = self.solver.interpolate_node(self.solver.v_p, self.solver.v_m)
        self.assertEqual(len(v), self.discretization['axial_nodes'])
        reference_v = np.array([7.5000, 7.5625, 7.6875, 7.8125, 7.9375, 8.0625, 8.1875, 8.3125, 8.4375, 8.5625,
                                8.6875, 8.8125, 8.9375, 9.0625, 9.1875, 9.3125, 9.4375, 9.5625, 9.6875, 9.8125,
                                9.9375, 10.0625, 10.1875, 10.3125, 10.4375])
        self.assertTrue(np.allclose(v, reference_v, rtol=1e-5))

    def test_DR_SR(self):
        self.assertEqual(len(self.solver.DRm), self.domain_size)
        self.assertEqual(len(self.solver.DRp), self.domain_size)
        self.assertEqual(len(self.solver.SRm), self.domain_size)
        self.assertEqual(len(self.solver.SRp), self.domain_size)
        reference_DRm = np.concatenate([[0] * self.discretization['axial_nodes'],
                                        [3.58817e-4] * self.discretization['axial_nodes'],
                                        [2.66505e-4] * self.discretization['axial_nodes'],
                                        [2.21566e-4] * self.discretization['axial_nodes'],
                                        [1.93694e-4] * self.discretization['axial_nodes'],
                                        [5.79557e-5] * self.discretization['axial_nodes'],
                                        [2.92545e-4] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(self.solver.DRm, reference_DRm, rtol=1e-5))
        reference_DRp = np.concatenate([[4.67619e-4] * self.discretization['axial_nodes'],
                                        [3.02497e-4] * self.discretization['axial_nodes'],
                                        [2.40939e-4] * self.discretization['axial_nodes'],
                                        [2.06229e-4] * self.discretization['axial_nodes'],
                                        [1.83201e-4] * self.discretization['axial_nodes'],
                                        [5.70443e-5] * self.discretization['axial_nodes'],
                                        [2.72455e-4] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(self.solver.DRp, reference_DRp, rtol=1e-5))
        reference_SRm = np.concatenate([[0] * self.discretization['axial_nodes'],
                                        [1.00314e-2] * self.discretization['axial_nodes'],
                                        [1.41866e-2] * self.discretization['axial_nodes'],
                                        [1.73750e-2] * self.discretization['axial_nodes'],
                                        [2.00629e-2] * self.discretization['axial_nodes'],
                                        [2.24310e-2] * self.discretization['axial_nodes'],
                                        [2.31535e-2] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(self.solver.SRm, reference_SRm, rtol=1e-5))
        reference_SRp = np.concatenate([[1.00314e-2] * self.discretization['axial_nodes'],
                                        [1.41866e-2] * self.discretization['axial_nodes'],
                                        [1.73750e-2] * self.discretization['axial_nodes'],
                                        [2.00629e-2] * self.discretization['axial_nodes'],
                                        [2.24310e-2] * self.discretization['axial_nodes'],
                                        [2.31535e-2] * self.discretization['axial_nodes'],
                                        [2.67035e-2] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(self.solver.SRp, reference_SRp, rtol=1e-5))

    def test_VR(self):
        self.assertEqual(len(self.solver.VR), self.domain_size)
        reference_VR = np.concatenate([[8.00786e-6] * self.fuel_size,
                                       [2.62111e-6] * self.discretization['axial_nodes'],
                                       [1.40846e-5] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(self.solver.VR, reference_VR, rtol=1e-5))
        self.assertEqual(len(self.solver.VRrel), self.fuel_size)
        reference_VRrel = np.full(self.fuel_size, 0.2)
        self.assertTrue(np.allclose(self.solver.VRrel, reference_VRrel, rtol=1e-5))

    def test_qp(self):
        expanded_qp = self.solver.expand_qp(self.qp)
        self.assertEqual(len(expanded_qp), self.fuel_size)
        reference_qp = np.array([1.88372e3, 5.62144e3, 9.27051e3, 1.27734e4, 1.60748e4,
                                 1.91227e4, 2.18691e4, 2.42705e4, 2.62892e4, 2.78933e4,
                                 2.90575e4, 2.97634e4, 3.00000e4, 2.97634e4, 2.90575e4,
                                 2.78933e4, 2.62892e4, 2.42705e4, 2.18691e4, 1.91227e4,
                                 1.60748e4, 1.27734e4, 9.27051e3, 5.62144e3, 1.88372e3]\
                                 * self.discretization['radial_nodes_pin'])
        self.assertTrue(np.allclose(expanded_qp, reference_qp, rtol=1e-5))

    def test_k(self):
        T_fuel, T_gap, T_clad = self.solver.split_solution(self.sol)
        k = self.solver.calculate_k(T_fuel, T_clad, self.qp)
        self.assertEqual(len(k), self.domain_size)
        reference_k = np.concatenate([[2.63666] * self.fuel_size,
                                      [3.39728e-1, 3.36570e-1, 3.42942e-1, 3.57941e-1, 3.80123e-1,
                                       4.07598e-1, 4.38158e-1, 4.69424e-1, 4.99008e-1, 5.24678e-1,
                                       5.44513e-1, 5.57035e-1, 5.61315e-1, 5.57035e-1, 5.44513e-1,
                                       5.24678e-1, 4.99008e-1, 4.69424e-1, 4.38158e-1, 4.07598e-1,
                                       3.80123e-1, 3.57941e-1, 3.42942e-1, 3.36570e-1, 3.39728e-1],
                                      [19.4602] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(k, reference_k, rtol=1e-5))

    def test_calculate_coef(self):
        self.solver.neg_pos_surf(self.T_rel, self.v_rel)
        T = self.solver.interpolate_node(self.solver.T_p, self.solver.T_m)
        v = self.solver.interpolate_node(self.solver.v_p, self.solver.v_m)
        h = co.h_Na(self.geometry, v, T)

        T_fuel, T_gap, T_clad = self.solver.split_solution(self.sol)
        k = self.solver.calculate_k(T_fuel, T_clad, self.qp)

        a, b, c, q = self.solver.calculate_coef(k, h, self.solver.expand_qp(self.qp), T)
        reference_a = np.concatenate([[-3.99661e6] * self.discretization['axial_nodes'],
                                      [-1.22058e7] * self.discretization['axial_nodes'],
                                      [-2.05785e7] * self.discretization['axial_nodes'],
                                      [-2.88872e7] * self.discretization['axial_nodes'],
                                      [-2.81855e7, -2.81082e7, -2.82637e7, -2.86232e7, -2.91385e7,
                                       -2.97513e7, -3.04018e7, -3.10357e7, -3.16082e7, -3.20847e7,
                                       -3.24408e7, -3.26603e7, -3.27345e7, -3.26603e7, -3.24408e7,
                                       -3.20847e7, -3.16082e7, -3.10357e7, -3.04018e7, -2.97513e7,
                                       -2.91385e7, -2.86232e7, -2.82637e7, -2.81082e7, -2.81855e7,
                                       -8.39312e7, -8.32828e7, -8.45891e7, -8.76338e7, -9.20613e7,
                                       -9.74256e7, -1.03244e8, -1.09043e8, -1.14395e8, -1.18937e8,
                                       -1.22385e8, -1.24534e8, -1.25264e8, -1.24534e8, -1.22385e8,
                                       -1.18937e8, -1.14395e8, -1.09043e8, -1.03244e8, -9.74256e7,
                                       -9.20613e7, -8.76338e7, -8.45891e7, -8.32828e7, -8.39312e7,
                                       -1.85367e8, -1.85590e8, -1.86358e8, -1.87351e8, -1.88532e8,
                                       -1.89853e8, -1.91257e8, -1.92688e8, -1.94094e8, -1.95427e8,
                                       -1.96649e8, -1.97733e8, -1.98662e8, -1.99432e8, -2.00050e8,
                                       -2.00535e8, -2.00919e8, -2.01240e8, -2.01551e8, -2.01906e8,
                                       -2.02365e8, -2.02988e8, -2.03827e8, -2.04924e8, -2.06306e8]])
        self.assertTrue(np.allclose(a, reference_a, rtol=1e-5))
        reference_b = np.concatenate([[3.99661e6] * self.discretization['axial_nodes'],
                                      [8.20923e6] * self.discretization['axial_nodes'],
                                      [1.23693e7] * self.discretization['axial_nodes'],
                                      [1.65179e7] * self.discretization['axial_nodes'],
                                      [1.16676e7, 1.15903e7, 1.17458e7, 1.21053e7, 1.26206e7,
                                       1.32334e7, 1.38839e7, 1.45178e7, 1.50903e7, 1.55668e7,
                                       1.59229e7, 1.61424e7, 1.62166e7, 1.61424e7, 1.59229e7,
                                       1.55668e7, 1.50903e7, 1.45178e7, 1.38839e7, 1.32334e7,
                                       1.26206e7, 1.21053e7, 1.17458e7, 1.15903e7, 1.16676e7,
                                       4.82850e7, 4.78727e7, 4.87039e7, 5.06504e7, 5.35035e7,
                                       5.69956e7, 6.08266e7, 6.46892e7, 6.82922e7, 7.13785e7,
                                       7.37383e7, 7.52170e7, 7.57205e7, 7.52170e7, 7.37383e7,
                                       7.13785e7, 6.82922e7, 6.46892e7, 6.08266e7, 5.69956e7,
                                       5.35035e7, 5.06504e7, 4.87039e7, 4.78727e7, 4.82850e7],
                                      np.zeros(self.discretization['axial_nodes'])])
        self.assertTrue(np.allclose(b, reference_b, rtol=1e-5))
        reference_c = np.concatenate([np.zeros(self.discretization['axial_nodes']),
                                     [3.99661e6] * self.discretization['axial_nodes'],
                                     [8.20923e6] * self.discretization['axial_nodes'],
                                     [1.23693e7] * self.discretization['axial_nodes'],
                                     [1.65179e7] * self.discretization['axial_nodes'],
                                     [3.56462e7, 3.54101e7, 3.58852e7, 3.69834e7, 3.85578e7,
                                      4.04299e7, 4.24172e7, 4.43540e7, 4.61030e7, 4.75588e7,
                                      4.86466e7, 4.93174e7, 4.95439e7, 4.93174e7, 4.86466e7,
                                      4.75588e7, 4.61030e7, 4.43540e7, 4.24172e7, 4.04299e7,
                                      3.85578e7, 3.69834e7, 3.58852e7, 3.54101e7, 3.56462e7,
                                      8.98570e6, 8.90897e6, 9.06366e6, 9.42591e6, 9.95685e6,
                                      1.06067e7, 1.13197e7, 1.20385e7, 1.27090e7, 1.32833e7,
                                      1.37225e7, 1.39977e7, 1.40914e7, 1.39977e7, 1.37225e7,
                                      1.32833e7, 1.27090e7, 1.20385e7, 1.13197e7, 1.06067e7,
                                      9.95685e6, 9.42591e6, 9.06366e6, 8.90897e6, 8.98570e6]])
        self.assertTrue(np.allclose(c, reference_c, rtol=1e-5))
        reference_q = np.concatenate([[4.70467e7, 1.40398e8, 2.31535e8, 3.19021e8, 4.01476e8,
                                       4.77599e8, 5.46190e8, 6.06167e8, 6.56585e8, 6.96648e8,
                                       7.25725e8, 7.43356e8, 7.49264e8, 7.43356e8, 7.25725e8,
                                       6.96648e8, 6.56585e8, 6.06167e8, 5.46190e8, 4.77599e8,
                                       4.01476e8, 3.19021e8, 2.31535e8, 1.40398e8, 4.70467e7,
                                       4.70467e7, 1.40398e8, 2.31535e8, 3.19021e8, 4.01476e8,
                                       4.77599e8, 5.46190e8, 6.06167e8, 6.56585e8, 6.96648e8,
                                       7.25725e8, 7.43356e8, 7.49264e8, 7.43356e8, 7.25725e8,
                                       6.96648e8, 6.56585e8, 6.06167e8, 5.46190e8, 4.77599e8,
                                       4.01476e8, 3.19021e8, 2.31535e8, 1.40398e8, 4.70467e7,
                                       4.70467e7, 1.40398e8, 2.31535e8, 3.19021e8, 4.01476e8,
                                       4.77599e8, 5.46190e8, 6.06167e8, 6.56585e8, 6.96648e8,
                                       7.25725e8, 7.43356e8, 7.49264e8, 7.43356e8, 7.25725e8,
                                       6.96648e8, 6.56585e8, 6.06167e8, 5.46190e8, 4.77599e8,
                                       4.01476e8, 3.19021e8, 2.31535e8, 1.40398e8, 4.70467e7,
                                       4.70467e7, 1.40398e8, 2.31535e8, 3.19021e8, 4.01476e8,
                                       4.77599e8, 5.46190e8, 6.06167e8, 6.56585e8, 6.96648e8,
                                       7.25725e8, 7.43356e8, 7.49264e8, 7.43356e8, 7.25725e8,
                                       6.96648e8, 6.56585e8, 6.06167e8, 5.46190e8, 4.77599e8,
                                       4.01476e8, 3.19021e8, 2.31535e8, 1.40398e8, 4.70467e7,
                                       4.70467e7, 1.40398e8, 2.31535e8, 3.19021e8, 4.01476e8,
                                       4.77599e8, 5.46190e8, 6.06167e8, 6.56585e8, 6.96648e8,
                                       7.25725e8, 7.43356e8, 7.49264e8, 7.43356e8, 7.25725e8,
                                       6.96648e8, 6.56585e8, 6.06167e8, 5.46190e8, 4.77599e8,
                                       4.01476e8, 3.19021e8, 2.31535e8, 1.40398e8, 4.70467e7],
                                      np.zeros(self.discretization['axial_nodes']),
                                      [1.18705e11, 1.21115e11, 1.25967e11, 1.30864e11, 1.35807e11,
                                       1.40798e11, 1.45839e11, 1.50933e11, 1.56081e11, 1.61288e11,
                                       1.66554e11, 1.71884e11, 1.77280e11, 1.82745e11, 1.88284e11,
                                       1.93900e11, 1.99596e11, 2.05379e11, 2.11251e11, 2.17220e11,
                                       2.23290e11, 2.29468e11, 2.35761e11, 2.42176e11, 2.48722e11]])
        self.assertTrue(np.allclose(q, reference_q, rtol=1e-5))

    def test_res(self):
        res = self.solver.res(self.sol, self.qp, self.v_rel, self.T_rel)
        reference_res = np.array([3.94265e-3, 1.17658e-2, 1.94033e-2, 2.67349e-2, 3.36448e-2,
                                  4.00242e-2, 4.57723e-2, 5.07986e-2, 5.50238e-2, 5.83811e-2,
                                  6.08178e-2, 6.22954e-2, 6.27905e-2, 6.22954e-2, 6.08178e-2,
                                  5.83811e-2, 5.50238e-2, 5.07986e-2, 4.57723e-2, 4.00242e-2,
                                  3.36448e-2, 2.67349e-2, 1.94033e-2, 1.17658e-2, 3.94265e-3,
                                  3.94265e-3, 1.17658e-2, 1.94033e-2, 2.67349e-2, 3.36448e-2,
                                  4.00242e-2, 4.57723e-2, 5.07986e-2, 5.50238e-2, 5.83811e-2,
                                  6.08178e-2, 6.22954e-2, 6.27905e-2, 6.22954e-2, 6.08178e-2,
                                  5.83811e-2, 5.50238e-2, 5.07986e-2, 4.57723e-2, 4.00242e-2,
                                  3.36448e-2, 2.67349e-2, 1.94033e-2, 1.17658e-2, 3.94265e-3,
                                  3.94265e-3, 1.17658e-2, 1.94033e-2, 2.67349e-2, 3.36448e-2,
                                  4.00242e-2, 4.57723e-2, 5.07986e-2, 5.50238e-2, 5.83811e-2,
                                  6.08178e-2, 6.22954e-2, 6.27905e-2, 6.22954e-2, 6.08178e-2,
                                  5.83811e-2, 5.50238e-2, 5.07986e-2, 4.57723e-2, 4.00242e-2,
                                  3.36448e-2, 2.67349e-2, 1.94033e-2, 1.17658e-2, 3.94265e-3,
                                  3.94265e-3, 1.17658e-2, 1.94033e-2, 2.67349e-2, 3.36448e-2,
                                  4.00242e-2, 4.57723e-2, 5.07986e-2, 5.50238e-2, 5.83811e-2,
                                  6.08178e-2, 6.22954e-2, 6.27905e-2, 6.22954e-2, 6.08178e-2,
                                  5.83811e-2, 5.50238e-2, 5.07986e-2, 4.57723e-2, 4.00242e-2,
                                  3.36448e-2, 2.67349e-2, 1.94033e-2, 1.17658e-2, 3.94265e-3,
                                  3.94265e-3, 1.17658e-2, 1.94033e-2, 2.67349e-2, 3.36448e-2,
                                  4.00242e-2, 4.57723e-2, 5.07986e-2, 5.50238e-2, 5.83811e-2,
                                  6.08178e-2, 6.22954e-2, 6.27905e-2, 6.22954e-2, 6.08178e-2,
                                  5.83811e-2, 5.50238e-2, 5.07986e-2, 4.57723e-2, 4.00242e-2,
                                  3.36448e-2, 2.67349e-2, 1.94033e-2, 1.17658e-2, 3.94265e-3,
                                  3.19683e-16, 0, -6.39366e-16, 3.19683e-16, 6.39366e-16,
                                  3.19683e-16, 0, -3.19683e-16, -1.27873e-15, 0,
                                  3.19683e-16, 6.39366e-16, 0, 3.19683e-16, 3.19683e-16,
                                  0, -1.27873e-15, -3.19683e-16, 3.19683e-16, 3.19683e-16,
                                  3.19683e-16, 3.19683e-16, -6.39366e-16, 0, 0,
                                  -2.55746e-15, 1.85080e-1, 5.57165e-1, 9.31915e-1, 1.30945,
                                  1.68990, 2.07339, 2.46008, 2.85011, 3.24363,
                                  3.64082, 4.04185, 4.44692, 4.85623, 5.26999,
                                  5.68845, 6.11185, 6.54046, 6.97458, 7.41453,
                                  7.86065, 8.31331, 8.77293, 9.23996, 9.71489])
        self.assertTrue(np.allclose(res, reference_res, rtol=1e-5))

    def test_solve_uni(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol,
                                     args=(self.qp_uniform, self.T_v_rel_uniform, self.T_v_rel_uniform),
                                     full_output=True, xtol=1e-6)
        reference_sol = np.concatenate([[2.37153] * self.discretization['axial_nodes'],
                                        [2.17060] * self.discretization['axial_nodes'],
                                        [1.97024] * self.discretization['axial_nodes'],
                                        [1.76653] * self.discretization['axial_nodes'],
                                        [1.55683] * self.discretization['axial_nodes'],
                                        [1.24207] * self.discretization['axial_nodes'],
                                        [1.01773] * self.discretization['axial_nodes']])
        self.assertTrue(np.allclose(sol, reference_sol, rtol=1e-5))

    def test_solve_cos(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol,
                                     args=(self.qp, self.T_v_rel_uniform, self.T_v_rel_uniform),
                                     full_output=True, xtol=1e-6)
        reference_sol = np.array([1.11989, 1.34459, 1.54084, 1.71288, 1.86267,
                                  1.99035, 2.09686, 2.18383, 2.25290, 2.30541,
                                  2.34233, 2.36426, 2.37153, 2.36426, 2.34233,
                                  2.30541, 2.25290, 2.18383, 2.09686, 1.99035,
                                  1.86267, 1.71288, 1.54084, 1.34459, 1.11989,
                                  1.10295, 1.29876, 1.47149, 1.62105, 1.74938,
                                  1.85729, 1.94617, 2.01793, 2.07444, 2.11713,
                                  2.14702, 2.16473, 2.17060, 2.16473, 2.14702,
                                  2.11713, 2.07444, 2.01793, 1.94617, 1.85729,
                                  1.74938, 1.62105, 1.47149, 1.29876, 1.10295,
                                  1.08635, 1.25299, 1.40202, 1.52982, 1.63728,
                                  1.72576, 1.79712, 1.85362, 1.89735, 1.92994,
                                  1.95254, 1.96585, 1.97024, 1.96585, 1.95254,
                                  1.92994, 1.89735, 1.85362, 1.79712, 1.72576,
                                  1.63728, 1.52982, 1.40202, 1.25299, 1.08635,
                                  1.06974, 1.20631, 1.33041, 1.43625, 1.52303,
                                  1.59212, 1.64586, 1.68689, 1.71759, 1.73983,
                                  1.75491, 1.76366, 1.76653, 1.76366, 1.75491,
                                  1.73983, 1.71759, 1.68689, 1.64586, 1.59212,
                                  1.52303, 1.43625, 1.33042, 1.20631, 1.06974,
                                  1.05308, 1.15869, 1.25605, 1.33874, 1.40442,
                                  1.45397, 1.48989, 1.51517, 1.53255, 1.54415,
                                  1.55149, 1.55554, 1.55683, 1.55554, 1.55149,
                                  1.54415, 1.53255, 1.51517, 1.48989, 1.45397,
                                  1.40442, 1.33874, 1.25605, 1.15869, 1.05308,
                                  1.02324, 1.06991, 1.11340, 1.15052, 1.17980,
                                  1.20144, 1.21664, 1.22690, 1.23359, 1.23781,
                                  1.24033, 1.24166, 1.24207, 1.24166, 1.24033,
                                  1.23781, 1.23359, 1.22690, 1.21664, 1.20144,
                                  1.17980, 1.15052, 1.11340, 1.06991, 1.02324,
                                  1.00113, 1.00335, 1.00552, 1.00760, 1.00955,
                                  1.01135, 1.01297, 1.01438, 1.01556, 1.01650,
                                  1.01718, 1.01760, 1.01773, 1.01760, 1.01718,
                                  1.01650, 1.01556, 1.01438, 1.01297, 1.01135,
                                  1.00955, 1.00760, 1.00552, 1.00335, 1.00113])
        self.assertTrue(np.allclose(sol, reference_sol, rtol=5e-5))

    def test_calculate_T_fuel_rel(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol,
                                     args=(self.qp, self.T_v_rel_uniform, self.T_v_rel_uniform),
                                     full_output=True, xtol=1e-6)
        T_fuel_rel = self.solver.calculate_T_fuel_rel(sol)
        reference_T_fuel_rel = np.array([1.08640, 1.25227, 1.40016, 1.52775, 1.63536,
                                         1.72390, 1.79518, 1.85149, 1.89497, 1.92730,
                                         1.94966, 1.96281, 1.96715, 1.96281, 1.94966,
                                         1.92730, 1.89497, 1.85149, 1.79518, 1.72390,
                                         1.63536, 1.52775, 1.40016, 1.25227, 1.08640])
        self.assertTrue(np.allclose(T_fuel_rel, reference_T_fuel_rel, rtol=5e-5))


if __name__ == '__main__':
    unittest.main()
