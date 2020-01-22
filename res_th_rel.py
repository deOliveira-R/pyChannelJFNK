import correlations as co
import thermo_Na as Na
import numpy as np
from scipy.optimize import fsolve
import unittest
import matplotlib.pyplot as plt


class thSolver:
    """
    Thermal hydraulics solver

    Solves a thermal hydraulics problem in 1D
    """

    def __init__(self, geometry, discretization, physics_parameters, boundary_conditions):
        self.T_in = boundary_conditions['T_in']
        self.v_in = boundary_conditions['v_in']

        self.geometry = geometry
        self.De = geometry['De']
        self.A = geometry['A']

        self.axial_nodes = discretization['axial_nodes']
        self.domain_size = self.axial_nodes * 2

        self.DZ = discretization['Dz']

        self.g = physics_parameters['g']

        self.T_p, self.T_m = None, None
        self.v_p, self.v_m = None, None

    def split_solution(self, sol):
        [T_rel, v_rel] = np.split(sol, 2)
        return T_rel, v_rel

    def surface_values(self, T_rel, v_rel):
        T = np.concatenate([[self.T_in], T_rel * self.T_in])
        v = np.concatenate([[self.v_in], v_rel * self.v_in])
        return T, v

    def neg_pos_surf(self, T_rel, v_rel):
        T, v = self.surface_values(T_rel, v_rel)

        self.T_m = T[:-1]
        self.T_p = T[1:]
        self.v_m = v[:-1]
        self.v_p = v[1:]

    def res_mass(self):
        return (self.v_p * Na.rho(self.T_p) - self.v_m * Na.rho(self.T_m)) / (Na.rho(self.T_in) * self.v_in)

    def deltaP_friction(self):
        return (self.DZ / self.De) * co.fric_factor(self.geometry , self.v_m, self.T_m) * ((Na.rho(self.T_m) * self.v_m ** 2) / 2)

    def deltaP_acceleration(self):
        return (Na.rho(self.T_p) * self.v_p ** 2) - (Na.rho(self.T_m) * self.v_m ** 2)

    def deltaP_gravity(self):
        return Na.rho((self.T_p + self.T_m) / 2) * self.g * self.DZ

    def deltaP(self):
        return - self.deltaP_gravity() - self.deltaP_friction() - self.deltaP_acceleration()

    def res_enthalpy(self, qp):
        return (self.v_p * Na.rho(self.T_p) * Na.h(self.T_p) - self.v_m * Na.rho(self.T_m) * Na.h(self.T_m)
                - 0.5 * (self.v_p + self.v_m) * self.deltaP()
                - qp * self.DZ / self.A) \
                / (Na.rho(self.T_in) * self.v_in * Na.h(self.T_in))

    def res(self, sol, qp):
        """
        THERMAL HYDRAULICS RESIDUAL VECTOR

        :param sol: a vector representing the the surface-wise temperature for the
                    first half of the vector and velocity for the second half of
                    the vector (from the first surface above the inlet to the outlet)
        :param qp: a vector representing the node-wise linear power density (from
                   the inlet to the outlet) in W.m^-1
        :return: the node-wise normalized residuals of the mass conservation equation
                 and enthalpy conservation equation (from the inlet to the outlet)
        """
        T_rel, v_rel = self.split_solution(sol)
        self.neg_pos_surf(T_rel, v_rel)
        return np.concatenate([self.res_mass(), self.res_enthalpy(qp)])


class THTest(unittest.TestCase):

    def setUp(self):
        self.geometry = {'De': 0.0039587,
                         'A': 1.32140e-5}
        self.discretization = {'axial_nodes': 25,
                               'Dz': 0.064}
        self.physics_parameters = {'g': 9.81}
        self.boundary_conditions = {'T_in': 673,
                                    'v_in': 7.5,
                                    'qp_ave': 3e4}
        self.solver = thSolver(self.geometry, self.discretization,
                               self.physics_parameters, self.boundary_conditions)
        self.T_rel = (self.boundary_conditions['T_in'] + np.arange(self.discretization['axial_nodes']) * 600
                 / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['T_in']
        self.v_rel = (self.boundary_conditions['v_in'] + np.arange(self.discretization['axial_nodes']) * 4
                 / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['v_in']
        self.qp = np.full(self.discretization['axial_nodes'], self.boundary_conditions['qp_ave'])
        self.sol = np.concatenate([self.T_rel, self.v_rel])

    def test_split(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.assertEqual(len(T_rel), self.solver.axial_nodes)
        self.assertEqual(len(v_rel), self.solver.axial_nodes)

    def test_tmp(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        T_tmp = T_rel * self.boundary_conditions['T_in']
        v_tmp = v_rel * self.boundary_conditions['v_in']
        self.assertEqual(len(T_tmp), self.discretization['axial_nodes'])
        self.assertEqual(len(v_tmp), self.discretization['axial_nodes'])
        reference_T_tmp = np.array([673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0, 898.0,
                                    923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0, 1148.0,
                                    1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        reference_v_tmp = np.array([7.5000, 7.6667, 7.8333, 8.0000, 8.1667, 8.3333, 8.5000, 8.6667, 8.8333, 9.0000,
                                    9.1667, 9.3333, 9.5000, 9.6667, 9.8333, 10.0000, 10.1667, 10.3333, 10.5000, 10.6667,
                                    10.8333, 11.0000, 11.1667, 11.3333, 11.5000])
        self.assertTrue(np.allclose(T_tmp, reference_T_tmp, rtol=1e-5))
        self.assertTrue(np.allclose(v_tmp, reference_v_tmp, rtol=1e-5))

    def test_T_v(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        T, v = self.solver.surface_values(T_rel, v_rel)
        self.assertEqual(len(T), self.discretization['axial_nodes'] + 1)
        self.assertEqual(len(v), self.discretization['axial_nodes'] + 1)
        reference_T = np.array([673.0, 673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0,
                                898.0, 923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0,
                                1148.0, 1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        reference_v = np.array([7.5000, 7.5000, 7.6667, 7.8333, 8.0000, 8.1667, 8.3333, 8.5000, 8.6667, 8.8333,
                                9.0000, 9.1667, 9.3333, 9.5000, 9.6667, 9.8333, 10.0000, 10.1667, 10.3333, 10.5000,
                                10.6667, 10.8333, 11.0000, 11.1667, 11.3333, 11.5000])
        self.assertTrue(np.allclose(T, reference_T, rtol=1e-5))
        self.assertTrue(np.allclose(v, reference_v, rtol=1e-5))

    def test_m_p(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        self.assertEqual(len(self.solver.T_m), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.T_p), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.v_m), self.discretization['axial_nodes'])
        self.assertEqual(len(self.solver.v_p), self.discretization['axial_nodes'])
        reference_T_m = np.array([673.0, 673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0,
                                  898.0, 923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0,
                                  1148.0, 1173.0, 1198.0, 1223.0, 1248.0])
        reference_T_p = np.array([673.0, 698.0, 723.0, 748.0, 773.0, 798.0, 823.0, 848.0, 873.0, 898.0,
                                  923.0, 948.0, 973.0, 998.0, 1023.0, 1048.0, 1073.0, 1098.0, 1123.0, 1148.0,
                                  1173.0, 1198.0, 1223.0, 1248.0, 1273.0])
        reference_v_m = np.array([7.5000, 7.5000, 7.6667, 7.8333, 8.0000, 8.1667, 8.3333, 8.5000, 8.6667, 8.8333,
                                  9.0000, 9.1667, 9.3333, 9.5000, 9.6667, 9.8333, 10.0000, 10.1667, 10.3333, 10.5000,
                                  10.6667, 10.8333, 11.0000, 11.1667, 11.3333])
        reference_v_p = np.array([7.5000, 7.6667, 7.8333, 8.0000, 8.1667, 8.3333, 8.5000, 8.6667, 8.8333, 9.0000,
                                  9.1667, 9.3333, 9.5000, 9.6667, 9.8333, 10.0000, 10.1667, 10.3333, 10.5000, 10.6667,
                                  10.8333, 11.0000, 11.1667, 11.3333, 11.5000])
        self.assertTrue(np.allclose(self.solver.T_m, reference_T_m, rtol=1e-5))
        self.assertTrue(np.allclose(self.solver.T_p, reference_T_p, rtol=1e-5))
        self.assertTrue(np.allclose(self.solver.v_m, reference_v_m, rtol=1e-5))
        self.assertTrue(np.allclose(self.solver.v_p, reference_v_p, rtol=1e-5))

    def test_res_mass(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        res_mass = self.solver.res_mass()
        self.assertEqual(len(res_mass), self.discretization['axial_nodes'])
        reference_res_mass = np.array([0, 0.0152051, 0.0149000, 0.0145949, 0.0142898,
                                       0.0139848, 0.0136797, 0.0133746, 0.0130695, 0.0127644,
                                       0.0124593, 0.0121542, 0.0118491, 0.0115440, 0.0112389,
                                       0.0109338, 0.0106287, 0.0103237, 0.0100186, 0.0097135,
                                       0.0094084, 0.0091033, 0.0087982, 0.0084931, 0.0081880])
        self.assertTrue(np.allclose(res_mass, reference_res_mass, rtol=1e-5))

    def test_deltaP_friction(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        deltaP_friction = self.solver.deltaP_friction()
        self.assertEqual(len(deltaP_friction), self.discretization['axial_nodes'])
        reference_deltaP_friction = np.array([10735.46, 10735.46, 10977.32, 11222.12, 11469.52,
                                              11719.22, 11970.92, 12224.39, 12479.39, 12735.70,
                                              12993.11, 13251.45, 13510.53, 13770.17, 14030.23,
                                              14290.55, 14551.98, 14811.38, 15071.60, 15331.53,
                                              15591.03, 15849.97, 16108.24, 16365.71, 16622.27])
        self.assertTrue(np.allclose(deltaP_friction, reference_deltaP_friction, rtol=1e-4))

    def test_deltaP_acceleration(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        deltaP_acceleration = self.solver.deltaP_acceleration()
        self.assertEqual(len(deltaP_acceleration), self.discretization['axial_nodes'])
        reference_deltaP_acceleration = np.array([0, 1818.07, 1835.26, 1851.47, 1866.71,
                                                  1880.96, 1894.24, 1906.54, 1917.85, 1928.19,
                                                  1937.55, 1945.93, 1953.33, 1959.75, 1965.20,
                                                  1969.66, 1973.15, 1975.65, 1977.18, 1977.72,
                                                  1977.29, 1975.88, 1973.49, 1970.12, 1965.77])
        self.assertTrue(np.allclose(deltaP_acceleration, reference_deltaP_acceleration, rtol=1e-4))

    def test_deltaP_gravity(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        deltaP_gravity = self.solver.deltaP_gravity()
        self.assertEqual(len(deltaP_gravity), self.discretization['axial_nodes'])
        reference_deltaP_gravity = np.array([537.33, 535.49, 531.80, 528.11, 524.42,
                                             520.74, 517.05, 513.36, 509.67, 505.98,
                                             502.29, 498.60, 494.92, 491.23, 487.54,
                                             483.85, 480.16, 476.47, 472.78, 469.10,
                                             465.41, 461.72, 458.03, 454.34, 450.65])
        self.assertTrue(np.allclose(deltaP_gravity, reference_deltaP_gravity, rtol=1e-4))

    def test_res_enthalpy(self):
        T_rel, v_rel = self.solver.split_solution(self.sol)
        self.solver.neg_pos_surf(T_rel, v_rel)
        res_enthalpy = self.solver.res_enthalpy(self.qp)
        self.assertEqual(len(res_enthalpy), self.discretization['axial_nodes'])
        reference_res_enthalpy = np.array([-0.056404, 0.039782, 0.041521, 0.043206, 0.044840,
                                            0.046426, 0.047966, 0.049462, 0.050918, 0.052334,
                                            0.053714, 0.055058, 0.056370, 0.057650, 0.058900,
                                            0.060122, 0.061317, 0.062487, 0.063632, 0.064754,
                                            0.065853, 0.066931, 0.067988, 0.069024, 0.070041])
        self.assertTrue(np.allclose(res_enthalpy, reference_res_enthalpy, rtol=1e-5))

    def test_res(self):
        res = self.solver.res(self.sol, self.qp)
        self.assertEqual(len(res), self.solver.domain_size)
        reference_res = np.array([0, 0.0152051, 0.0149000, 0.0145949, 0.0142898,
                                  0.0139847, 0.0136797, 0.0133746, 0.0130695, 0.0127644,
                                  0.0124593, 0.0121542, 0.0118491, 0.0115440, 0.0112389,
                                  0.0109338, 0.0106287, 0.0103237, 0.0100186, 0.0097135,
                                  0.0094084, 0.0091033, 0.0087982, 0.0084931, 0.0081880,
                                  -0.056404, 0.039782, 0.041521, 0.043206, 0.044840,
                                  0.046426, 0.047966, 0.049462, 0.050918, 0.052334,
                                  0.053714, 0.055058, 0.056370, 0.057650, 0.058900,
                                  0.060122, 0.061317, 0.062487, 0.063632, 0.064754,
                                  0.065853, 0.066931, 0.067988, 0.069024, 0.070041])
        self.assertTrue(np.allclose(res, reference_res, rtol=1e-5))

    def test_solve(self):
        sol, info, ier, msg = fsolve(self.solver.res, np.ones(self.solver.domain_size), args=self.qp,
                                     full_output=True, xtol=1e-6)
        reference_sol = np.array([1.0262518, 1.0525815, 1.0789846, 1.1054561, 1.1319913,
                                  1.1585847, 1.1852313, 1.2119254, 1.2386616, 1.2654341,
                                  1.2922371, 1.3190648, 1.3459111, 1.3727701, 1.3996358,
                                  1.4265020, 1.4533626, 1.4802115, 1.5070427, 1.5338501,
                                  1.5606276, 1.5873694, 1.6140696, 1.6407224, 1.6673220,
                                  1.0048748, 1.0098121, 1.0148121, 1.0198750, 1.0250010,
                                  1.0301902, 1.0354428, 1.0407588, 1.0461381, 1.0515807,
                                  1.0570865, 1.0626555, 1.0682874, 1.0739820, 1.0797390,
                                  1.0855583, 1.0914393, 1.0973819, 1.1033855, 1.1094497,
                                  1.1155742, 1.1217583, 1.1280015, 1.1343034, 1.1406633])
        self.assertTrue(np.allclose(sol, reference_sol, rtol=1e-5))
        T_rel, v_rel = np.split(sol, 2)

        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax1.plot(T_rel)
        ax2 = fig.add_subplot(222)
        ax2.plot(v_rel)
        ax1 = fig.add_subplot(223)
        ax1.plot(T_rel * self.boundary_conditions['T_in'])
        ax2 = fig.add_subplot(224)
        ax2.plot(v_rel * self.boundary_conditions['v_in'])

        plt.show()


if __name__ == '__main__':
    unittest.main()
