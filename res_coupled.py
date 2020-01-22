import res_nk_rel as nk
import res_th_rel as th
import res_pin_rel as pin
from math import tau
import numpy as np
from scipy.optimize import fsolve
import unittest


class coupledSolver:
    """
    OVERALL RESIDUAL VECTOR
    """
    def __init__(self, geometry, discretization, physics_parameters, library, boundary_conditions):
        self.T_in = boundary_conditions['T_in']

        self.axial_nodes = discretization['axial_nodes']

        self.nkSolver = nk.nkSolver(discretization, boundary_conditions, library)
        self.thSolver = th.thSolver(geometry, discretization, physics_parameters,
                                    boundary_conditions)
        self.pinSolver = pin.pinSolver(geometry, discretization, boundary_conditions)

        # self.nkSolver = nk.nkSolver(self.discretization, self.boundary_conditions, self.library)
        # self.thSolver = th.thSolver(self.geometry, self.discretization, self.physics_parameters, self.boundary_conditions)
        # self.pinSolver = pin.pinSolver(self.geometry, self.discretization, self.boundary_conditions)

        self.domain_size = self.nkSolver.domain_size + self.thSolver.domain_size + self.pinSolver.domain_size

    def split_solution(self, sol):
        [sol_nk, sol_th, sol_pin] = np.split(sol, [self.nkSolver.domain_size, self.nkSolver.domain_size + self.thSolver.domain_size])
        return sol_nk, sol_th, sol_pin

    def res(self, sol):
        """
        :param sol: a vector representing the solution of the entire
                    coupled problem. It can be seen as a vector made of three sub-vectors:
                    - sol_nk is a vector representing: the volume-averaged neutron fluxes and
                      and the corresponding eigenvalue. The numbering of the elements
                      is from the highest neutron energies (from the inlet to the
                      outlet) to the lowest neutron energies (from the inlet to the outlet).
                      The last element correspond to the eigenvalue.
                    - sol_th is a vector representing:
                      * the surface-wise coolant temperature for the first half of the vector
                        (from the first surface above the inlet to the outlet)
                      * the surface-wise coolant velocity for the second half of the vector
                        (from the first surface above the inlet to the outlet)
                    - sol_pin is a vector representing: the volume-averaged temperature in
                      the fuel pin (i.e. pellet, gap, and cladding). The numbering of the elements
                      is from the inner fuel pin and all axial nodes (from the inlet to the
                      outlet) to the outer fuel pin and all axial nodes (from the inlet to the
                      outlet)
        :return: a vector representing the solution of the entire
                 coupled problem, also composed of one for nk, for th and for pin
        """
        sol_nk, sol_th, sol_pin = self.split_solution(sol)

        # STEP 1 - ESTIMATING NECESSARY VARIABLES FOR CALLING THE RESIDUALS OF THE NEUTRON TRANSPORT SOLVER

        T_tmp = sol_th[:self.axial_nodes]
        T_rel_tmp = np.concatenate([[1], T_tmp])

        T_rel_m = T_rel_tmp[:-1]
        T_rel_p = T_rel_tmp[1:]

        T_rel_ave = (T_rel_p + T_rel_m)/2
        T_fuel_rel = self.pinSolver.calculate_T_fuel_rel(sol_pin)

        res_nk = self.nkSolver.res(sol_nk, T_rel_ave, T_fuel_rel)

        # STEP 2 - ESTIMATING NECESSARY VARIABLES FOR CALLING THE RESIDUALS OF THE FLOW TRANSPORT SOLVER

        qp = self.nkSolver.calculate_qp(sol_nk[:self.nkSolver.phi_size])

        res_th = self.thSolver.res(sol_th, qp)

        # STEP 3 - ESTIMATING NECESSARY VARIABLES FOR CALLING THE RESIDUALS OF THE PIN TEMPERATURE SOLVER
        v_tmp = sol_th[self.axial_nodes:self.axial_nodes * 2]
        v_rel_tmp = np.concatenate([[1], v_tmp])

        res_pin = self.pinSolver.res(sol_pin, qp, v_tmp, T_tmp)

        res = np.concatenate([res_nk, res_th, res_pin])

        return res


class coupledTest(unittest.TestCase):

    def setUp(self):
        self.geometry = {'H': 1.6,
                         'Rfo': 3.57E-3,
                         'Rci': 3.685E-3,
                         'Rco': 0.00425,
                         'De':0.0039587,
                         'A': 1.32140e-5,
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
        self.physics_parameters = {'g': 9.81}
        self.boundary_conditions = {'T_in': 673,
                                    'v_in': 7.5,
                                    'qp_ave': 3e4}

        self.solver = coupledSolver(self.geometry, self.discretization, self.physics_parameters, 'XS_data.hdf5', self.boundary_conditions)

        self.sol_nk = np.outer(np.arange(self.solver.nkSolver.energy_groups) + 1, np.sin(
            (np.arange(self.discretization['axial_nodes']) + 0.5) / self.discretization['axial_nodes'] * (tau / 2)))/(self.solver.nkSolver.energy_groups+1)
        self.sol_nk = np.concatenate([self.sol_nk.ravel(), [1]])

        self.T_rel = (self.boundary_conditions['T_in'] + np.arange(self.discretization['axial_nodes']) * 600
                      / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['T_in']
        self.v_rel = (self.boundary_conditions['v_in'] + np.arange(self.discretization['axial_nodes']) * 4
                      / (self.discretization['axial_nodes'] - 1)) / self.boundary_conditions['v_in']
        self.sol_th = np.concatenate([self.T_rel, self.v_rel])

        self.sol_pin = np.outer((np.arange(self.solver.pinSolver.radial_nodes + 2) + 1)[::-1], np.sin(
            (np.arange(self.discretization['axial_nodes']) + 0.5) / self.discretization['axial_nodes'] * (tau / 2)))/(self.solver.nkSolver.energy_groups+1)
        self.sol_pin = self.sol_pin.ravel()

        self.sol_test = np.concatenate([self.sol_nk, self.sol_th, self.sol_pin])
        self.sol_guess = np.ones(self.solver.domain_size)

    def test_split(self):
        sol_nk, sol_th, sol_pin = self.solver.split_solution(self.sol_test)
        self.assertTrue(np.allclose(sol_nk, self.sol_nk, rtol=1e-5))
        self.assertTrue(np.allclose(sol_th, self.sol_th, rtol=1e-5))
        self.assertTrue(np.allclose(sol_pin, self.sol_pin, rtol=1e-5))

    def test_res(self):
        sol, info, ier, msg = fsolve(self.solver.res, self.sol_guess, full_output=True, xtol=1e-6)
        sol_nk, sol_th, sol_pin = self.solver.split_solution(sol)
        self.assertEqual(len(sol_nk), self.solver.nkSolver.domain_size)
        reference_sol_nk = np.array([5.26152e-2, 9.41874e-2, 1.36632e-1, 1.77881e-1, 2.16749e-1,
                                     2.52458e-1, 2.84426e-1, 3.12187e-1, 3.35359e-1, 3.53634e-1,
                                     3.66775e-1, 3.74615e-1, 3.77068e-1, 3.74093e-1, 3.65759e-1,
                                     3.52182e-1, 3.33554e-1, 3.10134e-1, 2.82248e-1, 2.50292e-1,
                                     2.14738e-1, 1.76165e-1, 1.35342e-1, 9.34250e-2, 5.24202e-2,
                                     1.07690e-1, 2.00851e-1, 2.93233e-1, 3.82205e-1, 4.65828e-1,
                                     5.42613e-1, 6.11358e-1, 6.71068e-1, 7.20918e-1, 7.60242e-1,
                                     7.88526e-1, 8.05412e-1, 8.10694e-1, 8.04327e-1, 7.86419e-1,
                                     7.57230e-1, 7.17175e-1, 6.66814e-1, 6.06851e-1, 5.38136e-1,
                                     4.61677e-1, 3.78671e-1, 2.90581e-1, 1.99290e-1, 1.07334e-1,
                                     2.24964e-1, 4.34028e-1, 6.36746e-1, 8.30500e-1, 1.01231,
                                     1.17936, 1.32912, 1.45944, 1.56847, 1.65471,
                                     1.71699, 1.75450, 1.76673, 1.75355, 1.71516,
                                     1.65209, 1.56523, 1.45578, 1.32528, 1.17562,
                                     1.00897, 8.27852e-1, 6.35062e-1, 4.33524e-1, 2.25794e-1,
                                     2.68573e-1, 5.40758e-1, 8.00984e-1, 1.04763, 1.27801,
                                     1.48916, 1.67819, 1.84246, 1.97970, 2.08806,
                                     2.16607, 2.21273, 2.22744, 2.21007, 2.16092,
                                     2.08072, 1.97060, 1.83215, 1.66731, 1.47842,
                                     1.26809, 1.03921, 7.94641e-1, 5.36937e-1, 2.67558e-1,
                                     2.07950e-1, 4.37085e-1, 6.54403e-1, 8.59099e-1, 1.04951,
                                     1.22358, 1.37916, 1.51423, 1.62697, 1.71589,
                                     1.77982, 1.81793, 1.82977, 1.81522, 1.77456,
                                     1.70839, 1.61767, 1.50370, 1.36804, 1.21256,
                                     1.03927, 8.50293e-1, 6.47597e-1, 4.32731e-1, 2.06380e-1,
                                     1.33831e-1, 2.91554e-1, 4.40542e-1, 5.80272e-1, 7.09774e-1,
                                     8.27811e-1, 9.33051e-1, 1.02421, 1.10011, 1.15980,
                                     1.20250, 1.22770, 1.23512, 1.22473, 1.19672,
                                     1.15155, 1.08989, 1.01261, 9.20773e-1, 8.15586e-1,
                                     6.98343e-1, 5.70334e-1, 4.32719e-1, 2.86363e-1, 1.31676e-1,
                                     1.27075e-1, 2.91993e-1, 4.46339e-1, 5.89997e-1, 7.22327e-1,
                                     8.42345e-1, 9.48905e-1, 1.04085, 1.11711, 1.17679,
                                     1.21920, 1.24386, 1.25057, 1.23931, 1.21036,
                                     1.16416, 1.10142, 1.02300, 9.29941e-1, 8.23383e-1,
                                     7.04503e-1, 5.74405e-1, 4.33977e-1, 2.83722e-1, 1.23591e-1,
                                     1.10839e-2, 2.53548e-2, 3.84172e-2, 5.02498e-2, 6.08721e-2,
                                     7.02702e-2, 7.84013e-2, 8.52135e-2, 9.06615e-2, 9.47154e-2,
                                     9.73625e-2, 9.86083e-2, 9.84744e-2, 9.69972e-2, 9.42261e-2,
                                     9.02211e-2, 8.50518e-2, 7.87947e-2, 7.15304e-2, 6.33384e-2,
                                     5.42874e-2, 4.44193e-2, 3.37295e-2, 2.21607e-2, 9.66805e-3,
                                     1.11666])
        self.assertTrue(np.allclose(sol_nk, reference_sol_nk, rtol=5e-5))
        self.assertEqual(len(sol_th), self.solver.thSolver.domain_size)
        reference_sol_th = np.array([1.00472, 1.01432, 1.02863, 1.04740, 1.07037,
                                     1.09718, 1.12747, 1.16078, 1.19663, 1.23450,
                                     1.27383, 1.31405, 1.35454, 1.39472, 1.43398,
                                     1.47174, 1.50747, 1.54062, 1.57074, 1.59740,
                                     1.62021, 1.63885, 1.65306, 1.66261, 1.66732,
                                     1.00087, 1.00265, 1.00532, 1.00884, 1.01317,
                                     1.01829, 1.02412, 1.03062, 1.03771, 1.04530,
                                     1.05330, 1.06161, 1.07011, 1.07868, 1.08719,
                                     1.09550, 1.10348, 1.11099, 1.11791, 1.12410,
                                     1.12945, 1.13386, 1.13724, 1.13953, 1.14066])
        self.assertTrue(np.allclose(sol_th, reference_sol_th, rtol=5e-5))
        self.assertEqual(len(sol_pin), self.solver.pinSolver.domain_size)
        reference_sol_pin = np.array([1.33514, 1.63500, 1.89261, 2.11228, 2.29788,
                                      2.45546, 2.59026, 2.70568, 2.80372, 2.88551,
                                      2.95173, 3.00285, 3.03920, 3.06100, 3.06833,
                                      3.06110, 3.03894, 3.00100, 2.94564, 2.86987,
                                      2.76854, 2.63377, 2.45577, 2.22774, 1.95444,
                                      1.29076, 1.55485, 1.77785, 1.96506, 2.12154,
                                      2.25419, 2.36859, 2.46818, 2.55473, 2.62898,
                                      2.69119, 2.74142, 2.77967, 2.80587, 2.81996,
                                      2.82175, 2.81088, 2.78654, 2.74722, 2.69007,
                                      2.61018, 2.50004, 2.35036, 2.15469, 1.91732,
                                      1.24646, 1.47497, 1.66440, 1.81952, 1.94648,
                                      2.05283, 2.14467, 2.22573, 2.29788, 2.36186,
                                      2.41779, 2.46549, 2.50473, 2.53522, 2.55669,
                                      2.56880, 2.57106, 2.56259, 2.54180, 2.50583,
                                      2.44977, 2.36619, 2.24604, 2.08299, 1.88103,
                                      1.20135, 1.39284, 1.54902, 1.67204, 1.76898,
                                      1.84810, 1.91597, 1.97672, 2.03248, 2.08415,
                                      2.13191, 2.17560, 2.21486, 2.24929, 2.27844,
                                      2.30180, 2.31866, 2.32795, 2.32787, 2.31533,
                                      2.28526, 2.23011, 2.14083, 2.01110, 1.84477,
                                      1.15536, 1.30735, 1.42966, 1.52057, 1.58719,
                                      1.63851, 1.68168, 1.72124, 1.75966, 1.79805,
                                      1.83674, 1.87563, 1.91438, 1.95253, 1.98951,
                                      2.02464, 2.05700, 2.08527, 2.10735, 2.11983,
                                      2.11730, 2.09196, 2.03459, 1.93881, 1.80842,
                                      1.06975, 1.14173, 1.20303, 1.25160, 1.29057,
                                      1.32427, 1.35616, 1.38831, 1.42179, 1.45694,
                                      1.49370, 1.53177, 1.57072, 1.61005, 1.64919,
                                      1.68750, 1.72423, 1.75843, 1.78879, 1.81334,
                                      1.82919, 1.83227, 1.81790, 1.78284, 1.72884,
                                      1.00559, 1.01602, 1.03103, 1.05035, 1.07367,
                                      1.10065, 1.13090, 1.16399, 1.19947, 1.23683,
                                      1.27557, 1.31514, 1.35497, 1.39451, 1.43319,
                                      1.47047, 1.50580, 1.53870, 1.56869, 1.59534,
                                      1.61828, 1.63717, 1.65171, 1.66167, 1.66685])
        self.assertTrue(np.allclose(sol_pin, reference_sol_pin, rtol=5e-5))


if __name__ == '__main__':
    unittest.main()
