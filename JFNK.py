import numpy as np
from numpy import linalg as npla
from scipy.sparse import linalg as spla
from scipy.optimize import fsolve
import unittest


class JvApproximate(spla.LinearOperator):
    """
    Approximate the product of the Jacobian matrix and the solution vector
    """

    def __init__(self, F, Fu, u):
        """

        :param F: function that return residuals
        :param Fu: function taken at reference point u
        :param u: reference point u
        """
        self.F = F
        self.update(Fu, u)

        self.eps = np.finfo(float).eps

        # self.shape is given as a square matrix in order to TRICK a condition inside
        # GMRES that checks that for A.shape[0] == A.shape[1]
        # we don't need it because we are approximating Av
        self.size = len(u)
        self.shape = (self.size, self.size)
        self.dtype = np.dtype('float64')

        super().__init__(self.dtype, self.shape)

    def _matvec(self, du):
        """

        :param du: vector to be multiplied by the Jacobian
        :return: vector representing the first-order approximation of the product
                 between the Jacobian corresponding to the function F evaluated at the
                 reference point and the vector delta_u
        """
        norm = npla.norm(du, 2)

        sum_u = np.sum(100 * np.sqrt(self.eps) * (1 + np.abs(self.u)))

        if norm > self.eps:
            per = sum_u / (self.size * norm)
        else:
            per = sum_u / self.size

        u_per_du = self.u + per * du
        Fu_per_du = self.F(u_per_du)
        y = (Fu_per_du - self.Fu)/per

        return y

    def update(self, Fu, u):
        self.Fu = Fu
        self.u = u


# class JFNK_folver:
#
#     def __init__(self, function, initial_guess, tol_newton, tol_krylov, maxiter_krylov=None):
#         self.F = function
#         self.u0 = initial_guess
#
#         self.tol_newton = tol_newton
#         self.tol_krylov = tol_krylov
#
#         if maxiter_krylov is None:
#             self.maxiter_krylov = len(initial_guess)
#         else:
#             self.maxiter_krylov = maxiter_krylov
#
#         self.Fu = self.F(self.u0)
#
#     def update(self, du):


def JFNK_solver(function, initial_guess, tol_newton, tol_krylov, maxiter_krylov=None):
    """
    JFNK SOLVER

    :param function: function "F" on which the JFNK solver will be applied
    :param initial_guess: vector "u0" representing a starting guess of the solution
    :param tol_newton: tolerance of the newton outer loop
    :param tol_krylov: tolerance of the krylov inner loop
    :param maxiter_krylov: maximum number of krylov iterations
    :return: solution vector "u" to the problem, i.e. u fulfils F(u) = 0 within the given tolerance (tol_newton)
    """
    if maxiter_krylov is None:
        maxiter_krylov = len(initial_guess)

    u = initial_guess
    Fu = function(u)

    jv_approx = JvApproximate(function, Fu, u)

    norm = npla.norm(Fu, 2)
    counter = 0

    while norm > tol_newton:
        jv_approx.update(Fu, u)
        du, info = spla.gmres(jv_approx, -Fu, tol=tol_krylov, maxiter=maxiter_krylov)
        # du, info = spla.lgmres(j_v_approx, -Fu, tol=tol_krylov, maxiter=maxiter_krylov)
        # du, info = spla.bicgstab(j_v_approx, -Fu, tol=tol_krylov, maxiter=maxiter_krylov)
        u = u + du  # update the solution vector with better estimate
        Fu = function(u)  # update residual vector
        norm = npla.norm(Fu, 2)
        print(f'Newton iteration = {counter}; norm(res)= {norm:.2e}; Number of Krylov iterations = {info};')
        counter += 1

    return u


class JFNKTest(unittest.TestCase):

    def setUp(self):
        self.function = lambda x: np.array([np.exp(-np.exp(-x[0]-x[1])) - x[1]*(1+x[0]**2), x[0]*np.cos(x[1]) + x[1]*np.sin(x[0]) - 0.5])
        self.reference_sol = np.array([0.353247, 0.606082])

        self.initial_guess = np.zeros(2)
        self.tol_newton = 1E-8
        self.tol_krylov = 1E-7

    def test_function(self):
        sol = fsolve(self.function, self.initial_guess)
        self.assertTrue(np.allclose(sol, self.reference_sol, rtol=1e-5))

    def test_JFNK(self):
        sol = JFNK_solver(self.function, self.initial_guess, self.tol_newton, self.tol_krylov)
        self.assertTrue(np.allclose(sol, self.reference_sol, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
