import initialize as ini
import res_coupled as coup
import yaml
from JFNK import JFNK_solver
import numpy as np
from scipy.optimize import newton_krylov, fsolve

with open('input.yml', 'r') as input:
    caseInput = yaml.safe_load(input)
    caseDef = ini.CaseDefinition(caseInput)

    coupler = coup.coupledSolver(caseDef.geometry, caseDef.discretization, caseDef.physics_parameters, caseDef.xs_file,
                                 caseDef.boundary_conditions)

    sol_guess = np.ones(coupler.domain_size)
    tol_newton = caseDef.numerics['tol_newton']
    tol_krylov = caseDef.numerics['tol_krylov']

    sol = JFNK_solver(coupler.res, sol_guess, tol_newton, tol_krylov)
    # sol = fsolve(coupler.res, sol_guess)
    print('general solution is:\n', sol)

    sol_nk, sol_th, sol_pin = coupler.split_solution(sol)
    print('sol_nk is:\n', sol_nk)
    print('sol_th is:\n', sol_th)
    print('sol_pin is:\n', sol_pin)

