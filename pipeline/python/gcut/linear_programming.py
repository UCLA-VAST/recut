import numpy as np
from ortools.linear_solver import pywraplp
from pipeline_util import timer


class LPSolver:
    def __init__(self):
        self.solver = pywraplp.Solver('GCut',  pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        # careful to always prepopulate variabes as self.variables = [[]] * n
        # the c++ call to NumVar returns a pointer to the variable
        # if list.append() call or other calls that copies the added object
        # is used, updates to self.variables will be disassociated with the
        # solver, who is meant to own the variables
        self.variables = None

    def create_variables(self, n, lbs, ubs):
        assert n > 0
        assert len(lbs) == len(ubs) == n
        self.variables = [[]] * n
        for i in range(n):
            self.variables[i] = self.solver.NumVar(lbs[i], ubs[i], 'v{}'.format(i))

    def add_constraint(self, variable_indices, coefficients, lb, ub):
        assert len(variable_indices) == len(coefficients)
        constraint = self.solver.Constraint(lb, ub)
        for index, coefficient in zip(variable_indices, coefficients):
            constraint.SetCoefficient(self.variables[index], coefficient)

    def define_objective(self, coefficients, minimize=True):
        assert len(coefficients) == len(self.variables)
        objective = self.solver.Objective()
        for i in range(len(coefficients)):
            objective.SetCoefficient(self.variables[i], coefficients[i])
        objective.SetMinimization() if minimize else objective.SetMaximization()

    def n_variables(self):
        return self.solver.NumVariables()

    def n_constraints(self):
        return self.solver.NumConstraints()

    @timer
    def solve(self):
        self.solver.Solve()

    def solutions(self):
        return np.array([v.solution_value() for v in self.variables])

    def clear(self):
        self.solver.Clear()
        self.variables = None
