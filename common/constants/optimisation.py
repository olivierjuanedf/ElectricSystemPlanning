from dataclasses import dataclass


@dataclass
class OptimSolvers:
    gurobi: str = 'gurobi'
    highs: str = 'highs'


@dataclass
class OptimResolStatus:
    optimal: str = 'optimal'
    infeasible: str = 'infeasible'
    

OPTIM_RESOL_STATUS = OptimResolStatus()


@dataclass
class SolverParams:
    name: str = 'highs'
    licence_file: str = None


DEFAULT_OPTIM_SOLVER_PARAMS = SolverParams(name=OptimSolvers.highs)
