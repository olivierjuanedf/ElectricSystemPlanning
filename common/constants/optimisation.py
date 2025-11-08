from dataclasses import dataclass


@dataclass
class OptimSolvers:
    gurobi: str = 'gurobi'
    highs: str = 'highs'


DEFAULT_OPTIM_SOLVER = OptimSolvers.highs


@dataclass
class OptimResolStatus:
    optimal: str = 'optimal'
    infeasible: str = 'infeasible'
    

OPTIM_RESOL_STATUS = OptimResolStatus()
