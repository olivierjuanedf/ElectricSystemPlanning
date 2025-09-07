from dataclasses import dataclass


@dataclass
class OptimResolStatus:
    optimal: str = 'optimal'
    infeasible: str = 'infeasible'
    

OPTIM_RESOL_STATUS = OptimResolStatus()
