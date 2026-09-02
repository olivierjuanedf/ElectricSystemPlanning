from pypsa import Network
import numpy as np
from typing import Dict


def get_generators_opt_p(network: Network) -> Dict[str, np.array]:
    generator_names = list(network.generators_t.p.columns)
    return {name: np.array(network.generators_t.p['Hard-Coal_ita']) for name in generator_names}


def generators_opt_p_to_csv():
    return None


def get_network_obj_value(network: Network) -> float:
    if hasattr(network, 'objective') and network.objective is not None:
        try:
            return float(network.objective)
        except (TypeError, ValueError):
            pass
    if hasattr(network, 'model') and network.model is not None:
        try:
            return float(network.model.objective.value)
        except (AttributeError, TypeError, ValueError):
            pass
    return 0.0
