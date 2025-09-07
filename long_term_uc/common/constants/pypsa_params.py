from dataclasses import dataclass


@dataclass
class GenUnitsPypsaParams:
    capa_factors: str = 'p_max_pu' 
    power_capa: str = 'p_nom'
    min_power: str = 'p_min_pu'
    marginal_cost: str = 'marginal_cost'
    co2_emissions: str = 'co2_emissions'  # TODO: check that aligned on PyPSA generators attribute names
    committable: str = 'committable'
    max_hours: str = 'max_hours'
    energy_capa: str = None

GEN_UNITS_PYPSA_PARAMS = GenUnitsPypsaParams()
