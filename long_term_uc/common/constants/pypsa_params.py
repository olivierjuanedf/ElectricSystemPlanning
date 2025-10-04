from dataclasses import dataclass


@dataclass
class GenUnitsPypsaParams:
    carrier: str = 'carrier'
    capa_factors: str = 'p_max_pu'
    committable: str = 'committable'
    co2_emissions: str = 'co2_emissions'  # TODO: check that aligned on PyPSA generators attribute names
    efficiency: str = 'efficiency'
    energy_capa: str = None
    marginal_cost: str = 'marginal_cost'
    max_hours: str = 'max_hours'
    max_power_pu: str = 'p_max_pu'
    min_power_pu: str = 'p_min_pu'
    name: str = 'name'
    nominal_power: str = 'p_nom'
    power_capa: str = 'p_nom'


GEN_UNITS_PYPSA_PARAMS = GenUnitsPypsaParams()
