from dataclasses import dataclass


@dataclass
class GenUnitsPypsaParams:
    bus: str = 'bus'
    carrier: str = 'carrier'
    capa_factors: str = 'p_max_pu'
    committable: str = 'committable'
    co2_emissions: str = 'co2_emissions'  # TODO: check that aligned on PyPSA generators attribute names
    efficiency: str = 'efficiency'
    efficiency_store: str = 'efficiency_store'  # if store/discharge efficiency to be distinguished, e.g. for stocks
    efficiency_dispatch: str = 'efficiency_dispatch'
    energy_capa: str = None
    inflow: str = 'inflow'
    marginal_cost: str = 'marginal_cost'
    max_hours: str = 'max_hours'
    max_power_pu: str = 'p_max_pu'
    min_power_pu: str = 'p_min_pu'
    name: str = 'name'
    nominal_power: str = 'p_nom'
    power_capa: str = 'p_nom'
    soc_init: float = 'state_of_charge_initial'
    set_power: str = 'p_set'


GEN_UNITS_PYPSA_PARAMS = GenUnitsPypsaParams()


@dataclass
class GenUnitsCustomParams:
    soc_min: str = 'soc_min'
    soc_max: str = 'soc_max'
    gen_min: str = 'gen_min'
    gen_max: str = 'gen_max'


@dataclass
class PypsaOptimVarNames:
    # related to generators
    generators_p: str = 'Generator-p'
    # related to storage units
    storage_p_dispatch: str = 'StorageUnit-p_dispatch'
    storage_p_store: str = 'StorageUnit-p_store'
    storage_soc: str = 'StorageUnit-state_of_charge'
    storage_spill: str = 'StorageUnit-spill'
    # related to links
    links_p: str = 'Link-p'


PYPSA_TO_EXPLICIT_VAR_NAMES = \
    {PypsaOptimVarNames.generators_p: 'p',
     PypsaOptimVarNames.links_p: 'p',
     PypsaOptimVarNames.storage_p_dispatch: 'p',
     PypsaOptimVarNames.storage_p_store: 'c',
     PypsaOptimVarNames.storage_soc: 'soc',
     PypsaOptimVarNames.storage_spill: 'spill'}


@dataclass
class PypsaOptimConstNames:
    generators_p_lb: str = 'Generator-fix-p-lower'
    generators_p_ub: str = 'Generator-fix-p-upper'
    link_lb: str = 'Link-fix-p-lower'
    link_ub: str = 'Link-fix-p-upper'
    storage_p_dispatch_lb: str = 'StorageUnit-fix-p_dispatch-lower'
    storage_p_dispatch_ub: str = 'StorageUnit-fix-p_dispatch-upper'
    storage_p_store_lb: str = 'StorageUnit-fix-p_store-lower'
    storage_p_store_ub: str = 'StorageUnit-fix-p_store-upper'
    storage_soc_state_eq: str = 'StorageUnit-energy_balance'
    storage_soc_lb: str = 'StorageUnit-fix-state_of_charge-lower'
    storage_soc_ub: str = 'StorageUnit-fix-state_of_charge-upper'
    supply_demand_eq: str = 'Bus-nodal_balance'


PYPSA_TO_EXPLICIT_CONST_NAMES = \
    {PypsaOptimConstNames.generators_p_lb: "p_lb",
     PypsaOptimConstNames.generators_p_ub: "p_ub",
     PypsaOptimConstNames.link_lb: "p_lb",
     PypsaOptimConstNames.link_ub: "p_lb",
     PypsaOptimConstNames.storage_p_dispatch_lb: "p_lb",
     PypsaOptimConstNames.storage_p_dispatch_ub: "p_ub",
     PypsaOptimConstNames.storage_p_store_lb: "c_lb",
     PypsaOptimConstNames.storage_p_store_ub: "c_ub",
     PypsaOptimConstNames.storage_soc_state_eq: "soc_stateeq",
     PypsaOptimConstNames.storage_soc_lb: "soc_lb",
     PypsaOptimConstNames.storage_soc_ub: "soc_ub",
     PypsaOptimConstNames.supply_demand_eq: "sde"}
