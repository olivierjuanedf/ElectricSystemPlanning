from dataclasses import dataclass

"""
**[Optional, for better parametrization of assets]**
"""


@dataclass
class FuelNames:
    biomass: str = 'biomass'
    coal: str = 'coal'
    gas: str = 'gas'
    hydro: str = 'hydro'
    oil: str = 'oil'
    other_renewables: str = 'other_renew'
    solar: str = 'solar'
    uranium: str = 'uranium'
    wind: str = 'wind'


@dataclass
class DummyFuelNames:
    bus: str = 'bus'
    demande_side_resp: str = 'dsr'
    failure: str = 'failure'
    flexibility: str = 'flexibility'
    link: str = 'link'
    load: str = 'load'


@dataclass
class FuelSources:
    name: str
    co2_emissions: float
    committable: bool = None
    energy_density_per_ton: float = None  # in MWh / ton
    cost_per_ton: float = None
    primary_cost: float = None  # € / MWh (multiply this by the efficiency of your power plant to get the marginal cost)

    # [Coding trick] this function will be applied automatically at initialization of an object of this class
    def __post_init__(self):
        if self.cost_per_ton is None or self.energy_density_per_ton is None:
            self.primary_cost = None
        elif self.energy_density_per_ton != 0:
            self.primary_cost = self.cost_per_ton / self.energy_density_per_ton
        else:
            self.primary_cost = 0


FUEL_SOURCES = {
    FuelNames.coal: FuelSources(FuelNames.coal.capitalize(), 760, True, 8, 128),
    FuelNames.gas: FuelSources(FuelNames.gas.capitalize(), 370, True, 14.89, 134.34),
    FuelNames.oil: FuelSources(FuelNames.oil.capitalize(), 406, True, 11.63, 555.78),
    FuelNames.uranium: FuelSources(FuelNames.uranium.capitalize(), 0, True, 22394, 150000.84),
    FuelNames.solar: FuelSources(FuelNames.solar.capitalize(), 0, False, 0, 0),
    FuelNames.wind: FuelSources(FuelNames.wind.capitalize(), 0, False, 0, 0),
    FuelNames.hydro: FuelSources(FuelNames.hydro.capitalize(), 0, True, 0, 0),
    FuelNames.biomass: FuelSources(FuelNames.biomass.capitalize(), 0, True, 5, 30)
}
# to have carriers defined for all prod units in PyPSA
# TODO: make code ok without dummy CO2 emission values
dummy_co2_emissions = 0
DUMMY_FUEL_SOURCES = {DummyFuelNames.failure: FuelSources(DummyFuelNames.failure.capitalize(), dummy_co2_emissions),
                      FuelNames.other_renewables: FuelSources(FuelNames.other_renewables.capitalize(),
                                                              FUEL_SOURCES[FuelNames.biomass].co2_emissions,
                                                              FUEL_SOURCES[FuelNames.biomass].committable,
                                                              FUEL_SOURCES[FuelNames.biomass].energy_density_per_ton,
                                                              FUEL_SOURCES[FuelNames.biomass].cost_per_ton),
                      DummyFuelNames.flexibility: FuelSources(DummyFuelNames.flexibility.capitalize(),
                                                              dummy_co2_emissions),
                      DummyFuelNames.demande_side_resp: FuelSources(DummyFuelNames.demande_side_resp.capitalize(),
                                                                    dummy_co2_emissions),
                      DummyFuelNames.link: FuelSources(DummyFuelNames.link.capitalize(), dummy_co2_emissions),
                      DummyFuelNames.bus: FuelSources(DummyFuelNames.bus.capitalize(), dummy_co2_emissions),
                      DummyFuelNames.load: FuelSources(DummyFuelNames.load.capitalize(), dummy_co2_emissions)
                      }
