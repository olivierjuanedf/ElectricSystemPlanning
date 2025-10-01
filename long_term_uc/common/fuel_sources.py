from dataclasses import dataclass

"""
**[Optional, for better parametrization of assets]**
"""

# [Coding trick] dataclass is a simple way to define object to store multiple attributes
@dataclass
class FuelSources:
    name: str
    co2_emissions: float
    committable: bool
    min_up_time: float
    min_down_time: float
    energy_density_per_ton: float  # in MWh / ton
    cost_per_ton: float
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
    'coal': FuelSources('Coal', 760, True, 4, 4, 8, 128),
    'gas': FuelSources('Gas', 370, True, 2, 2, 14.89, 134.34),
    'oil': FuelSources('Oil', 406, True, 1, 1, 11.63, 555.78),
    'uranium': FuelSources('Uranium', 0, True, 10, 10, 22394, 150000.84),
    'solar': FuelSources('Solar', 0, False, 1, 1, 0, 0),
    'wind': FuelSources('Wind', 0, False, 1, 1, 0, 0),
    'hydro': FuelSources('Hydro', 0, True, 2, 2, 0, 0),
    'biomass': FuelSources('Biomass', 0, True, 2, 2, 5, 30)
}
# to have carriers defined for all prod units in PyPSA
# TODO: make code ok without dummy CO2 emission values
dummy_co2_emissions = 50
DUMMY_FUEL_SOURCES = {'failure': FuelSources('Failure', dummy_co2_emissions, None, None, None, None, None),
                      'other_renew': FuelSources('Other_renew', 
                                                 FUEL_SOURCES['biomass'].co2_emissions, 
                                                 FUEL_SOURCES['biomass'].committable, 
                                                 FUEL_SOURCES['biomass'].min_up_time, 
                                                 FUEL_SOURCES['biomass'].min_down_time, 
                                                 FUEL_SOURCES['biomass'].energy_density_per_ton, 
                                                 FUEL_SOURCES['biomass'].cost_per_ton),
                      'flexibility': FuelSources('Failure', dummy_co2_emissions, None, None, None, None, None),
                      'dsr': FuelSources('Failure', dummy_co2_emissions, None, None, None, None, None)
                      }
