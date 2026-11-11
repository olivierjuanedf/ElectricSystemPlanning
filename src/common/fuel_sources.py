import logging
from dataclasses import dataclass
from typing import Dict

from src.common.logger import format_relative_path_for_log
from src.common.long_term_uc_io import get_json_fuel_sources_raw_params_file, get_json_fuel_sources_tb_modif_file
from src.utils.read import check_and_load_json_file

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
    other_non_renewables: str = 'other_non_renew'
    other_renewables: str = 'other_renew'
    solar: str = 'solar'
    uranium: str = 'uranium'
    wind: str = 'wind'


@dataclass
class DummyFuelNames:
    ac: str = 'ac'
    dc: str = 'dc'


@dataclass
class FuelSource:
    name: str
    co2_emissions: float
    committable: bool = None
    energy_density_per_ton: float = None  # in MWh / ton
    cost_per_ton: float = None
    primary_cost: float = None  # € / MWh (multiply this by the efficiency of your power plant to get the marginal cost)
    # with a little abuse, associate marginal cost to fuel params to modify it from
    # input/long_term_uc_fuel_params_to-be_modif.json
    marginal_cost: float = None

    # [Coding trick] this function will be applied automatically at initialization of an object of this class
    def __post_init__(self):
        if self.cost_per_ton is None or self.energy_density_per_ton is None:
            self.primary_cost = None
        elif self.energy_density_per_ton != 0:
            self.primary_cost = self.cost_per_ton / self.energy_density_per_ton
        else:
            self.primary_cost = 0

    def set_co2_emissions(self, value: float):
        self.co2_emissions = value

    def set_marginal_cost(self, value: float):
        self.marginal_cost = value

    def reset_cost_params(self):
        self.cost_per_ton = None
        self.energy_density_per_ton = None
        self.primary_cost = None
        self.marginal_cost = None

    def reset_marginal_cost(self):
        self.marginal_cost = None


def add_other_sources(fuel_sources: Dict[str, FuelSource]) -> Dict[str, FuelSource]:
    source_matching = {FuelNames.other_renewables: FuelNames.biomass, FuelNames.other_non_renewables: FuelNames.oil}
    matched_params = ['co2_emissions', 'committable', 'energy_density_per_ton', 'cost_per_ton']
    for name, matched_name in source_matching.items():
        fs_params = [name.capitalize()]
        current_matched_params = fuel_sources[matched_name].__dict__
        fs_params.extend([current_matched_params[elt] for elt in matched_params])
        fuel_sources[name] = FuelSource(*fs_params)
    return fuel_sources


def get_fuel_sources_with_set_marg_cost(fuel_sources: dict[str, list[FuelSource]]) -> dict[str, list[FuelSource]]:
    return {name: source for name, source in fuel_sources.items() if source.marginal_cost is not None}


def set_fuel_sources_from_raw_params_json(add_other_fs: bool = True, reset_detailed_cost_params: bool = True) \
        -> dict[str, FuelSource]:
    """
    Set fuel sources paremeters from detailed file data/fuel_sources/raw_params.json
    :param add_other_fs
    :param reset_detailed_cost_params: to avoid confusion with marginal_cost parameter values fixed from
    input/_env_params/elec-europe_params_fixed.json, reset detailed infos to None?
    """
    filepath = get_json_fuel_sources_raw_params_file()
    fuel_sources_data = check_and_load_json_file(json_file=filepath, file_descr='JSON fuel sources params')
    fuel_sources = {}
    for fuel_name, param_vals in fuel_sources_data.items():
        param_vals |= {'name': fuel_name.capitalize()}
        fuel_sources[fuel_name] = FuelSource(**param_vals)
    if add_other_fs:
        fuel_sources = add_other_sources(fuel_sources=fuel_sources)
    fuel_sources = reset_fuel_sources_cost_params(fuel_sources)
    return fuel_sources


def reset_fuel_sources_cost_params(fuel_sources: dict[str, list[FuelSource]]) -> dict[str, list[FuelSource]]:
    """
    Rest (detailed) cost params of fuel sources, to avoid confusing with values set from
    input/long_term_uc/fuel_sources_to-be_modif.json
    """
    for name, fuel_source in fuel_sources.items():
        fuel_source.reset_cost_params()
    return fuel_sources


def reset_fuel_sources_marginal_costs(fuel_sources: dict[str, list[FuelSource]]) -> dict[str, list[FuelSource]]:
    for name, fuel_source in fuel_sources.items():
        fuel_source.reset_marginal_cost()
    return fuel_sources


def update_fuel_sources(fuel_sources: dict[str, FuelSource], updated_params: dict[str, dict[str, float]],
                        print_out_modif_marg_cost: bool = False) -> dict[str, FuelSource]:
    """
    Update some marginal cost, CO2 emission values based on JSON file in input folder
    :param fuel_sources
    :param updated_params
    :param print_out_modif_marg_cost: can be confusing to print the changed values, given that initial ones are None
    -> changed values will be printed out when updating generation units data hereafter
    """
    modif_fuel_sources_params = {}
    for name, fuel_source in fuel_sources.items():
        if name in updated_params:
            values_tb_modif = updated_params[name]
            if "co2_emissions" in values_tb_modif:
                new_value = values_tb_modif["co2_emissions"]
                if fuel_source.co2_emissions is None or not new_value == fuel_source.co2_emissions:
                    change_tuple = ("co2_emissions", fuel_source.co2_emissions, new_value)
                    if name not in modif_fuel_sources_params:
                        modif_fuel_sources_params[name] = []
                    modif_fuel_sources_params[name].append(change_tuple)
                    fuel_source.set_co2_emissions(new_value)
            if "marginal_cost" in values_tb_modif:
                new_value = values_tb_modif["marginal_cost"]
                if fuel_source.marginal_cost is None or not new_value == fuel_source.marginal_cost:
                    change_tuple = ("marginal_cost", fuel_source.marginal_cost, new_value)
                    if print_out_modif_marg_cost:
                        if name not in modif_fuel_sources_params:
                            modif_fuel_sources_params[name] = []
                        modif_fuel_sources_params[name].append(change_tuple)
                    fuel_source.set_marginal_cost(new_value)
    if len(modif_fuel_sources_params) > 0:
        json_fuel_param_filepath = format_relative_path_for_log(path=get_json_fuel_sources_tb_modif_file())
        changed_content_msg = "CO2 emission and marginal_cost" if print_out_modif_marg_cost else "CO2 emission"
        if not print_out_modif_marg_cost:  # remove "co2_emissions" from change tuple
            modif_fuel_sources_params = {name: tuple(change_tuples[0][1:])
                                         for name, change_tuples in modif_fuel_sources_params.items()}
        tuple_def_msg = "(content, init. value, new one)" if print_out_modif_marg_cost else "(init. value, new one)"
        logging.info(f"The following {changed_content_msg} values have been modified in fuel sources from "
                     f"file {json_fuel_param_filepath} - each tuple of the form {tuple_def_msg}: "
                     f"{modif_fuel_sources_params}")
    return fuel_sources


# to have carriers defined for all prod units in PyPSA
# TODO: make code ok without dummy CO2 emission values
dummy_co2_emissions = 0
DUMMY_FUEL_SOURCES = {DummyFuelNames.ac: FuelSource(DummyFuelNames.ac.capitalize(), dummy_co2_emissions),
                      DummyFuelNames.dc: FuelSource(DummyFuelNames.dc.capitalize(), dummy_co2_emissions)
                      }
