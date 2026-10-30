import os
import warnings
from itertools import product
from pathlib import Path
from datetime import datetime

import linopy.model
import xarray
import xarray as xr
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pypsa
import matplotlib.pyplot as plt
import re

from src.common.constants.countries import set_country_trigram, is_country_trigram
from src.common.constants.links import set_link_name, get_bus_from_link_name
from src.common.constants.optimisation import OptimSolvers, DEFAULT_OPTIM_SOLVER_PARAMS, SolverParams, \
    OptimPbCharacteristics, OptimPbTypes, CustomConstraintDirection, ZoneAndTempProdSumConstraint, CustomConstraintNames
from src.common.constants.prod_types import get_country_from_unit_name, ProdTypeNames, PROD_TYPE_IN_LP, \
    get_prod_type_from_unit_name
from src.common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS, PypsaOptimVarNames, PYPSA_TO_EXPLICIT_VAR_NAMES, \
    PYPSA_TO_EXPLICIT_CONST_NAMES
from src.common.constants.temporal import Timescale
from src.common.error_msgs import print_errors_list
from src.common.fuel_sources import FuelSource
from src.common.long_term_uc_io import get_network_figure, FigNamesPrefix, get_output_figure
from src.include.uc_postprocessing import UCSummaryMetrics, UCOptimalSolution
from src.utils.basic_utils import (lexico_compar_str, rm_elts_with_none_val, rm_elts_in_str, sort_lexicographically,
                                   format_with_spaces, get_default_values)
from src.utils.dir_utils import make_dir
from src.utils.pypsa_utils import get_network_obj_value
from src.utils.serializer import array_serializer


@dataclass
class GenerationUnitData:
    name: str
    type: str
    carrier: str = None
    p_nom: Union[float, np.ndarray] = None
    p_min_pu: Union[float, np.ndarray] = None
    p_max_pu: Union[float, np.ndarray] = None
    efficiency: float = None
    efficiency_store: float = None
    efficiency_dispatch: float = None
    marginal_cost: float = None
    committable: bool = False
    max_hours: float = None
    cyclic_state_of_charge: bool = None
    inflow: np.ndarray = None
    soc_min: np.ndarray = None
    soc_max: np.ndarray = None
    # extreme generation levels, over an aggregate temporal period - typically week in ERAA data for hydro. reservoirs
    gen_min: np.ndarray = None
    gen_max: np.ndarray = None
    state_of_charge_initial: float = None

    def get_non_none_attr_names(self):
        return [key for key, val in self.__dict__.items() if val is not None]

    def serialize(self) -> dict:
        unit_data_dict = self.__dict__
        # (1d) nd array to list
        unit_data_dict = {key: array_serializer(my_array=val, stat_repres=True) if isinstance(val, np.ndarray) else val
                          for key, val in unit_data_dict.items()}
        return unit_data_dict


def select_gen_units_data(gen_units_data: List[GenerationUnitData], countries: List[str],
                          unit_types: List[str]) -> List[GenerationUnitData]:
    return [elt for elt in gen_units_data
            if get_country_from_unit_name(elt.name) in countries and elt.type in unit_types]


GEN_UNITS_DATA_TYPE = Dict[str, List[GenerationUnitData]]
PYPSA_RESULT_TYPE = Tuple[str, str]


def check_gen_unit_params(params: dict, n_ts: int) -> bool:
    # check that max and min power pu are either constant or of the length of considered horizon
    for param_name in [GEN_UNITS_PYPSA_PARAMS.min_power_pu, GEN_UNITS_PYPSA_PARAMS.max_power_pu]:
        if param_name in params:
            param_value = params[param_name]
            if isinstance(param_value, list) or isinstance(param_value, np.ndarray):
                if not len(param_value) == n_ts:
                    return False
    return True


def set_per_bus_asset_msg(asset_names: List[str]):
    """
    List of {bus name}_{asset name} to be converted to more elegant msg
    """
    name_sep = '_'
    # get dict. gathering asset names per bus
    per_bus_assets = {}
    for full_name in asset_names:
        name_split = full_name.split(name_sep)
        bus_name = name_split[0]
        asset_name = name_sep.join(name_split[1:])
        if bus_name not in per_bus_assets:
            per_bus_assets[bus_name] = []
        per_bus_assets[bus_name].append(asset_name)
    # get log message with one line per bus
    per_bus_msg = ''
    for bus, assets in per_bus_assets.items():
        per_bus_msg += f'\n- {bus}: {assets}'
    return per_bus_msg


def set_per_origin_bus_links_msg(link_names: List[str]) -> str:
    link_sep = '-'
    links_msg = ''
    n_links = len(link_names)
    i_link = 0
    while i_link < n_links:
        common_origin_links = [link_names[i_link]]
        origin = link_names[i_link].split(link_sep)[0]
        j = 1
        while i_link + j < n_links:
            current_origin = link_names[i_link + j].split(link_sep)[0]
            if current_origin == origin:
                common_origin_links.append(link_names[i_link + j])
                j += 1
            else:
                break
        links_msg += f'\n- from {origin}: {[tuple(elt_link.split(link_sep)) for elt_link in common_origin_links]}'
        i_link += j
    return links_msg


def set_optim_pb_type(model: linopy.model.Model) -> Optional[str]:
    if model.is_linear:
        if len(model.integers) > 0:
            return OptimPbTypes.milp
        else:
            return OptimPbTypes.lp
    # quadratic if not linear (or other possibilities?)
    if model.is_quadratic:
        if len(model.integers) > 0:
            return OptimPbTypes.miqp
        else:
            return OptimPbTypes.qp
    return None


def prepro_bound_params(bound_values: Dict[str, Union[float, np.ndarray]], max_feas_bound: Dict[str, float],
                        n_ts: int) -> Dict[str, np.ndarray]:
    """
    Preprocess extreme (min/max) SoC values
    :param bound_values: to be processed, made of per asset (name) constant or vectorial value
    :param max_feas_bound: of the considered assets
    :param n_ts: number of time-slots in considered model
    """
    # project bound min/max values on [0, capa.], to induce real constraints (not <0, or bigger than energy capacity)
    bound_values = {asset_name: np.maximum(0, np.minimum(max_feas_bound[asset_name], extr_vect))
                    for asset_name, extr_vect in bound_values.items()}
    # cast constant values as vectors. TODO: Q necessary? (or Linopy can broadcast?)
    bound_values = {asset_name: extr_vect if isinstance(extr_vect, np.ndarray) else extr_vect * np.ones(n_ts)
                    for asset_name, extr_vect in bound_values.items()}
    return bound_values


def stack_per_storage_bound_as_xarray(per_gen_bound: Dict[str, np.ndarray], snapshots,
                                      storage_dim: str = "StorageUnit") -> xr.DataArray:
    return xr.DataArray(
        np.column_stack([bound_vect for pu_name, bound_vect in per_gen_bound.items()]),
        dims=("snapshot", storage_dim),
        coords={
            "snapshot": snapshots,
            storage_dim: list(per_gen_bound),
        },
    )


def set_sum_of_prod_const_name(const_type: str, temporal_granularity: str, direction: str) -> str:
    # remove prefix (min/max) from constraint type
    for prefix in ["min_", "max_"]:
        if const_type.startswith(prefix):
            const_type = const_type[len(prefix):]

    return f"{const_type}_{temporal_granularity}_{direction}"


def simplify_lp_elt_mapping(mapping: dict[str, str], snapshots: pd.DatetimeIndex, elt_type: str,
                            shorten_asset_name: bool = True) -> dict[str, str]:
    """
    Convert PyPSA variable names such as

        Generator-p[2033-01-01T00:00:00.000000000, fra_nuclear]

    into

        prod_fra_nuclear_1

    where:
        - 'p' is mapped to 'prod'
        - the timestamp is replaced by its 1-based index in snapshots

    Parameters
    ----------
    mapping : dict[str, str]
        Mapping returned by get_lp_variable_mapping().

    elt_type : variable, or constraint

    snapshots : pd.DatetimeIndex
        Reference date range used to convert timestamps into
        1-based time-step indices.

    shorten_asset_name : use reduced names for the assets?

    Returns
    -------
    dict[str, str]
        Mapping of LP variable names to simplified names.
    """
    allowed_elt_types = ["variable", "constraint"]
    if elt_type not in allowed_elt_types:
        raise Exception(f"Element type for mapping simplification (in LP) {elt_type} not in list of allowed "
                        f"ones {allowed_elt_types}")

    # rename some variables
    timestamp_to_idx = {pd.Timestamp(ts): i for i, ts in enumerate(snapshots)}

    simplified = {}

    if elt_type == "variable":
        pattern = re.compile(
            r"(?P<component>[^-]+)-(?P<elt>[^\[]+)"
            r"\[(?P<timestamp>[^,]+),\s*(?P<asset>[^\]]+)\]"
        )
    else:  # constraint
        pattern = re.compile(
            r"(?P<component>[^-]+)-(?P<elt>[^\[]+)"
            r"\[(?P<coords>[^\]]+)\]"
        )

    for lp_name, pypsa_name in mapping.items():

        match = pattern.fullmatch(pypsa_name)

        if match is None:
            simplified[lp_name] = pypsa_name
            continue

        if elt_type == "variable":
            timestamp = pd.Timestamp(match.group("timestamp"))
            asset = match.group("asset")
        else:
            # TODO: see if both regex cannot be merged in fact...
            const_coords = match.group("coords").split(", ")
            const_coords = [elt_coord.strip() for elt_coord in const_coords]
            # depending on the constraints, dims are not ordered the same way...
            if is_country_trigram(my_str=const_coords[0]):
                asset_idx = 0
                timestamp_idx = 1
            else:
                timestamp_idx = 0
                asset_idx = 1
            timestamp = pd.Timestamp(const_coords[timestamp_idx])
            # can be a prod. asset name, a zone (SDE) or a couple of zone for link constraints
            asset = const_coords[asset_idx]

        if shorten_asset_name:
            # Specific case of links
            o_and_d_bus = get_bus_from_link_name(link_name=asset)
            if o_and_d_bus is not None:
                asset = "-".join(o_and_d_bus)
            else:
                # In specific case when asset is a (unique) bus it will return this bus name
                bus_name = get_country_from_unit_name(prod_unit_name=asset)
                prod_type = get_prod_type_from_unit_name(prod_unit_name=asset)
                asset = "_".join([bus_name, PROD_TYPE_IN_LP.get(prod_type, prod_type)])

        full_elt = match.group("component") + "-" + match.group("elt")

        explicit_corresp = PYPSA_TO_EXPLICIT_VAR_NAMES if elt_type == "variable" else PYPSA_TO_EXPLICIT_CONST_NAMES
        elt = explicit_corresp.get(full_elt, full_elt)

        time_idx = timestamp_to_idx[timestamp]

        simplified_name = f"{elt}_{asset}_{time_idx}"

        simplified[lp_name] = simplified_name

    return simplified


@dataclass
class PypsaModel:
    # TODO: json dump to have an aggreg. view of such a model in saved files (and check stress test effect rapidly)
    name: str
    network: pypsa.Network = None
    uc_summary_metrics: UCSummaryMetrics = None  # UC summary metrics (ENS, nber of failure hours, costs...)
    optim_solver_params: SolverParams = None
    DEFAULT_CARRIER = 'ac'

    def init_pypsa_network(self, date_idx: pd.Index, date_range: pd.DatetimeIndex = None):
        logging.info('Initialize PyPSA network')
        self.network = pypsa.Network(name=self.name, snapshots=date_idx)
        if date_range is not None:
            self.network.set_snapshots(date_range[:-1])

    def add_gps_coordinates(self, countries_gps_coords: Dict[str, Tuple[float, float]], carrier_name: str = None):
        if carrier_name is None:
            carrier_name = self.DEFAULT_CARRIER

        logging.info('Add GPS coordinates')
        for country, gps_coords in countries_gps_coords.items():
            country_bus_name = get_country_bus_name(country=country)
            self.network.add(GEN_UNITS_PYPSA_PARAMS.bus.capitalize(), name=f'{country_bus_name}',
                             x=gps_coords[0], y=gps_coords[1], carrier=carrier_name)

    def add_energy_carriers(self, fuel_sources: Dict[str, FuelSource]):
        logging.info('Add energy carriers')
        for carrier in list(fuel_sources.keys()):
            self.network.add(GEN_UNITS_PYPSA_PARAMS.carrier.capitalize(), name=carrier,
                             co2_emissions=fuel_sources[carrier].co2_emissions / 1000)

    def add_per_bus_energy_carriers(self, fuel_sources: Dict[str, FuelSource], carrier_name: str = None):
        if carrier_name is None:
            carrier_name = self.DEFAULT_CARRIER
        all_bus_names = self.get_bus_names()
        logging.info(f'Add per-bus energy carriers for: {all_bus_names}')
        for bus_name in all_bus_names:
            self.network.add(GEN_UNITS_PYPSA_PARAMS.carrier.capitalize(), name=bus_name,
                             co2_emissions=fuel_sources[carrier_name].co2_emissions / 1000)

    def add_generators(self, generators_data: Dict[str, List[GenerationUnitData]],
                       agg_info_over_cases: bool = True):
        logging.info('Add generators - associated to their respective buses')
        gen_units_with_default_init_soc = []
        default_init_soc_pu = 0.8
        for country, gen_units_data in generators_data.items():
            country_bus_name = get_country_bus_name(country=country)
            for gen_unit_data in gen_units_data:
                pypsa_gen_unit_dict = gen_unit_data.__dict__
                # remove elements with None values, as all attrs were listed in this dict.
                pypsa_gen_unit_dict = rm_elts_with_none_val(my_dict=pypsa_gen_unit_dict)
                logging.debug(f'{country}, {pypsa_gen_unit_dict}')
                params_ok = check_gen_unit_params(params=pypsa_gen_unit_dict, n_ts=len(self.network.snapshots))
                if not params_ok:
                    logging.warning(f'Pb with generator parameters {pypsa_gen_unit_dict} '
                                    f'\n-> generator not added to the PyPSA model')
                    continue

                # case of storage units, identified via the presence of max_hours param
                if pypsa_gen_unit_dict.get(GEN_UNITS_PYPSA_PARAMS.max_hours, None) is not None:
                    if pypsa_gen_unit_dict.get(GEN_UNITS_PYPSA_PARAMS.soc_init, None) is None:
                        # initial SoC fixed to 80% statically here
                        if agg_info_over_cases:
                            gen_units_with_default_init_soc.append(pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.name])
                        else:
                            logging.info(f'Default value set for {pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.name]} '
                                         f'init. SOC as {int(100 * default_init_soc_pu)}% of energy storage capa.')
                        init_soc = (pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.power_capa]
                                    * pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.max_hours] * 0.8)
                        pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.soc_init] = init_soc
                    self.network.add('StorageUnit', bus=f'{country_bus_name}', **pypsa_gen_unit_dict)
                else:
                    self.network.add('Generator', bus=f'{country_bus_name}', **pypsa_gen_unit_dict)
        generator_names = self.get_generator_names()
        n_gen_with_default_init_soc = len(gen_units_with_default_init_soc)
        if agg_info_over_cases and n_gen_with_default_init_soc > 0:
            logging.info(f'Default value set for init. SOC as {int(100 * default_init_soc_pu)}% of energy storage '
                         f'capa. for generators ({n_gen_with_default_init_soc}):'
                         f'{set_per_bus_asset_msg(asset_names=gen_units_with_default_init_soc)}')

        logging.info(f'Considered generators ({len(generator_names)}): '
                     f'{set_per_bus_asset_msg(asset_names=generator_names)}')
        storage_unit_names = self.get_storage_unit_names()
        logging.info(f'Considered storage units ({len(storage_unit_names)}): '
                     f'{set_per_bus_asset_msg(asset_names=storage_unit_names)}')

    def add_loads(self, demand: Dict[str, pd.DataFrame], carrier_name: str = None):
        if carrier_name is None:
            carrier_name = self.DEFAULT_CARRIER
        logging.info('Add loads - associated to their respective buses')
        for country in demand:
            country_bus_name = get_country_bus_name(country=country)
            load_data = {GEN_UNITS_PYPSA_PARAMS.name: f'{country_bus_name}-load',
                         GEN_UNITS_PYPSA_PARAMS.bus: f'{country_bus_name}',
                         GEN_UNITS_PYPSA_PARAMS.carrier: carrier_name,
                         GEN_UNITS_PYPSA_PARAMS.set_power: demand[country]['value'].values}
            self.network.add('Load', **load_data)

    def add_interco_links(self, countries: List[str], interco_capas: Dict[Tuple[str, str], float],
                          carrier_name: str = None):
        if carrier_name is None:
            carrier_name = self.DEFAULT_CARRIER

        logging.info(f'Add interco. links - between the selected countries: {countries}')
        links = []
        symmetric_links = []
        links_wo_capa_msg = []
        for country_origin, country_dest in product(countries, countries):
            link_tuple = (country_origin, country_dest)
            # do not add link for (country, country); neither for symmetric links already treated 
            # (as bidirectional setting p_min_pu=-1)
            if not country_origin == country_dest and link_tuple not in symmetric_links:
                # TODO: fix AC/DC.... all AC here in names but not true (cf. CS students data)
                current_interco_capa, is_sym_interco = \
                    get_current_interco_capa(interco_capas=interco_capas, country_origin=country_origin,
                                             country_dest=country_dest)
                if current_interco_capa is None:
                    # if symmetrical interco order lexicographically to fit with input data format
                    if is_sym_interco:
                        link_wo_capa = lexico_compar_str(string1=country_origin,
                                                         string2=country_dest, return_tuple=True)
                    else:
                        link_wo_capa = link_tuple
                    link_wo_capa_msg = f'({link_wo_capa[0]}, {link_wo_capa[1]})'
                    if link_wo_capa_msg not in links_wo_capa_msg:
                        links_wo_capa_msg.append(f'({link_wo_capa[0]}, {link_wo_capa[1]})')
                else:
                    country_origin_bus_name = get_country_bus_name(country=country_origin)
                    country_dest_bus_name = get_country_bus_name(country=country_dest)
                    if is_sym_interco:
                        p_min_pu, p_max_pu = -1, 1
                        symmetric_links.append(link_tuple)
                    else:
                        p_min_pu, p_max_pu = 0, 1
                    links.append({GEN_UNITS_PYPSA_PARAMS.name:
                                      set_link_name(origin_bus_name=country_origin_bus_name,
                                                    dest_bus_name=country_dest_bus_name, carrier_name=carrier_name),
                                  f'{GEN_UNITS_PYPSA_PARAMS.bus}0': country_origin_bus_name,
                                  f'{GEN_UNITS_PYPSA_PARAMS.bus}1': country_dest_bus_name,
                                  GEN_UNITS_PYPSA_PARAMS.nominal_power: current_interco_capa,
                                  GEN_UNITS_PYPSA_PARAMS.min_power_pu: p_min_pu,
                                  GEN_UNITS_PYPSA_PARAMS.max_power_pu: p_max_pu,
                                  GEN_UNITS_PYPSA_PARAMS.carrier: carrier_name}
                                 )
        if len(links_wo_capa_msg) > 0:
            print_errors_list(error_name='-> interco. links without capacity data', errors_list=links_wo_capa_msg)

        # add to PyPSA network
        for link in links:
            if link[GEN_UNITS_PYPSA_PARAMS.power_capa] > 0:
                self.network.add('Link', **link)
        link_names = self.get_link_names()
        logging.info(f'Considered links - the ones with nonzero capacity ({len(link_names)}), in alphabetic order '
                     f'of origin: {set_per_origin_bus_links_msg(link_names=link_names)}')

    def build_model_before_adding_custom_const(self):
        logging.info('Build PyPSA network model before adding custom constraints')
        # In PyPSA 1.0+, build_only=True works natively without solving.
        # We attempt with quiet/logfile options first, then fallback to standard build_only.
        try:
            self.network.optimize(build_only=True, solver_options={'logfile': '/dev/null'})
        except Exception:
            self.network.optimize(build_only=True)

    def add_sum_of_prod_custom_const(self, prod_sum_const: ZoneAndTempProdSumConstraint):
        """
        Add sum-of-production custom constraints, of the form sum_{z, t} coeff(z, t) * production(z, t) <= ub (or >=, =)
        N.B. Can be applied to CO2 max emission constraints ; max production cost
        :param prod_sum_const: object with attrs
            - type: name of the type of constraint, e.g. co2_emis
            - direction: either upper (zone-and-temp sum prod. <= bound) or lower (idem >= bound)
            or equal (idem = bound)
            - mult_coeff_name: name of the multiplicative coeffs to be used, in {'co2_emis_factor', 'variable_cost'}
            - temporal_granularity: associated to this constraint
            - countries: over which constraints is to be imposed
            - bound: np.ndarray, one value per period
            - (optional) name: of the constraint. Default: None
        """
        # TODO: make this method functional in the case of monthly const.
        const_name = set_sum_of_prod_const_name(const_type=prod_sum_const.type,
                                                temporal_granularity=prod_sum_const.temporal_granularity,
                                                direction=prod_sum_const.direction)
        logging.info(f'Add custom sum of prod constraints {const_name}')
        # countries selected for this constraint - using bus names (country trigrams) in PyPSA
        selec_countries = [get_country_bus_name(country=elt) for elt in prod_sum_const.countries]
        prod_var = self.network.model.variables[PypsaOptimVarNames.generators_p]

        # In PyPSA 1.0+, generator variable dimensions in Linopy are ('snapshot', 'name')
        # rather than ('snapshot', 'Generator'). Dynamically extract the non-snapshot dimension name.
        gen_dim = [d for d in prod_var.dims if d != 'snapshot'][0]

        # Identify generators belonging to selected countries and present in the Linopy model
        selec_gens = self.network.generators.index[self.network.generators.bus.isin(selec_countries)]
        selec_gens = [g for g in selec_gens if g in prod_var.coords[gen_dim].values]
        if len(selec_gens) == 0:
            logging.info(f"No matching generators for custom constraint {const_name}")
            return

        if prod_sum_const.type == CustomConstraintNames.max_co2_emissions:
            factor = (
                self.network.generators.carrier
                .map(self.network.carriers.co2_emissions)
                .fillna(0.0)
            )
        elif prod_sum_const.type == CustomConstraintNames.max_prod_cost:
            factor = self.network.generators.marginal_cost

        # Align factor series with gen_dim so xarray / Linopy matches coordinates properly
        factor_sel = factor.loc[selec_gens]
        factor_sel.index.name = gen_dim
        prod_var_sel = prod_var.sel({gen_dim: selec_gens})

        # Ensure snapshot_weights index is named 'snapshot' for Linopy time-axis alignment
        snapshot_weights = self.network.snapshot_weightings.generators  # time-slots (snapshots) weights
        if snapshot_weights.index.name != 'snapshot':
            snapshot_weights = snapshot_weights.copy()
            snapshot_weights.index.name = 'snapshot'

        # Sum over the component dimension dynamically (gen_dim) to obtain a 1D expression over snapshots
        expr = ((prod_var_sel
                 * factor_sel
                 * snapshot_weights)
                .sum(gen_dim)
                )

        # temporal aggregation/sum -> over whole period
        if prod_sum_const.temporal_granularity == Timescale.whole_period:
            lhs = expr.sum()
            rhs = prod_sum_const.bound
            per_dir_const = {
                CustomConstraintDirection.lower: lhs >= rhs,
                CustomConstraintDirection.equal: lhs == rhs,
                CustomConstraintDirection.upper: lhs <= rhs,
            }
            self.network.model.add_constraints(per_dir_const[prod_sum_const.direction], name=const_name)
        else:  # per period constraint
            if prod_sum_const.temporal_granularity == Timescale.week:
                periods = pd.Series(
                    self.network.snapshots.to_period("W"),
                    index=self.network.snapshots,
                )
            elif prod_sum_const.temporal_granularity == Timescale.day:
                periods = pd.Series(
                    self.network.snapshots.normalize(),
                    index=self.network.snapshots,
                )

            # get start of periods dates
            period_values = list(set(periods.values))
            period_values.sort()
            if prod_sum_const.temporal_granularity in [Timescale.week, Timescale.month]:
                logging.info(f"Following {prod_sum_const.temporal_granularity}ly start-of-periods (Monday) used "
                             f"for constraints: {period_values}")

            for i, (period, snaps) in enumerate(periods.groupby(periods)):
                # Select snapshots belonging to the current period and sum temporally
                lhs = expr.sel(snapshot=snaps.index).sum()
                rhs = prod_sum_const.bound[i]
                per_dir_const = {
                    CustomConstraintDirection.lower: lhs >= rhs,
                    CustomConstraintDirection.equal: lhs == rhs,
                    CustomConstraintDirection.upper: lhs <= rhs,
                }
                self.network.model.add_constraints(per_dir_const[prod_sum_const.direction],
                                                   name=f"{const_name}_{period}")

    def add_hydro_extreme_levels_constraint(self, soc_min_pu: Dict[str, Union[float, np.ndarray]],
                                            soc_max_pu: Dict[str, Union[float, np.ndarray]],
                                            energy_capa: Dict[str, float]):
        """
        Add constraint on hydro extreme SOC levels
        :param soc_min_pu: dict {unit name: soc min vector} - with SoC min an absolute value,
        NOT AS A PERCENTAGE of capacity
        :param soc_max_pu: idem, max
        :param energy_capa: dict {unit name: energy capa value}
        """
        logging.info("Add hydro extreme SoC levels constraints")
        # preprocess min/max params data -> (i) project values on [0, capa.], to induce real constraints (not <0,
        # or bigger than energy capacity), and (ii) unify as vectors
        n_ts = self.get_n_time_slots()
        # from PU to ABSOLUTE values of min/max SoC (ABSOLUTE values have to be used in PyPSA)
        soc_min = {asset_name: energy_capa[asset_name] * bound_pu for asset_name, bound_pu in soc_min_pu.items()}
        soc_max = {asset_name: energy_capa[asset_name] * bound_pu for asset_name, bound_pu in soc_max_pu.items()}
        # project on feasible bound interval [0, capa], and vectorize
        soc_min = prepro_bound_params(bound_values=soc_min, max_feas_bound=energy_capa, n_ts=n_ts)
        soc_max = prepro_bound_params(bound_values=soc_max, max_feas_bound=energy_capa, n_ts=n_ts)
        # set associated constraints in Linopy
        # In PyPSA 1.0+, variable names follow lowercase/plural convention: 'storage_units-state_of_charge'
        soc_var_name = PypsaOptimVarNames.storage_soc
        if soc_var_name not in self.network.model.variables:
            soc_var_name = 'storage_units-state_of_charge'
        hydro_soc_var = self.network.model.variables[soc_var_name]

        # In PyPSA 1.0+, storage unit dimension is 'name' instead of 'StorageUnit'.
        # Dynamically extract the non-snapshot dimension.
        storage_dim = [d for d in hydro_soc_var.dims if d != 'snapshot'][0]

        # IMPORTANT: hydro_soc_var contains ALL storage units (batteries, PSP/STEP, hydro reservoirs).
        # soc_min/soc_max bounds are only defined for hydro reservoirs (subset of storage units).
        # We MUST filter hydro_soc_var to only hydro reservoirs; otherwise, xarray alignment fills
        # non-hydro storage units with NaN, creating invalid constraints and causing solver infeasibility.
        hydro_units = [u for u in soc_min if u in hydro_soc_var.coords[storage_dim].values]
        if len(hydro_units) == 0:
            logging.info("No matching storage units found for hydro extreme levels constraints")
            return

        soc_min = {u: soc_min[u] for u in hydro_units}
        soc_max = {u: soc_max[u] for u in hydro_units}
        hydro_soc_var_sel = hydro_soc_var.sel({storage_dim: hydro_units})

        # cast dict of bound values as xarray - format to be used in Linopy
        soc_min_array = stack_per_storage_bound_as_xarray(per_gen_bound=soc_min, snapshots=self.network.snapshots,
                                                          storage_dim=storage_dim)
        soc_max_array = stack_per_storage_bound_as_xarray(per_gen_bound=soc_max, snapshots=self.network.snapshots,
                                                          storage_dim=storage_dim)
        self.network.model.add_constraints(hydro_soc_var_sel >= soc_min_array, name=CustomConstraintNames.hydro_soc_min)
        self.network.model.add_constraints(hydro_soc_var_sel <= soc_max_array, name=CustomConstraintNames.hydro_soc_max)

    def add_hydro_extreme_gen_constraint(self, generation_min: Dict[str, np.ndarray],
                                         generation_max: Dict[str, np.ndarray], power_capa: Dict[str, float],
                                         extr_gen_temp_period: str = Timescale.week):
        """
        :param generation_min: per asset vector/value with min generation over each of/all the periods
        :param generation_max: idem
        :param power_capa: of the assets, to check/make the bound feasible
        :param extr_gen_temp_period: either day, or week
        """
        logging.info("Add hydro extreme GENERATION levels constraints")
        # preprocess min/max params data -> (i) project values on [0, capa.], to induce real constraints (not <0,
        # or bigger than power capacity * nber of ts in considered period), and (ii) unify as vectors
        n_ts = self.get_n_time_slots()
        n_ts_in_period = 168 if extr_gen_temp_period == Timescale.week else 24
        max_feas_bound = {pu_name: capa * n_ts_in_period for pu_name, capa in power_capa.items()}
        generation_min = prepro_bound_params(bound_values=generation_min, max_feas_bound=max_feas_bound, n_ts=n_ts)
        generation_max = prepro_bound_params(bound_values=generation_max, max_feas_bound=max_feas_bound, n_ts=n_ts)

        # Generator production constraints, NOT CONSIDERING charging part
        # In PyPSA 1.0+, dispatch variable name is 'storage_units-p_dispatch'
        gen_var_name = PypsaOptimVarNames.storage_p_dispatch
        if gen_var_name not in self.network.model.variables:
            gen_var_name = 'storage_units-p_dispatch'
        hydro_gen_var = self.network.model.variables[gen_var_name]

        # Identify storage unit dimension dynamically and filter for hydro reservoirs only
        storage_dim = [d for d in hydro_gen_var.dims if d != 'snapshot'][0]
        hydro_units = [u for u in generation_min if u in hydro_gen_var.coords[storage_dim].values]
        if len(hydro_units) == 0:
            logging.info("No matching storage units found for hydro extreme generation constraints")
            return

        generation_min = {u: generation_min[u] for u in hydro_units}
        generation_max = {u: generation_max[u] for u in hydro_units}
        hydro_gen_var_sel = hydro_gen_var.sel({storage_dim: hydro_units})

        # IMPORTANT: ERAA min/max hydro generation limits represent total energy over a week/day (MWh),
        # not hourly instantaneous power limits (MW). Imposing weekly MWh volume on hourly dispatch
        # caused massive infeasibility. We group snapshots by period and sum hourly dispatch (.sum("snapshot")),
        # scaling bounds by dur_ratio = len(snaps) / n_ts_in_period to handle partial first/last periods.
        if extr_gen_temp_period in [Timescale.week, Timescale.day]:
            if extr_gen_temp_period == Timescale.week:
                periods = pd.Series(
                    self.network.snapshots.to_period("W"),
                    index=self.network.snapshots,
                )
            else:
                periods = pd.Series(
                    self.network.snapshots.normalize(),
                    index=self.network.snapshots,
                )

            for i, (period, snaps) in enumerate(periods.groupby(periods)):
                # Sum hourly production over all snapshots in this period (MWh)
                gen_sum = hydro_gen_var_sel.sel(snapshot=snaps.index).sum("snapshot")
                t_start_idx = self.network.snapshots.get_loc(snaps.index[0])
                dur_ratio = len(snaps) / n_ts_in_period
                rhs_min = xr.DataArray(
                    [generation_min[u][t_start_idx] * dur_ratio for u in hydro_units],
                    dims=(storage_dim,),
                    coords={storage_dim: hydro_units},
                )
                rhs_max = xr.DataArray(
                    [generation_max[u][t_start_idx] * dur_ratio for u in hydro_units],
                    dims=(storage_dim,),
                    coords={storage_dim: hydro_units},
                )
                self.network.model.add_constraints(gen_sum >= rhs_min, name=f"hydro_gen_min_{period}")
                self.network.model.add_constraints(gen_sum <= rhs_max, name=f"hydro_gen_max_{period}")
        else:
            generation_min_array = stack_per_storage_bound_as_xarray(per_gen_bound=generation_min,
                                                                     snapshots=self.network.snapshots,
                                                                     storage_dim=storage_dim)
            generation_max_array = stack_per_storage_bound_as_xarray(per_gen_bound=generation_max,
                                                                     snapshots=self.network.snapshots,
                                                                     storage_dim=storage_dim)
            self.network.model.add_constraints(hydro_gen_var_sel >= generation_min_array, name="hydro_gen_min")
            self.network.model.add_constraints(hydro_gen_var_sel <= generation_max_array, name="hydro_gen_max")

    def get_n_time_slots(self) -> int:
        return len(self.network.snapshots)

    def get_bus_names(self) -> List[str]:
        return list(set(self.network.buses.index))

    def get_generator_names(self) -> List[str]:
        return list(self.network.generators.index)

    def get_storage_unit_names(self) -> List[str]:
        return list(self.network.storage_units.index)

    def get_link_names(self, only_links_with_nonzero_capa: bool = True, rm_carrier_name: bool = True,
                       lexico_sort: bool = True) -> List[str]:
        df_links = self.network.links
        if only_links_with_nonzero_capa:
            df_links = df_links[df_links[GEN_UNITS_PYPSA_PARAMS.nominal_power] > 0]
        link_names = list(df_links.index)
        if rm_carrier_name:
            link_names = [rm_elts_in_str(my_str=full_link_name, elts_tb_removed=['_ac', '_dc'])
                          for full_link_name in link_names]
        if lexico_sort:
            link_names = sort_lexicographically(strings=link_names)
        return link_names

    def get_per_bus_total_installed_capa(self):
        bus_names = self.get_bus_names()
        df_generators = self.network.generators
        return {name: df_generators.loc[(df_generators.index.str.startswith(f'{name}_'))
                                        & (df_generators['type'] != ProdTypeNames.failure), 'p_nom'].sum()
                for name in bus_names}

    def get_per_bus_max_load(self) -> Dict[str, float]:
        bus_names = self.get_bus_names()
        return {name: max(self.network.loads_t['p_set'][f'{name}-load']) for name in bus_names}

    def plot_network(self, toy_model_output: bool = False, country: str = None, output_tag: str = None):
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.network.plot(title=f'{self.name.capitalize()} PyPSA network', color_geomap=True, jitter=0.3)
            except TypeError:
                # 'color_geomap' was removed/deprecated in PyPSA 1.0+
                self.network.plot(title=f'{self.name.capitalize()} PyPSA network', jitter=0.3)
            plt.savefig(get_network_figure(toy_model_output=toy_model_output, country=country,
                                           n_bus=len(self.network.buses), output_tag=output_tag)
                        )
            plt.close()

    def set_default_optim_solver(self, warning_msg: str):
        msg_default_solver_used = f'-> default {DEFAULT_OPTIM_SOLVER_PARAMS.name} will be used instead'
        logging.warning(f'{warning_msg} {msg_default_solver_used}')
        self.optim_solver_params = DEFAULT_OPTIM_SOLVER_PARAMS

    def set_optim_solver(self, solver_params: SolverParams = None):
        # if no solver provided in arg. -> set default one
        if solver_params is None:
            self.optim_solver_params = DEFAULT_OPTIM_SOLVER_PARAMS
        # else check if coherent parameters; otherwise set default solver
        else:
            all_solver_names = OptimSolvers.__dict__.values()
            solver_name = solver_params.name
            if solver_name not in all_solver_names:
                warning_msg = f'Solver name {solver_name} not in allowed list {all_solver_names}'
                self.set_default_optim_solver(warning_msg=warning_msg)
            else:
                self.optim_solver_params = solver_params
            if not self.optim_solver_params.name == DEFAULT_OPTIM_SOLVER_PARAMS.name:
                # check that license file param is defined
                solver_license_file = self.optim_solver_params.license_file
                if solver_license_file is None:
                    warning_msg = f'Licence file for optim. solver {self.optim_solver_params.name} not provided'
                    self.set_default_optim_solver(warning_msg=warning_msg)
                else:
                    # license file must be at root of the project
                    if not os.path.exists(path=solver_license_file):
                        warning_msg = f'Licence file {solver_license_file} does not exist (at root of project)'
                        self.set_default_optim_solver(warning_msg=warning_msg)
                    else:
                        os.environ[f'{self.optim_solver_params.name.upper()}_LICENSE_FILE'] = solver_license_file

    def get_optim_pb_characteristics(self) -> OptimPbCharacteristics:
        """
        N.B. (i) This method can be called only after the network model has been optimized or built
        (ii) network.model.constraints contains per type of constraint info (dimensions and size)
        """
        linopy_model = self.network.model
        return OptimPbCharacteristics(type=set_optim_pb_type(model=linopy_model),
                                      n_variables=len(linopy_model.variables.flat),
                                      n_int_variables=len(linopy_model.integers),
                                      n_constraints=len(linopy_model.constraints.flat))

    def optimize_network(self, year: int, n_countries: int, period_start: datetime, save_lp_file: bool = True,
                         save_explicit_vars_corresp: bool = False, save_explicit_consts_corresp: bool = False,
                         toy_model_output: bool = False, countries: List[str] = None, output_tag: str = None,
                         save_lp_with_explicit_names: bool = False) \
            -> PYPSA_RESULT_TYPE:
        """
        Solve the optimization UC problem associated to current network
        :param year: of the considered UC problem
        :param n_countries: in considered UC problem
        :param period_start
        :param save_lp_file: to get a standard saving of optim. model (that can be re-used with solvers outside of this
        environment)
        :param save_explicit_vars_corresp: save corresp. between xi variable names in PyPSA and explicit ones?
        :param save_explicit_consts_corresp: idem, for constraint names
        :param toy_model_output
        :param countries
        :param output_tag: used for output folder
        :param save_lp_with_explicit_names: save lp with explicit variable and/or constraint names?
        :returns a tuple (xxx, status of resolution)
        """
        logging.info('Optimise "network" - i.e. solve associated UC problem')
        result = self.network.optimize.solve_model(solver_name=self.optim_solver_params.name)
        logging.info(f'Obtained result: {result}')
        if save_lp_file:
            lp_filepath = save_lp_model(self.network, year=year, n_countries=n_countries, period_start=period_start,
                                        toy_model_output=toy_model_output, countries=countries, output_tag=output_tag)
            if save_explicit_vars_corresp:
                logging.info(f"Save correspondence between PyPSA abstract variable names and explicit ones")
                lp_var_mapping = self.get_lp_variable_mapping()
                df_var_mapping = pd.DataFrame(lp_var_mapping.items(), columns=["pypsa", "explicit"])
                df_var_mapping.to_csv(lp_filepath.replace(".lp", f"_var-mapping.csv"), index=False)
            if save_explicit_consts_corresp:
                logging.info(f"Save correspondence between PyPSA abstract constraint names and explicit ones")
                lp_const_mapping = self.get_lp_constraint_mapping()
                df_const_mapping = pd.DataFrame(lp_const_mapping.items(), columns=["pypsa", "explicit"])
                df_const_mapping.to_csv(lp_filepath.replace(".lp", f"_const-mapping.csv"), index=False)
            if save_lp_with_explicit_names and (lp_var_mapping is not None or lp_const_mapping is not None):
                rename_elts_in_lp_model(lp_filepath=lp_filepath, var_readable_mapping=lp_var_mapping,
                                        const_readable_mapping=lp_const_mapping)

        return result

    def set_uc_opt_solution(self) -> UCOptimalSolution:
        """
        Returns: an object containing variables + methods on the UC optimal solution
        """
        # init.
        uc_opt_solution = UCOptimalSolution(network_name=self.network.name)
        # get primal optimal values from PyPSA network
        uc_opt_solution.get_prod_var_opt(network=self.network)
        uc_opt_solution.get_storage_vars_opt(network=self.network)
        uc_opt_solution.get_link_flow_vars_opt(network=self.network)
        # and dual variables - some of them "entering" into Linopy framework
        uc_opt_solution.get_sde_dual_var_opt(network=self.network)
        uc_opt_solution.get_link_capa_dual_var_opt(network=self.network)
        return uc_opt_solution

    def get_opt_value(self, pypsa_resol_status: str) -> float:
        objective_value = get_network_obj_value(network=self.network)
        objective_value_refmted = format_with_spaces(number=int(objective_value / 1e6))
        logging.info(
            f'Optimisation resolution status is {pypsa_resol_status} with objective value (cost) = '
            f'{objective_value_refmted} (M€) -> output data (resp. figures) can be generated')
        return objective_value

    def get_constraint_bound_value(self, constraint_name: str) -> Optional[xarray.DataArray]:
        if constraint_name in self.network.model.constraints:
            return self.network.model.constraints[constraint_name].rhs
        else:
            logging.warning(f"Bound cannot be obtained for constraint {constraint_name}, "
                            f"not present in current PyPSA model")
            return None

    def plot_installed_capas(self, country: str, year: int, toy_model_output: bool = False, output_tag: str = None):
        country_trigram = set_country_trigram(country=country)
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # N.B. p_nom_opt is the optimized capacity (that can be also a variable in PyPSA but here...
            # not optimized - only UC problem -> values plotted correspond to the ones that can be found in input data)
            # all but failure asset capacity will be used in plot
            self.network.generators.p_nom_opt.drop(f'{country_trigram}_failure').div(1e3).plot.bar(ylabel='GW',
                                                                                                   figsize=(8, 3))
            plt.tight_layout()
            plt.savefig(get_output_figure(fig_name=FigNamesPrefix.capacity, country=country, year=year,
                                          toy_model_output=toy_model_output, output_tag=output_tag))
            plt.close()

    import numpy as np

    def get_lp_variable_mapping(self, simplify_mapping: bool = True) -> dict[str, str]:
        """
        Build a mapping between LP variable names (x1, x2, ...)
        and their corresponding PyPSA/Linopy variable names.
        Returns
        -------
        dict[str, str]
            Dictionary of the form

                {
                    "x1": "Generator-p[Gen1,2025-01-01 00:00:00]",
                    "x2": "Generator-p[Gen1,2025-01-01 01:00:00]",
                    ...
                }
        """

        mapping = {}

        for var_name, var in self.network.model.variables.items():

            labels = np.asarray(var.labels).ravel()

            # Build a coordinate array with the same shape as labels
            coord_arrays = np.meshgrid(
                *[var.coords[dim].values for dim in var.dims],
                indexing="ij",
            )

            flat_coords = list(zip(*(c.ravel() for c in coord_arrays)))

            for label, coords in zip(labels, flat_coords):

                # Negative labels correspond to empty positions
                if label < 0:
                    continue

                coord_str = ", ".join(str(c) for c in coords)

                mapping[f"x{label}"] = f"{var_name}[{coord_str}]"

        if simplify_mapping:
            mapping = simplify_lp_elt_mapping(mapping=mapping, elt_type="variable", snapshots=self.network.snapshots)

        return mapping

    def get_lp_constraint_mapping(self, simplify_mapping: bool = True) -> dict[str, str]:
        """
        Build a mapping between LP constraint names (c1, c2, ...)
        and their corresponding PyPSA/Linopy constraint names.

        Returns
        -------
        dict[str, str]
            Dictionary of the form

                {
                    "c1": "Generator-fix-p-upper[...]",
                    "c2": "Generator-fix-p-upper[...]",
                    ...
                }
        """

        mapping = {}

        for constraint_name, constraint in self.network.model.constraints.items():

            labels = np.asarray(constraint.labels).ravel()

            coord_arrays = np.meshgrid(
                *[constraint.coords[dim].values for dim in constraint.dims],
                indexing="ij",
            )

            flat_coords = list(zip(*(c.ravel() for c in coord_arrays)))

            for label, coords in zip(labels, flat_coords):

                if label < 0:
                    continue

                coord_str = ", ".join(str(c) for c in coords)

                mapping[f"c{label}"] = (
                    f"{constraint_name}[{coord_str}]"
                )

        if simplify_mapping:
            mapping = simplify_lp_elt_mapping(mapping=mapping, elt_type="constraint", snapshots=self.network.snapshots)

        return mapping

# def overwrite_gen_units_fuel_src_params(generation_units_data: GEN_UNITS_DATA_TYPE, updated_fuel_sources_params: Dict[
#     str, Dict[str, float]]) -> GEN_UNITS_DATA_TYPE:
#     for _, units_data in generation_units_data.items():
#         # loop over all units in current country
#         for indiv_unit_data in units_data:
#             current_prod_type = get_prod_type_from_unit_name(prod_unit_name=indiv_unit_data.name)
#             if current_prod_type in updated_fuel_sources_params:
#                 # TODO: add CO2 emissions, and merge both case? Q2OJ: how-to properly?
#                 if GEN_UNITS_PYPSA_PARAMS.marginal_cost in updated_fuel_sources_params[current_prod_type]:
#                     indiv_unit_data.marginal_cost = updated_fuel_sources_params[current_prod_type][
#                         GEN_UNITS_PYPSA_PARAMS.marginal_cost]
#
#         # TODO: from units data info on fuel source extract and apply updated params values
#         updated_fuel_sources_params = None


def get_country_bus_name(country: str) -> str:
    return country.lower()[:3]


STORAGE_LIKE_UNITS = ['batteries', 'flexibility', 'hydro']


# TODO: suppr?
# def add_loads(network, demand: Dict[str, pd.DataFrame]):
#     print("Add loads - associated to their respective buses")
#     for country in demand:
#         country_bus_name = get_country_bus_name(country=country)
#         load_data = {"name": f"{country_bus_name}-load", "bus": f"{country_bus_name}",
#                      "carrier": "AC", "p_set": demand[country]["value"].values}
#         network.add("Load", **load_data)
#     return network


def get_current_interco_capa(interco_capas: Dict[Tuple[str, str], float], country_origin: str,
                             country_dest: str) -> Tuple[Optional[float], Optional[bool]]:
    link_tuple = (country_origin, country_dest)
    reverse_link_tuple = (country_dest, country_origin)
    if link_tuple in interco_capas:
        current_interco_capa = interco_capas[link_tuple]
        is_sym_interco = reverse_link_tuple not in interco_capas
    elif reverse_link_tuple in interco_capas:
        current_interco_capa = interco_capas[reverse_link_tuple]
        is_sym_interco = True
    else:
        current_interco_capa = None
        is_sym_interco = None
    return current_interco_capa, is_sym_interco


def set_period_start_file(year: int, period_start: datetime) -> str:
    return datetime(year=year, month=period_start.month, day=period_start.day).strftime('%Y-%m-%d')


def save_lp_model(network: pypsa.Network, year: int, period_start: datetime, countries: List[str] = None,
                  n_countries: int = None, add_random_suffix: bool = False, toy_model_output: bool = False,
                  output_tag: str = None) -> str:
    from src.common.long_term_uc_io import set_full_lt_uc_output_folder, OutputFolderNames

    # In PyPSA 1.0+, opt.create_model is removed. Instead we use the built-in linopy model.
    if hasattr(network, "model") and network.model is not None:
        m = network.model
    else:
        try:
            import pypsa.optimization as opt
            m = opt.create_model(network)
        except (ImportError, AttributeError):
            # Fallback for PyPSA 1.0+ if model is not built yet
            network.optimize(build_only=True)
            m = network.model

    # set prefix
    n_countries_max_in_prefix = 3
    if countries is not None:
        if len(countries) <= n_countries_max_in_prefix:
            prefix = '-'.join(countries)
            n_countries = None
        else:
            n_countries = len(countries)
    if n_countries is not None:
        prefix = '1-country' if n_countries == 1 else f'{n_countries}-countries'

    # to avoid suppressing previous runs results
    if add_random_suffix:
        run_id = np.random.randint(99)
        random_suffix = f'_{run_id}'
    else:
        random_suffix = ''

    period_start_file = set_period_start_file(year=year, period_start=period_start)
    file_suffix = f'{prefix}_{period_start_file}{random_suffix}'
    # if more than 1 country lp will be saved in a europe output folder (not monozone_{country})
    country_output_folder = countries[0] if countries is not None and len(countries) == 1 else None
    output_folder_data = set_full_lt_uc_output_folder(folder_type=OutputFolderNames.data, country=country_output_folder,
                                                      toy_model_output=toy_model_output, output_tag=output_tag)
    make_dir(full_path=output_folder_data)
    lp_filepath = f'{output_folder_data}/model_{file_suffix}.lp'
    logging.info(f'Save model in .lp file: {lp_filepath}')
    m.to_file(Path(lp_filepath))
    return lp_filepath


def rename_elts_in_lp_model(lp_filepath: str, var_readable_mapping: dict[str, str] = None,
                            const_readable_mapping: dict[str, str] = None):
    renamed_elts = []
    if var_readable_mapping is not None:
        renamed_elts.append("variables")
    if const_readable_mapping is not None:
        renamed_elts.append("constraints")
    info_msg = " and ".join(renamed_elts)
    logging.info(f"Rename {info_msg} explicitly in file {os.path.split(lp_filepath)[1]}")
    with open(lp_filepath, encoding="utf-8") as f:
        content = f.read()

    mapping_dict = {"var": var_readable_mapping, "const": const_readable_mapping}
    for elt_name, readable_mapping in mapping_dict.items():
        if readable_mapping is not None:
            # Replace longest names first
            for old_name in sorted(readable_mapping, key=len, reverse=True):
                content = re.sub(
                    rf"\b{re.escape(old_name)}\b",
                    readable_mapping[old_name],
                    content,
                )

    with open(lp_filepath, "w", encoding="utf-8") as f:
        f.write(content)
