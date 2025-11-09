import os
from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datetime import datetime

from common.constants.aggreg_operations import AggregOpeNames
from common.constants.datatypes import DATATYPE_NAMES
from common.error_msgs import print_errors_list
from common.long_term_uc_io import COLUMN_NAMES, DT_FILE_PREFIX, DT_SUBFOLDERS, FILES_FORMAT, \
    GEN_CAPA_SUBDT_COLS, INPUT_CY_STRESS_TEST_SUBFOLDER, INPUT_ERAA_FOLDER
from common.uc_run_params import UCRunParams
from include.dataset_builder import GenerationUnitData, GEN_UNITS_PYPSA_PARAMS, get_val_of_agg_pt_in_df, \
    set_gen_unit_name
from utils.basic_utils import get_intersection_of_lists
from utils.df_utils import create_dict_from_cols_in_df, selec_in_df_based_on_list, set_aggreg_col_based_on_corresp
from utils.eraa_data_reader import filter_input_data, gen_capa_pt_str_sanitizer, select_interco_capas, \
    set_aggreg_cf_prod_types_data

N_SPACES_MSG = 2
PROD_TYPE_AGG_COL = f'{COLUMN_NAMES.production_type}_agg'


def get_demand_data(folder: str, file_suffix: str, climatic_year: int, period: Tuple[datetime, datetime],
                    is_stress_test: bool = False) -> pd.DataFrame:
    # get demand
    logging.debug('Get demand')
    if is_stress_test:
        demand_folder_full = f'{folder}/{INPUT_CY_STRESS_TEST_SUBFOLDER}'
    else:
        demand_folder_full = folder
    demand_file = f'{demand_folder_full}/{DT_FILE_PREFIX.demand}_{file_suffix}.csv'
    df_demand = pd.read_csv(demand_file, sep=FILES_FORMAT.column_sep, decimal=FILES_FORMAT.decimal_sep)
    # then keep only selected period date range and climatic year
    df_demand = filter_input_data(df=df_demand, date_col=COLUMN_NAMES.date,
                                  climatic_year_col=COLUMN_NAMES.climatic_year, period_start=period[0],
                                  period_end=period[1], climatic_year=climatic_year)
    return df_demand


def get_cf_agg_prod_types_tb_read(selected_agg_prod_types: List[str], agg_prod_types_with_cf_data: List[str],
                                  subdt_selec: List[str] = None) -> List[str]:
    if subdt_selec is not None:
        agg_prod_types_tb_read = get_intersection_of_lists(list1=selected_agg_prod_types, list2=subdt_selec)
    else:
        agg_prod_types_tb_read = selected_agg_prod_types
    # list of prod types with CF data
    return [agg_prod_type for agg_prod_type in agg_prod_types_tb_read if agg_prod_type in agg_prod_types_with_cf_data]


def get_res_capa_factors_data(folder: str, file_suffix: str, climatic_year: int, cf_agg_prod_types_tb_read: List[str],
                              aggreg_pt_cf_def, period: Tuple[datetime, datetime],
                              is_stress_test: bool = False) -> Optional[pd.DataFrame]:
    # TODO: type
    logging.debug('Get RES capacity factors')
    date_col = COLUMN_NAMES.date
    # full path to folder in which RES CF data can be read
    if is_stress_test:
        res_cf_folder_full = f'{folder}/{INPUT_CY_STRESS_TEST_SUBFOLDER}'
    else:
        res_cf_folder_full = folder
    # loop over the agg. production types to be read, the ones with CF data
    df_res_cf_list = []
    for agg_prod_type in cf_agg_prod_types_tb_read:
        logging.debug(N_SPACES_MSG * ' ' + f'- For aggreg. prod. type: {agg_prod_type}')
        current_agg_pt_df_res_cf_list = []
        for prod_type in aggreg_pt_cf_def[agg_prod_type]:
            cf_filename = f'{DT_FILE_PREFIX.res_capa_factors}_{prod_type}_{file_suffix}.csv'
            cf_data_file = f'{res_cf_folder_full}/{cf_filename}'
            if not os.path.exists(cf_data_file):
                logging.warning(
                    2 * N_SPACES_MSG * ' ' + f'RES capa. factor data file does not exist: '
                                             f'{prod_type} not accounted for here')
            else:
                logging.debug(2 * N_SPACES_MSG * ' ' + f'* Prod. type: {prod_type}')
                current_df_res_cf = pd.read_csv(cf_data_file, sep=FILES_FORMAT.column_sep,
                                                decimal=FILES_FORMAT.decimal_sep)
                current_df_res_cf = \
                    filter_input_data(df=current_df_res_cf, date_col=date_col,
                                      climatic_year_col=COLUMN_NAMES.climatic_year,
                                      period_start=period[0], period_end=period[1],
                                      climatic_year=climatic_year)
                if len(current_df_res_cf) == 0:
                    logging.warning(
                        2 * N_SPACES_MSG * ' ' + f'No RES capa. factor data for prod. type '
                                                 f'{prod_type} and climatic year {climatic_year}')
                else:
                    # add column with production type (for later aggreg.)
                    current_df_res_cf[PROD_TYPE_AGG_COL] = agg_prod_type
                    current_agg_pt_df_res_cf_list.append(current_df_res_cf)
        if len(current_agg_pt_df_res_cf_list) == 0:
            logging.warning(
                N_SPACES_MSG * ' ' + f'No data available for aggregate RES prod. type '
                                     f'{agg_prod_type} -> not accounted for in UC model here')
        else:
            df_res_cf_list.extend(current_agg_pt_df_res_cf_list)
    # concatenate, aggreg. over prod type of same aggreg. type and avg
    if len(df_res_cf_list) == 0:
        return None
    agg_cf_data_read = (
        set_aggreg_cf_prod_types_data(df_cf_list=df_res_cf_list, pt_agg_col=PROD_TYPE_AGG_COL,
                                      date_col=date_col, val_col=COLUMN_NAMES.value))
    return agg_cf_data_read


def get_installed_gen_capas_data(folder: str, file_suffix: str, country: str, aggreg_pt_gen_capa_def,
                                 selected_agg_prod_types: List[str]) -> Optional[pd.DataFrame]:
    # TODO: type
    # get installed generation capacity data
    logging.debug(
        'Get installed generation capacities (1 file per country and year)')
    gen_capa_data_file = f'{folder}/{DT_FILE_PREFIX.generation_capas}_{file_suffix}.csv'
    prod_type_col = COLUMN_NAMES.production_type
    if not os.path.exists(gen_capa_data_file):
        logging.warning(f'Generation capas data file does not exist: {country} not accounted for here')
        return None
    else:
        df_gen_capa = pd.read_csv(gen_capa_data_file, sep=FILES_FORMAT.column_sep, decimal=FILES_FORMAT.decimal_sep)
        # Keep sanitize prod. types col values
        df_gen_capa[prod_type_col] = df_gen_capa[prod_type_col].apply(gen_capa_pt_str_sanitizer)
        # Keep only selected aggreg. prod. types
        df_gen_capa = (
            set_aggreg_col_based_on_corresp(df=df_gen_capa, col_name=prod_type_col,
                                            created_agg_col_name=PROD_TYPE_AGG_COL, val_cols=GEN_CAPA_SUBDT_COLS,
                                            agg_corresp=aggreg_pt_gen_capa_def, common_aggreg_ope=AggregOpeNames.sum)
        )
        df_gen_capa = \
            selec_in_df_based_on_list(df=df_gen_capa, selec_col=PROD_TYPE_AGG_COL, selec_vals=selected_agg_prod_types)
        return df_gen_capa


def overwrite_gen_capas_data(df_gen_capa: pd.DataFrame, new_power_capas: Dict[str, Dict[str, float]],
                             country: str) -> pd.DataFrame:
    if df_gen_capa is not None and country in new_power_capas:
        logging.info(f'OVERWRITTEN ERAA prod. capacity values, in MW: {new_power_capas[country]}')
        for agg_prod_type, new_capa_val in new_power_capas[country].items():
            df_gen_capa.loc[
                df_gen_capa[PROD_TYPE_AGG_COL] == agg_prod_type, 'power_capacity'] = new_capa_val
    return df_gen_capa


def add_failure_asset_to_capas_data(df_gen_capa: pd.DataFrame, failure_power_capa: float) -> pd.DataFrame:
    failure_df = pd.DataFrame.from_dict({
        PROD_TYPE_AGG_COL: ['failure'],
        'power_capacity': [failure_power_capa],
        'power_capacity_turbine': [0.0],
        'power_capacity_pumping': [0.0],
        'power_capacity_injection': [0.0],
        'power_capacity_offtake': [0.0],
        'energy_capacity': [0.0]
    })
    return pd.concat([df_gen_capa, failure_df], ignore_index=True)


def capa_info_log(df_gen_capa: pd.DataFrame):
    # get dict. with only power capacity values to get less verbose logs
    power_capa_dict = create_dict_from_cols_in_df(df=df_gen_capa, key_col=PROD_TYPE_AGG_COL, val_col='power_capacity')
    logging.info(f'-> power capacity values, in MW: {power_capa_dict}')


@dataclass
class Dataset:
    agg_prod_types_with_cf_data: List[str]
    source: str = 'eraa_2023.2'
    is_stress_test: bool = False
    demand: Dict[str, pd.DataFrame] = None
    net_demand: Dict[str, pd.DataFrame] = None
    agg_cf_data: Dict[str, pd.DataFrame] = None
    agg_gen_capa_data: Dict[str, pd.DataFrame] = None
    interco_capas: Dict[Tuple[str, str], float] = None
    # {country: list of associated generation units data}
    generation_units_data: Dict[str, List[GenerationUnitData]] = None

    def get_countries_data(self, uc_run_params: UCRunParams, aggreg_prod_types_def: Dict[str, Dict[str, List[str]]],
                           datatypes_selec: List[str] = None, subdt_selec: List[str] = None,
                           capas_aggreg_pt_with_cf: Dict[str, int] = None):
        """
        Get ERAA data necessary for the selected countries
        :param uc_run_params: UC run parameters, from which main reading infos will be obtained
        :param aggreg_prod_types_def: per-datatype definition of aggreg. to indiv. production types
        :param datatypes_selec: list of datatypes for which data must be read
        :param subdt_selec: list of sub-datatypes for which data must be read
        :param capas_aggreg_pt_with_cf: capacities of prod types with CF data to be used for prod. values calculation
        :returns: {country: df with demand of this country}, {country: df with - per aggreg. prod type CF},
        {country: df with installed generation capas}, df with all interconnection capas (for considered 
        countries and year)
        """
        # default is to read all data, excepting net demand (only used for data-analysis)
        if datatypes_selec is None:
            datatypes_selec = list(DATATYPE_NAMES.__dict__.values())
            datatypes_selec.remove(DATATYPE_NAMES.net_demand)
        # and not to apply capa. values fixed in arg
        if capas_aggreg_pt_with_cf is None:
            capas_aggreg_pt_with_cf = {}

        # set shorter names for simplicity
        countries = uc_run_params.selected_countries
        year = uc_run_params.selected_target_year
        climatic_year = uc_run_params.selected_climatic_year
        selec_agg_prod_types = uc_run_params.selected_prod_types
        power_capacities = uc_run_params.capacities_tb_overwritten
        period_start = uc_run_params.uc_period_start
        period_end = uc_run_params.uc_period_end
        # get - per datatype - folder names
        demand_folder = os.path.join(INPUT_ERAA_FOLDER, DT_SUBFOLDERS.demand)
        res_cf_folder = os.path.join(INPUT_ERAA_FOLDER, DT_SUBFOLDERS.res_capa_factors)
        gen_capas_folder = os.path.join(INPUT_ERAA_FOLDER, DT_SUBFOLDERS.generation_capas)
        interco_capas_folder = os.path.join(INPUT_ERAA_FOLDER, DT_SUBFOLDERS.interco_capas)
        # file prefix
        interco_capas_prefix = DT_FILE_PREFIX.interco_capas
        value_col = COLUMN_NAMES.value

        self.demand = {}
        self.net_demand = {}
        self.agg_cf_data = {}
        self.agg_gen_capa_data = {}

        dts_tb_read = deepcopy(datatypes_selec)
        # datatypes to be added to list of read ones, to be able to obtain net demand
        if DATATYPE_NAMES.net_demand in datatypes_selec:
            dts_tb_read.extend([DATATYPE_NAMES.demand, DATATYPE_NAMES.installed_capa, DATATYPE_NAMES.capa_factor])
            dts_tb_read = list(set(dts_tb_read))

        aggreg_pt_gen_capa_def = aggreg_prod_types_def[DATATYPE_NAMES.installed_capa]

        for country in countries:
            logging.info(3 * '#' + f' For country: {country}')
            # read csv files for different types of data
            current_suffix = f'{year}_{country}'  # common suffix to all ERAA data files
            if DATATYPE_NAMES.demand in dts_tb_read:
                # get demand
                current_df_demand = get_demand_data(folder=demand_folder, file_suffix=current_suffix,
                                                    climatic_year=climatic_year, period=(period_start, period_end),
                                                    is_stress_test=self.is_stress_test)
                # if demand selected add it to dataset
                if DATATYPE_NAMES.demand in datatypes_selec:
                    self.demand[country] = current_df_demand

            if DATATYPE_NAMES.capa_factor in dts_tb_read:
                # get RES capacity factor data
                logging.debug('Get RES capacity factors')
                if DATATYPE_NAMES.capa_factor in datatypes_selec:
                    self.agg_cf_data[country] = None
                # get list of agg. prod. types for which data must be read
                cf_agg_prod_types_tb_read = (
                    get_cf_agg_prod_types_tb_read(selected_agg_prod_types=selec_agg_prod_types[country],
                                                  agg_prod_types_with_cf_data=self.agg_prod_types_with_cf_data,
                                                  subdt_selec=subdt_selec)
                )
                # get RES CF data for these prod. types
                agg_cf_data_read = (
                    get_res_capa_factors_data(folder=res_cf_folder, file_suffix=current_suffix,
                                              climatic_year=climatic_year,
                                              cf_agg_prod_types_tb_read=cf_agg_prod_types_tb_read,
                                              aggreg_pt_cf_def=aggreg_prod_types_def[DATATYPE_NAMES.capa_factor],
                                              period=(period_start, period_end), is_stress_test=self.is_stress_test)
                )

                if agg_cf_data_read is None:
                    logging.warning(
                        N_SPACES_MSG * ' ' + f'No RES data available for country {country} '
                                             f'-> not accounted for in UC model here')
                elif DATATYPE_NAMES.capa_factor in datatypes_selec:
                    self.agg_cf_data[country] = agg_cf_data_read

            if DATATYPE_NAMES.installed_capa in dts_tb_read:
                # fixed capas for agg. prod types with CF data not accounted for here
                if capas_aggreg_pt_with_cf is not None and len(capas_aggreg_pt_with_cf) > 0:
                    logging.warning(f'ERAA capas data for following agg. prod types (with CF data) will not be '
                                    f'accounted for: {capas_aggreg_pt_with_cf} -> replaced by values provided in arg, '
                                    f'for net demand calculation only')
                # get ERAA capas for gen. assets
                current_df_gen_capa = (
                    get_installed_gen_capas_data(folder=gen_capas_folder, file_suffix=current_suffix,
                                                 country=country, aggreg_pt_gen_capa_def=aggreg_pt_gen_capa_def,
                                                 selected_agg_prod_types=selec_agg_prod_types[country])
                )
                # add failure fictive one
                if 'failure' in selec_agg_prod_types[country]:
                    current_df_gen_capa = (
                        add_failure_asset_to_capas_data(df_gen_capa=current_df_gen_capa,
                                                        failure_power_capa=uc_run_params.failure_power_capa)
                    )
                # overwrite capacity values - based on the ones provided in input JSON file(s)
                current_df_gen_capa = (
                    overwrite_gen_capas_data(df_gen_capa=current_df_gen_capa, new_power_capas=power_capacities,
                                             country=country)
                )
                if DATATYPE_NAMES.installed_capa in datatypes_selec:
                    self.agg_gen_capa_data[country] = current_df_gen_capa
                capa_info_log(df_gen_capa=current_df_gen_capa)

            if DATATYPE_NAMES.net_demand in datatypes_selec:
                pts_with_capa_from_arg = []
                # TODO: directly in pd to avoid creation of np arrays?
                # convert to float so that subtraction of CF can be done hereafter
                current_np_net_demand = np.array(current_df_demand[value_col]).astype(np.float64)
                for agg_prod_type in cf_agg_prod_types_tb_read:
                    # get current capa either from fixed data provided as arg of this function
                    if agg_prod_type in capas_aggreg_pt_with_cf:
                        current_capa = capas_aggreg_pt_with_cf[agg_prod_type]
                        pts_with_capa_from_arg.append(agg_prod_type)
                    else:  # or from (ERAA) dataset data
                        current_capa = (
                            current_df_gen_capa.loc[current_df_gen_capa[PROD_TYPE_AGG_COL] == agg_prod_type,
                            'power_capacity'].values)[0]
                    current_cf_data = agg_cf_data_read[agg_cf_data_read[PROD_TYPE_AGG_COL] == agg_prod_type]
                    current_np_net_demand -= current_capa * np.array(current_cf_data[value_col])
                current_df_net_demand = deepcopy(current_df_demand)
                current_df_net_demand[value_col] = current_np_net_demand
                self.net_demand[country] = current_df_net_demand
                if len(pts_with_capa_from_arg) > 0:
                    used_capas_from_arg = {pt: capas_aggreg_pt_with_cf[pt] for pt in pts_with_capa_from_arg}
                    logging.info(f'For net demand calculation, the following prod types have capa values used '
                                 f'from arg, in MW: {used_capas_from_arg}')

        if DATATYPE_NAMES.interco_capa in datatypes_selec:
            # read interconnection capas file
            logging.info('Get interconnection capacities (1 file with data of all countries and years)')
            interco_capas_data_file = f'{interco_capas_folder}/{interco_capas_prefix}_{year}.csv'
            if not os.path.exists(interco_capas_data_file):
                logging.warning(f'Generation capas data file does not exist: {country} not accounted for here')
            else:
                df_interco_capas = pd.read_csv(interco_capas_data_file, sep=FILES_FORMAT.column_sep,
                                               decimal=FILES_FORMAT.decimal_sep)
            # and select information needed for selected countries
            df_interco_capas = select_interco_capas(df_intercos_capa=df_interco_capas, countries=countries)
            # set as dictionary
            origin_col = COLUMN_NAMES.zone_origin
            destination_col = COLUMN_NAMES.zone_destination
            tuple_key_col = 'tuple_key'
            df_interco_capas[tuple_key_col] = \
                df_interco_capas.apply(lambda col: (col[origin_col], col[destination_col]),
                                       axis=1)
            interco_capas = create_dict_from_cols_in_df(df=df_interco_capas, key_col=tuple_key_col, val_col=value_col)
            # add interco capas values set by user
            interco_capas |= uc_run_params.interco_capas_tb_overwritten
            self.interco_capas = interco_capas

    def get_generation_units_data(self, uc_run_params: UCRunParams, pypsa_unit_params_per_agg_pt: Dict[str, dict],
                                  units_complem_params_per_agg_pt: Dict[str, Dict[str, str]]):
        """
        Get generation units data to create them hereafter
        :param uc_run_params
        :param pypsa_unit_params_per_agg_pt: dict of per aggreg. prod type main Pypsa params
        :param units_complem_params_per_agg_pt: # for each aggreg. prod type, a dict. {complem. param name: source
        - "from_json_tb_modif"/"from_eraa_data"}
        """
        countries = list(self.agg_gen_capa_data)
        prod_type_col = COLUMN_NAMES.production_type
        prod_type_agg_col = f'{prod_type_col}_agg'
        value_col = COLUMN_NAMES.value
        # TODO: set as global constants/unify...
        power_capa_key = 'power_capa'
        capa_factor_key = 'capa_factors'

        n_spaces_msg = 2

        self.generation_units_data = {}
        for country in countries:
            logging.debug(f'- for country {country}')
            self.generation_units_data[country] = []
            current_capa_data = self.agg_gen_capa_data[country]
            current_res_cf_data = self.agg_cf_data[country]
            # get list of assets to be treated from capa. data
            agg_prod_types = list(set(current_capa_data[prod_type_agg_col]))
            # initialize set of params for each unit by using pypsa default values
            current_assets_data = {agg_pt: pypsa_unit_params_per_agg_pt[agg_pt] for agg_pt in agg_prod_types}
            # and loop over pt to add complementary params
            for agg_pt in agg_prod_types:
                logging.debug(n_spaces_msg * ' ' + f'* for aggreg. prod. type {agg_pt}')
                # set and add asset name
                gen_unit_name = set_gen_unit_name(country=country, agg_prod_type=agg_pt)
                current_assets_data[agg_pt]['name'] = gen_unit_name
                # and 'type' (the aggreg. prod types used here, with a direct corresp. to PyPSA generators; 
                # made explicit in JSON fixed params files)
                current_assets_data[agg_pt]['type'] = agg_pt
                if agg_pt in units_complem_params_per_agg_pt and len(units_complem_params_per_agg_pt[agg_pt]) > 0:
                    # add pnom attribute if needed
                    if power_capa_key in units_complem_params_per_agg_pt[agg_pt]:
                        logging.debug(2 * n_spaces_msg * ' ' + f'-> add {power_capa_key}')
                        current_power_capa = \
                            get_val_of_agg_pt_in_df(df_data=current_capa_data, prod_type_agg_col=prod_type_agg_col,
                                                    agg_prod_type=agg_pt, value_col='power_capacity',
                                                    static_val=True)
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = int(current_power_capa)

                    # add pmax_pu when variable for RES/fatal units
                    if capa_factor_key in units_complem_params_per_agg_pt[agg_pt]:
                        logging.debug(2 * N_SPACES_MSG * ' ' + f'-> add {capa_factor_key}')
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.capa_factors] = \
                            get_val_of_agg_pt_in_df(df_data=current_res_cf_data, prod_type_agg_col=prod_type_agg_col,
                                                    agg_prod_type=agg_pt, value_col=value_col, static_val=False)
                    # max hours for storage-like assets (energy capa/power capa)

                    # marginal costs/efficiency, from FuelSources
                elif agg_pt == 'failure':
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = \
                        self.agg_gen_capa_data[country].loc[
                            self.agg_gen_capa_data[country]['production_type_agg'] == 'failure', 'power_capacity'].iloc[
                            0]
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.marginal_cost] = uc_run_params.failure_penalty
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.committable] = False
                power_capacity = self.agg_gen_capa_data[country].loc[
                    self.agg_gen_capa_data[country]['production_type_agg'] == agg_pt, 'power_capacity'].iloc[0]
                current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = power_capacity
                energy_capacity = self.agg_gen_capa_data[country].loc[
                    self.agg_gen_capa_data[country]['production_type_agg'] == agg_pt, 'energy_capacity'].iloc[0]
                power_capacity_turbine = self.agg_gen_capa_data[country].loc[
                    self.agg_gen_capa_data[country]['production_type_agg'] == agg_pt, 'power_capacity_turbine'].iloc[0]
                if energy_capacity > 0:
                    power_capacity_pumping = (
                        self.agg_gen_capa_data[country].loc[self.agg_gen_capa_data[country][
                                                                'production_type_agg'] == agg_pt,
                        'power_capacity_pumping'].iloc)[0]
                    if power_capacity_turbine > 0:
                        p_nom = max(abs(power_capacity_turbine), abs(power_capacity_pumping))
                        p_min_pu = power_capacity_pumping / p_nom
                        p_max_pu = power_capacity_turbine / p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.min_power_pu] = p_min_pu
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.capa_factors] = p_max_pu
                        max_hours = energy_capacity / p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.max_hours] = max_hours
                    power_capacity_injection = (
                        self.agg_gen_capa_data[country].loc[self.agg_gen_capa_data[country][
                                                                'production_type_agg'] == agg_pt,
                        'power_capacity_injection'].iloc)[0]
                    power_capacity_offtake = (
                        self.agg_gen_capa_data[country].loc[self.agg_gen_capa_data[country][
                                                                'production_type_agg'] == agg_pt,
                        'power_capacity_offtake'].iloc)[0]
                    if power_capacity_injection > 0:
                        p_nom = max(abs(power_capacity_injection), abs(power_capacity_offtake))
                        p_min_pu = -power_capacity_offtake / p_nom
                        p_max_pu = power_capacity_injection / p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.min_power_pu] = p_min_pu
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.capa_factors] = p_max_pu
                        max_hours = energy_capacity / p_nom
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.max_hours] = max_hours
                    if power_capacity > 0:
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = power_capacity
                elif power_capacity_turbine > 0:
                    p_nom = abs(power_capacity_turbine)
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = p_nom
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.min_power_pu] = 0
                    current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.capa_factors] = 1
                    if power_capacity > 0:
                        current_assets_data[agg_pt][GEN_UNITS_PYPSA_PARAMS.power_capa] = power_capacity

                self.generation_units_data[country].append(GenerationUnitData(**current_assets_data[agg_pt]))

    def set_generation_units_data(self, gen_units_data: Dict[str, List[GenerationUnitData]]):
        self.generation_units_data = gen_units_data

    def set_committable_param(self):
        for country, val in self.generation_units_data.items():
            for i in range(len(val)):
                val[i].committable = False

    def control_min_pypsa_params_per_gen_units(self, pypsa_min_unit_params_per_agg_pt: Dict[str, List[str]]):
        """
        Control that minimal PyPSA parameter infos has been provided before creating generation units
        """
        pypsa_params_errors_list = []
        # loop over countries
        for country, gen_units_data in self.generation_units_data.items():
            # and unit in them
            for elt_unit_data in gen_units_data:
                current_unit_type = elt_unit_data.type
                pypsa_min_unit_params_set = set(pypsa_min_unit_params_per_agg_pt[current_unit_type])
                params_with_init_val_set = set(elt_unit_data.get_non_none_attr_names())
                missing_pypsa_params = list(pypsa_min_unit_params_set - params_with_init_val_set)
                if len(missing_pypsa_params) > 0:
                    current_unit_name = elt_unit_data.name
                    current_msg = (f'country {country}, unit name {current_unit_name} and type {current_unit_type} '
                                   f'-> {missing_pypsa_params}')
                    pypsa_params_errors_list.append(current_msg)
        if len(pypsa_params_errors_list) > 0:
            print_errors_list(error_name='on "minimal" PyPSA gen. units parameters; missing ones for',
                              errors_list=pypsa_params_errors_list)
        else:
            logging.info('PyPSA NEEDED PARAMETERS FOR GENERATION UNITS CREATION HAVE BEEN LOADED!')
