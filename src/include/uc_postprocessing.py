import logging
import warnings
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Optional
import json
import pypsa
from matplotlib import pyplot as plt

from src.common.constants.countries import set_country_trigram
from src.common.constants.links import set_link_name, REVERSE_SIGN
from src.common.constants.prod_types import STOCK_LIKE_PROD_TYPES, ProdTypeNames, add_suffix_to_storage_unit_col, \
    get_prod_type_from_unit_name, set_gen_unit_name, get_country_from_unit_name, PT_ORDER_IN_DATA_OUTPUT, \
    STORAGE_COL_SUFFIX
from src.common.long_term_uc_io import FigNamesPrefix, get_output_figure, get_figure_file_named, get_opt_power_file, \
    get_storage_opt_dec_file, get_marginal_prices_file, get_link_flow_opt_dec_file, get_uc_summary_file
from src.common.plot_params import PlotParams
from src.utils.basic_utils import format_with_spaces, get_default_values, dict_to_str
from src.utils.df_utils import rename_df_columns, sort_out_cols_with_zero_values

OUTPUT_DATE_COL = 'date'


def set_full_prod_type_col_order(prod_type_cols_ordered: List[str], stock_cons_first: bool = True) -> List[str]:
    """
    Update cols_ordered with prod/cons suffix for storage assets
    Args:
        prod_type_cols_ordered:
        stock_cons_first: put first all cons. columns of the storage-like assets; otherwise storage columns with
        cons./prod. (e.g., 'hydro_reservoir_prod') suffix will be added just after the corresponding one without
        suffix (e.g., 'hydro_reservoir')
    Returns:

    """
    # check that columns are all prod types
    known_prod_types = get_default_values(obj=ProdTypeNames)
    unknown_prod_types = set(prod_type_cols_ordered) - set(known_prod_types)
    if len(unknown_prod_types) > 0:
        logging.warning(f'Unknown production types in columns to be completed with stock-columns with-added-suffix '
                        f'("_prod", or "_cons"): {unknown_prod_types}\n-> will not be integrated in plot')
        prod_type_cols_ordered = [pt_col for pt_col in prod_type_cols_ordered if pt_col in known_prod_types]
    # case with _cons columns for stock-like assets first
    if stock_cons_first:
        cols_ordered_stock_cons = [add_suffix_to_storage_unit_col(col=col, col_type='cons')
                                   for col in prod_type_cols_ordered if col in STOCK_LIKE_PROD_TYPES]
        pt_cols_ordered_prod_compl = []
        for pt_col in prod_type_cols_ordered:
            # col without suffix (+ with prod suffix if storage-like prod type)
            pt_cols_ordered_prod_compl.append(pt_col)
            if pt_col in STOCK_LIKE_PROD_TYPES:
                pt_cols_ordered_prod_compl.append(add_suffix_to_storage_unit_col(col=pt_col, col_type='prod'))
        # concat. cons and prod columns
        cols_ordered_stock_cons.extend(pt_cols_ordered_prod_compl)
        return cols_ordered_stock_cons
    # case with _cons columns for stock-like asset just after their pt (wo suffix)
    pt_cols_ordered_compl = []
    cons_first = True
    for pt_col in prod_type_cols_ordered:
        # col without suffix (+ with prod suffix if storage-like prod type)
        pt_cols_ordered_compl.append(pt_col)
        if pt_col in STOCK_LIKE_PROD_TYPES:
            pt_col_cons = add_suffix_to_storage_unit_col(col=pt_col, col_type='cons')
            pt_col_prod = add_suffix_to_storage_unit_col(col=pt_col, col_type='prod')
            added_pt_cols = [pt_col_cons, pt_col_prod] if cons_first else [pt_col_prod, pt_col_cons]
            pt_cols_ordered_compl.extend(added_pt_cols)
    return pt_cols_ordered_compl


def set_col_order_for_plot(df: pd.DataFrame, cols_ordered: List[str], is_prod_type_cols: bool = False,
                           stock_cons_first: bool = True) -> pd.DataFrame:
    """
    Set column order in a df for plot
    Args:
        df: considered dataframe for plot
        cols_ordered: predefined order of columns in plot - see input\functional_params\plot_params.json
        is_prod_type_cols: are columns prod types?
        stock_cons_first: put consumption columns of the stock first?

    Returns: df with columns in requested order
    """
    current_df_cols = list(df.columns)
    if is_prod_type_cols:
        cols_ordered_completed = set_full_prod_type_col_order(prod_type_cols_ordered=cols_ordered,
                                                              stock_cons_first=stock_cons_first)
    else:
        cols_ordered_completed = cols_ordered
    current_df_cols_ordered = [col for col in cols_ordered_completed if col in current_df_cols]
    df = df[current_df_cols_ordered]
    return df


def set_full_cols_for_storage_df(df: pd.DataFrame, col_type: str) -> pd.DataFrame:
    old_cols = list(df.columns)
    new_cols = {col: add_suffix_to_storage_unit_col(col=col, col_type=col_type) for col in old_cols}
    df = rename_df_columns(df=df, old_to_new_cols=new_cols)
    return df


@dataclass
class UCSummaryMetrics:
    ENERGY_UNIT = 'GWh'
    COST_UNIT = 'M€'
    CO2_EMIS_UNIT = '??'
    # Energy Not Served (ENS), as the sum of failure volumes over horizon simulated
    # (attention, in GWh)
    per_country_ens: Dict[str, float]
    # Number of hours with nonzero failure volume (may be a criterion integrated in national energy regulation
    # for capa. sizing, e.g. in France with maximally 3hours in average over a set of climatic scenarios)
    per_country_n_failure_hours: Dict[str, float]
    total_cost: float  # over Europe
    total_operational_cost: float  # without failure (fictive) penalty cost
    total_co2_emissions: float  # over Europe
    per_country_total_cost: Dict[str, float] = None  # including failure penalty (fictive) cost
    per_country_total_operational_cost: Dict[str, float] = None  # without this cost
    per_country_co2_emissions: Dict[str, float] = None

    def __repr__(self, europe_name: str = None) -> str:
        metric_sep = '\n-> '
        uc_summary_metrics_str = 'UCSummaryMetrics'
        if europe_name is not None:
            uc_summary_metrics_str += f'for {europe_name}'
        uc_summary_metrics_str += \
            f'{metric_sep}Energy Not Served ({self.ENERGY_UNIT}): {dict_to_str(d=self.per_country_ens)}'
        uc_summary_metrics_str += \
            f'{metric_sep}Number of failure hours: {dict_to_str(d=self.per_country_n_failure_hours)}'
        uc_summary_metrics_str += \
            f'{metric_sep}Total cost (over Europe, {self.COST_UNIT}): {format_with_spaces(number=self.total_cost)}'
        uc_summary_metrics_str += \
            (f'{metric_sep}Total operational cost (Europe, without failure penalty incl., '
             f'{self.COST_UNIT}): {format_with_spaces(number=self.total_operational_cost)}')
        uc_summary_metrics_str += \
            (f'{metric_sep}Total CO2 emissions (over Europe, {self.CO2_EMIS_UNIT}): '
             f'{format_with_spaces(number=self.total_co2_emissions)}')
        if self.per_country_total_cost is not None:
            uc_summary_metrics_str += \
                (f'{metric_sep}PER COUNTRY total cost ({self.COST_UNIT}): '
                 f'{dict_to_str(d=self.per_country_total_cost, nbers_with_spaces=True)}')
        if self.per_country_total_operational_cost is not None:
            uc_summary_metrics_str += \
                (f'{metric_sep}PER COUNTRY total operational cost (without failure penalty incl., '
                 f'{self.COST_UNIT}): {dict_to_str(d=self.per_country_total_operational_cost, nbers_with_spaces=True)}')
        if self.per_country_co2_emissions is not None:
            uc_summary_metrics_str += \
                (f'{metric_sep}PER COUNTRY total CO2 emissions ({self.CO2_EMIS_UNIT}): '
                 f'{dict_to_str(d=self.per_country_co2_emissions, nbers_with_spaces=True)}')
        return uc_summary_metrics_str

    def json_dump(self, year: int, climatic_year: int, start_horizon: datetime, country: str = 'europe',
                  toy_model_output: bool = False, output_tag: str = None):
        summary_dict = asdict(self)
        # remove None values
        summary_dict = {key: val for key, val in summary_dict.items() if val is not None}
        file = get_uc_summary_file(country=country, year=year, climatic_year=climatic_year,
                                   start_horizon=start_horizon, toy_model_output=toy_model_output,
                                   output_tag=output_tag)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f)


def add_storage_decisions_to_prod_df(country_trigram: str, generator_prod: pd.DataFrame, storage_prod: pd.DataFrame,
                                     storage_cons: pd.DataFrame, cast_cons_as_prod: bool = False) -> pd.DataFrame:
    prod_cols = list(generator_prod.columns)
    storage_prod_cols = [unit_name for unit_name in list(storage_prod) if unit_name.startswith(country_trigram)]
    prod_cols.extend(storage_prod_cols)
    current_storage_prod = storage_prod[storage_prod_cols]
    current_storage_cons = storage_cons[storage_prod_cols]
    # Key step; switch to prod. convention for consumption? Necessary for stack prod. plot for ex.
    if cast_cons_as_prod:
        current_storage_cons *= -1
    # add suffix to prod/cons columns to distinguish them after concatenation below
    new_prod_cols = {col: add_suffix_to_storage_unit_col(col=get_prod_type_from_unit_name(prod_unit_name=col),
                                                         col_type='prod') for col in storage_prod_cols}
    new_cons_cols = {col: add_suffix_to_storage_unit_col(col=get_prod_type_from_unit_name(prod_unit_name=col),
                                                         col_type='cons') for col in storage_prod_cols}
    current_storage_prod = rename_df_columns(df=current_storage_prod, old_to_new_cols=new_prod_cols)
    current_storage_cons = rename_df_columns(df=current_storage_cons, old_to_new_cols=new_cons_cols)
    return pd.concat([generator_prod, current_storage_cons, current_storage_prod], axis=1)


def set_reverse_flow_col(flow_col: str) -> str:
    flow_col_split = flow_col.split("_")
    origin_and_dest = flow_col_split[0].split("-")
    carrier_name = flow_col_split[1]
    return set_link_name(origin_bus_name=origin_and_dest[0], dest_bus_name=origin_and_dest[1],
                         carrier_name=carrier_name, sign=REVERSE_SIGN)


def get_and_order_bus_and_pt_from_cols(cols: List[str]) -> (List[str], List[str]):
    # get bus and prod type names from columns
    all_bus = list(set([get_country_from_unit_name(prod_unit_name=col) for col in cols]))
    all_prod_types = list(set([get_prod_type_from_unit_name(prod_unit_name=col) for col in cols]))
    # order them
    all_bus.sort()
    all_prod_types = [prod_type for prod_type in PT_ORDER_IN_DATA_OUTPUT if prod_type in all_prod_types]
    return all_bus, all_prod_types


def set_lexico_order_prod_cols(prod_cols: List[str]) -> List[str]:
    all_bus, all_prod_types = get_and_order_bus_and_pt_from_cols(cols=prod_cols)
    ordered_prod_cols = []
    for bus_name, prod_type in product(all_bus, all_prod_types):
        prod_col = set_gen_unit_name(country=bus_name, agg_prod_type=prod_type)
        if prod_col in prod_cols:
            ordered_prod_cols.append(prod_col)
    return ordered_prod_cols


def set_lexico_order_storage_cols(storage_cols: List[str]) -> List[str]:
    all_bus, all_prod_types = get_and_order_bus_and_pt_from_cols(cols=storage_cols)
    ordered_storage_cols = []
    for bus_name, prod_type, suffix_type in product(all_bus, all_prod_types, STORAGE_COL_SUFFIX.keys()):
        storage_col = set_gen_unit_name(country=bus_name, agg_prod_type=prod_type)
        storage_col = add_suffix_to_storage_unit_col(col=storage_col, col_type=suffix_type)
        if storage_col in storage_cols:
            ordered_storage_cols.append(storage_col)
    return ordered_storage_cols


def set_consec_opposite_flow_cols(flow_cols: List[str], reverse_flow_cols_after: bool = True,
                                  flow_carrier_name: str = "ac") -> List[str]:
    # get list of zone used in (reverse) flow columns
    # first list of (origin, dest.)
    all_o_and_d = [col.split("_")[0].split("-") for col in flow_cols]
    all_bus = set([o_and_d[0] for o_and_d in all_o_and_d])
    all_bus |= set([o_and_d[1] for o_and_d in all_o_and_d])
    all_bus = list(all_bus)
    # sort, by alpha. order
    all_bus.sort()
    ordered_flow_cols = []
    ordered_reverse_flow_cols = []
    for origin_bus, dest_bus in product(all_bus, all_bus):
        if dest_bus == origin_bus:
            continue
        current_link_col = set_link_name(origin_bus_name=origin_bus, dest_bus_name=dest_bus,
                                         carrier_name=flow_carrier_name)
        if current_link_col in flow_cols and current_link_col not in ordered_flow_cols:
            ordered_flow_cols.append(current_link_col)
            opposite_link_col = set_link_name(origin_bus_name=dest_bus, dest_bus_name=origin_bus,
                                              carrier_name=flow_carrier_name)
            if opposite_link_col in flow_cols and opposite_link_col not in ordered_flow_cols:
                ordered_flow_cols.append(opposite_link_col)
            # reverse flow columns
            for (origin_bus_2, dest_bus_2) in [(origin_bus, dest_bus), (dest_bus, origin_bus)]:
                current_col_reverse = set_link_name(origin_bus_name=origin_bus_2, dest_bus_name=dest_bus_2,
                                                    carrier_name=flow_carrier_name, sign=REVERSE_SIGN)
                if current_col_reverse in flow_cols:
                    # all reverse columns will be put at the end
                    if reverse_flow_cols_after and current_col_reverse not in ordered_reverse_flow_cols:
                        ordered_reverse_flow_cols.append(current_col_reverse)
                    elif current_col_reverse not in ordered_flow_cols:
                        ordered_flow_cols.append(current_col_reverse)
    ordered_flow_cols.extend(ordered_reverse_flow_cols)
    return ordered_flow_cols


def process_data_to_csv(data_type: str, df: pd.DataFrame, rename_snapshot_col: bool, order_cols: bool) -> pd.DataFrame:
    if rename_snapshot_col:
        df.index.name = OUTPUT_DATE_COL
    if order_cols:
        current_cols = list(df.columns)
        if data_type == "prod":
            ordered_cols = set_lexico_order_prod_cols(prod_cols=current_cols)
        elif data_type == "storage_all_decs":
            ordered_cols = set_lexico_order_storage_cols(storage_cols=current_cols)
        elif data_type == "link_flow":
            ordered_cols = set_consec_opposite_flow_cols(flow_cols=current_cols)
        else:
            raise Exception(f"Unknown data type {data_type} for output data processing before saving it to csv")
        if len(current_cols) > len(ordered_cols):
            lost_cols = list(set(current_cols) - set(ordered_cols))
            n_lost_cols = len(lost_cols)
            logging.warning(f"{n_lost_cols} columns lost when ordering them before saving to csv: {lost_cols}")
        df = df[ordered_cols]
    # cast to int to avoid useless numeric precisions and associated... issues!
    df = df.astype(int)
    return df


# TODO: check typing
class UCOptimalSolution:
    def __init__(self, network_name: str):
        self.name = network_name
        self.prod: pd.DataFrame = None  # production profiles optimal decisions
        self.storage_prod: pd.DataFrame = None  # storage 3 variables: prod, cons. and State-of-Charge
        self.storage_cons: pd.DataFrame = None
        self.storage_soc: pd.DataFrame = None
        self.link_flow_direct: pd.DataFrame = None  # flow to the first bus of the link (origin)
        self.link_flow_reverse: pd.DataFrame = None  # to the second bus (destination)
        # dual variables associated to supply-demand equilibrium const. - interpreted as marginal prices
        self.sde_dual: pd.DataFrame = None
        self.link_capa_dual: pd.DataFrame = None

    def get_prod_var_opt(self, network: pypsa.Network):
        self.prod = network.generators_t.p

    def get_storage_vars_opt(self, network: pypsa.Network):
        self.storage_prod = network.storage_units_t.p_dispatch
        self.storage_cons = network.storage_units_t.p_store
        self.storage_soc = network.storage_units_t.state_of_charge

    def get_link_flow_vars_opt(self, network: pypsa.Network):
        self.link_flow_direct = network.links_t.p0
        self.link_flow_reverse = network.links_t.p1

    def get_sde_dual_var_opt(self, network: pypsa.Network):
        self.sde_dual = network.buses_t.marginal_price

    def get_link_capa_dual_var_opt(self, network: pypsa.Network):
        logging.warning('Link capa. dual variable code to be fixed (needs to "enter" into Linopy framework '
                        '- not directly available in PyPSA)')
        linopy_model = network.model
        # TODO: fix it, going into linopy framework... apparently not possible to directly access this info from PyPSA
        # see linopy_model.constraints to get a list of considered constraints (Link-fix-p-lower, but cannot be accessed
        # with link_fix_p_lower)
        # Issue self.network.model.dual.keys() empty...
        # loop over links... and build df
        # con_obj = linopy_model.link_fix_p_upper[link_name]
        # Retrieve the dual value (shadow price)
        # dual_value = linopy_model.dual[con_obj]
        self.link_capa_dual = None

    def plot_prod(self, plot_params_agg_pt: PlotParams, country: str, year: int, climatic_year: int,
                  start_horizon: datetime, toy_model_output: bool = False, rm_all_zero_curves: bool = True,
                  include_storage: bool = False, output_tag: str = None):
        """
        Plot 'stack' of optimized production profiles
        """
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with ((warnings.catch_warnings())):
            warnings.simplefilter("ignore")
            # sort values to get only prod of given country
            country_trigram = set_country_trigram(country=country)
            country_prod_cols = [unit_name for unit_name in list(self.prod) if unit_name.startswith(country_trigram)]
            current_prod = self.prod[country_prod_cols]
            # suppress trigram from prod unit names to simplify legend in figures
            new_prod_cols = {unit_name: get_prod_type_from_unit_name(prod_unit_name=unit_name)
                             for unit_name in country_prod_cols}
            current_prod = rename_df_columns(df=current_prod, old_to_new_cols=new_prod_cols)
            # include storage prod and cons. data in this plot
            if include_storage:
                current_prod = add_storage_decisions_to_prod_df(
                    country_trigram=country_trigram, generator_prod=current_prod, storage_prod=self.storage_prod,
                    storage_cons=self.storage_cons, cast_cons_as_prod=True)
                # add identical colors for the prod type with prod/cons suffix added in df for plot
                plot_params_agg_pt.add_colors_for_stock_with_suffix()
            current_prod = set_col_order_for_plot(df=current_prod, cols_ordered=plot_params_agg_pt.order,
                                                  is_prod_type_cols=True, stock_cons_first=True)
            if rm_all_zero_curves:
                current_prod = sort_out_cols_with_zero_values(df=current_prod, abs_val_threshold=1e-2)
            # TODO: put alpha into plot_params.json
            prod_gw = current_prod.div(1e3)

            fig, ax = plt.subplots()

            x = prod_gw.index

            # Cols always negatives / positives
            neg_cols = [c for c in prod_gw.columns if (prod_gw[c] <= 0).all()]
            pos_cols = [c for c in prod_gw.columns if (prod_gw[c] >= 0).all()]

            if len(pos_cols) == len(prod_gw.columns):  # simple case, with only all-positive cols
                prod_gw.plot.area(subplots=False, ylabel='GW', color=plot_params_agg_pt.per_case_color, alpha=0.3)
            else:
                # Stack all-negative columns
                neg_bottom = np.zeros(len(prod_gw))
                for col in neg_cols:
                    values = prod_gw[col].values
                    ax.fill_between(
                        x,
                        neg_bottom,
                        neg_bottom + values,
                        color=plot_params_agg_pt.per_case_color[col],
                        alpha=0.3,
                        label=col
                    )
                    neg_bottom += values

                # Stack all-positive columns
                pos_bottom = np.zeros(len(prod_gw))
                for col in pos_cols:
                    values = prod_gw[col].values
                    ax.fill_between(
                        x,
                        pos_bottom,
                        pos_bottom + values,
                        color=plot_params_agg_pt.per_case_color[col],
                        alpha=0.3,
                        label=col
                    )
                    pos_bottom += values

            ax.set_ylabel("GW")
            ax.legend(title=None)
            plt.tight_layout()

            # to avoid having the index name in the legend block ('Generator'; not very useful here)
            plt.legend(title=None)
            plt.tight_layout()
            plt.savefig(get_output_figure(fig_name=FigNamesPrefix.production, country=country, year=year,
                                          climatic_year=climatic_year, start_horizon=start_horizon,
                                          toy_model_output=toy_model_output, output_tag=output_tag))
            plt.close()

    def plot_link_flows(self, origin_country: str, year: int, climatic_year: int,
                        start_horizon: datetime, toy_model_output: bool = False, output_tag: str = None):
        # logging.info('Preliminary code for optimal flow plotting to be adapted/tested to get output figures')
        # select links with origin_country as p0
        direct_flows = self.link_flow_direct
        reverse_flows = self.link_flow_reverse
        links_for_plot = []
        # cf. Copilot piece of code :) -> to be tested/adapted
        # flow_df = direct_flows.copy()
        # flow_df.columns = [f"{col} (p0)" for col in flow_df.columns]
        # for col in reverse_flows.columns:
        #     flow_df[f"{col} (p1)"] = reverse_flows[col]
        #     flow_df.reset_index(inplace=True)
        # # plot -> to be activated after having checked code above
        # # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     self.link_flow_var_opt_direct.div(1e3)[links_for_plot].plot.line(subplots=False, ylabel='GW')
        #     plt.tight_layout()
        #     plt.savefig(get_figure_file_named('link_flows', country=origin_country, year=year, climatic_year=climatic_year,
        #                                       start_horizon=start_horizon, toy_model_output=toy_model_output)
        #                 )
        #     plt.close()

    def plot_cum_export_flows(self):
        # TODO: aggreg. plot with sum of exported flow per country (as a function of time)
        bob = 1

    def plot_geo_synthesis_of_flows(self):
        # TODO: cf. other proposition from Copilot for geographic representation
        bob = 1

    def plot_failure(self, country: str, year: int, climatic_year: int, start_horizon: datetime,
                     toy_model_output: bool = False, output_tag: str = None):
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            failure_unit_name = set_gen_unit_name(country=country, agg_prod_type='failure')
            self.prod.div(1e3)[failure_unit_name].plot.line(subplots=False, ylabel='GW')
            plt.tight_layout()
            plt.savefig(get_figure_file_named('failure', country=country, year=year, climatic_year=climatic_year,
                                              start_horizon=start_horizon, toy_model_output=toy_model_output,
                                              output_tag=output_tag)
                        )
            plt.close()

    def plot_marginal_price(self, plot_params_zone: PlotParams, year: int, climatic_year: int, start_horizon: datetime,
                            country: str = 'europe', toy_model_output: bool = False, output_tag: str = None):
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sde_dual_var_opt_plot = set_col_order_for_plot(df=self.sde_dual, cols_ordered=plot_params_zone.order)
            sde_dual_var_opt_plot.plot.line(figsize=(8, 3), ylabel='Euro per MWh',
                                            color=plot_params_zone.per_case_color)
            plt.tight_layout()
            plt.savefig(get_output_figure(fig_name=FigNamesPrefix.prices, country=country, year=year,
                                          climatic_year=climatic_year, start_horizon=start_horizon,
                                          toy_model_output=toy_model_output, output_tag=output_tag)
                        )
            plt.close()

    def save_decisions_to_csv(self, year: int, climatic_year: int, start_horizon: datetime,
                              rename_snapshot_col: bool = True, toy_model_output: bool = False,
                              country: str = 'europe', lexico_order_prod_cols: bool = True,
                              lexico_order_storage_cols: bool = True, consec_opposite_flow: bool = True,
                              save_storage_soc: bool = True, output_tag: str = None):
        # TODO: check if unique country and in this case (i) suppress country prefix in asset names
        # opt prod decisions for all but Storage assets
        opt_p_csv_file = get_opt_power_file(country=country, year=year, climatic_year=climatic_year,
                                            start_horizon=start_horizon, toy_model_output=toy_model_output,
                                            output_tag=output_tag)
        logging.info(f'Save - all but Storage assets - optimal dispatch decisions to csv file {opt_p_csv_file}')
        df_prod_opt = self.prod
        df_prod_opt = process_data_to_csv(data_type="prod", df=df_prod_opt, rename_snapshot_col=rename_snapshot_col,
                                          order_cols=lexico_order_prod_cols)
        df_prod_opt.to_csv(opt_p_csv_file)
        # then storage assets decisions
        storage_opt_dec_csv_file = \
            get_storage_opt_dec_file(country=country, year=year, climatic_year=climatic_year,
                                     start_horizon=start_horizon, toy_model_output=toy_model_output,
                                     output_tag=output_tag)
        logging.info(f'Save Storage optimal decisions to csv file {storage_opt_dec_csv_file}')
        # join the 3 Storage result dfs
        df_prod_opt = self.storage_prod
        df_cons_opt = self.storage_cons
        # rename first the different columns -> adding prod/cons/soc suffixes
        df_prod_opt = set_full_cols_for_storage_df(df=df_prod_opt, col_type='prod')
        df_cons_opt = set_full_cols_for_storage_df(df=df_cons_opt, col_type='cons')
        df_storage_all_decs = df_prod_opt.join(df_cons_opt)
        if save_storage_soc:
            # Storage SoC
            df_soc_opt = self.storage_soc
            df_soc_opt = set_full_cols_for_storage_df(df=df_soc_opt, col_type='soc')
            df_storage_all_decs = df_storage_all_decs.join(df_soc_opt)
        df_storage_all_decs = (
            process_data_to_csv(data_type="storage_all_decs", df=df_storage_all_decs,
                                rename_snapshot_col=rename_snapshot_col, order_cols=lexico_order_storage_cols)
        )
        df_storage_all_decs.to_csv(storage_opt_dec_csv_file)
        # and finally link flow decisions
        link_flow_opt_dec_csv_file = \
            get_link_flow_opt_dec_file(country=country, year=year, climatic_year=climatic_year,
                                       start_horizon=start_horizon, toy_model_output=toy_model_output,
                                       output_tag=output_tag)
        logging.info(f'Save link flow optimal decisions to csv file {link_flow_opt_dec_csv_file}')
        df_link_flow_opt_direct = self.link_flow_direct
        df_link_flow_opt_reverse = self.link_flow_reverse
        # add reverse suffix to reverse flows
        new_cols = []
        for flow_col in df_link_flow_opt_reverse.columns:
            new_cols.append(set_reverse_flow_col(flow_col=flow_col))
        df_link_flow_opt_reverse.columns = new_cols
        df_link_flow_opt = pd.concat([df_link_flow_opt_direct, df_link_flow_opt_reverse], axis=1)
        df_link_flow_opt = process_data_to_csv(data_type="link_flow", df=df_link_flow_opt,
                                               rename_snapshot_col=rename_snapshot_col, order_cols=consec_opposite_flow)
        df_link_flow_opt.to_csv(link_flow_opt_dec_csv_file)

    def save_marginal_prices_to_csv(self, year: int, climatic_year: int, start_horizon: datetime,
                                    rename_snapshot_col: bool = True, toy_model_output: bool = False,
                                    country: str = 'europe', output_tag: str = None):
        logging.info('Save marginal prices decisions to .csv file')
        marginal_prices_csv_file = get_marginal_prices_file(country=country, year=year,
                                                            climatic_year=climatic_year,
                                                            start_horizon=start_horizon,
                                                            toy_model_output=toy_model_output,
                                                            output_tag=output_tag)
        df_sde_dual_var_opt = self.sde_dual
        if rename_snapshot_col:
            df_sde_dual_var_opt.index.name = OUTPUT_DATE_COL
        # do NOT cast this df, given that price values can be accurate at some decimals
        # -> may be useful to observe the correspondence with (input) marginal cost values
        df_sde_dual_var_opt.to_csv(marginal_prices_csv_file)

    def get_prod_given_bus(self, bus_name: str) -> Optional[pd.DataFrame]:
        if self.prod is None:
            logging.warning(f'Optimal variable prod. df cannot be obtained for bus {bus_name}; '
                            f'the prod_var_opt variable needs to be got first')
            return None
        prod_unit_prefix = f'{bus_name}_'
        current_cols = [col for col in self.prod.columns if col.startswith(prod_unit_prefix)]
        return self.prod[current_cols]

    def calc_co2_emissions(self, countries: List[str], snapshot_weightings: pd.DataFrame, co2_emi_factors: pd.DataFrame,
                           per_country: bool = False) \
            -> Union[float, Dict[str, float]]:
        if per_country:
            per_country_co2_emissions = {}
            for country in countries:
                current_prod_var_opt = self.get_prod_given_bus(bus_name=country)
                per_country_co2_emissions[country] = (
                    float(current_prod_var_opt.multiply(snapshot_weightings, axis=0)
                          .multiply(co2_emi_factors, axis=1).sum().sum())
                )
            return per_country_co2_emissions
        return self.prod.multiply(snapshot_weightings, axis=0).multiply(co2_emi_factors, axis=1).sum().sum()

    def calc_per_country_total_cost(self, countries: List[str], snapshot_weightings: pd.DataFrame,
                                    marginal_costs: pd.DataFrame, is_operational_cost: bool = False) \
            -> Dict[str, float]:
        """
        Calculate per-country (bus) total cost over the considered horizon: sum_t sum_{prod unit i} marginal
        cost(i) * prod(i,t):
        :param countries: for which calculation must be done
        :param snapshot_weightings: weights of the different "snapshots" (time-slots in PyPSA terminology)
        :param marginal_costs: of different prod. types considered
        :param is_operational_cost: in this case do not integrate failure penalty cost in this calculation
        """
        per_country_total_cost = {}
        for country in countries:
            current_prod_var_opt = self.get_prod_given_bus(bus_name=country)
            if is_operational_cost:
                # drop failure column from df TODO: set failure column name from constants (functions)
                current_prod_var_opt = current_prod_var_opt.drop(f'{country}_failure', axis=1)
            per_country_total_cost[country] = (
                float(current_prod_var_opt.multiply(snapshot_weightings, axis=0)
                      .multiply(marginal_costs, axis=1).sum().sum())
            )
        return per_country_total_cost

    def set_uc_summary_metrics(self, network: pypsa.Network, total_cost: float,
                               failure_penalty: float = None) -> UCSummaryMetrics:
        logging.info('Set UC summary metrics')
        failure_prod_cols = [col for col in self.prod.columns if col.endswith('_failure')]
        df_failure_opt = self.prod[failure_prod_cols]
        per_country_ens = {key: float(val) for key, val in dict(df_failure_opt.sum()).items()}
        per_country_n_failure_h = {key: int(val) for key, val in dict((df_failure_opt > 0).sum(axis=0)).items()}
        # remove '_failure' suffix from two previous dict. keys
        per_country_ens = {key.split('_')[0]: val for key, val in per_country_ens.items()}
        per_country_n_failure_h = {key.split('_')[0]: val for key, val in per_country_n_failure_h.items()}
        if failure_penalty is not None:
            eur_failure_volume = sum(per_country_ens.values())
            eur_total_ope_cost = total_cost - failure_penalty * eur_failure_volume
        else:
            eur_total_ope_cost = None
        # co2 emissions
        countries = list(set(network.buses.index))
        co2_emi_factors = network.generators.carrier.map(network.carriers.co2_emissions)
        total_co2_emissions = (
            self.calc_co2_emissions(countries=countries, snapshot_weightings=network.snapshot_weightings.generators,
                                    co2_emi_factors=co2_emi_factors)
        )
        per_country_co2_emissions = (
            self.calc_co2_emissions(countries=countries, snapshot_weightings=network.snapshot_weightings.generators,
                                    co2_emi_factors=co2_emi_factors, per_country=True)
        )
        per_country_total_cost = (
            self.calc_per_country_total_cost(countries=countries,
                                             snapshot_weightings=network.snapshot_weightings.generators,
                                             marginal_costs=network.generators.marginal_cost)
        )
        per_country_total_operational_cost = (
            self.calc_per_country_total_cost(countries=countries,
                                             snapshot_weightings=network.snapshot_weightings.generators,
                                             marginal_costs=network.generators.marginal_cost, is_operational_cost=True)
        )
        # attention convert to GWh/M€ and int to get smaller values for synthesis. TODO: check CO2 emissions unit!
        cost_conversion_factor = 1e-6
        co2_emis_conversion_factor = 1e-3
        return UCSummaryMetrics(
            per_country_ens={c: int(val / 1e3) for c, val in per_country_ens.items()},
            per_country_n_failure_hours=per_country_n_failure_h,
            total_cost=int(total_cost * cost_conversion_factor),
            total_operational_cost=int(eur_total_ope_cost * cost_conversion_factor),
            total_co2_emissions=int(total_co2_emissions * co2_emis_conversion_factor),
            per_country_total_cost={c: int(val * cost_conversion_factor) for c, val in per_country_total_cost.items()},
            per_country_total_operational_cost={c: int(val * cost_conversion_factor)
                                                for c, val in per_country_total_operational_cost.items()},
            per_country_co2_emissions={c: int(val * co2_emis_conversion_factor) for
                                       c, val in per_country_co2_emissions.items()})
