import logging
import warnings
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Union, Optional
import json
import pypsa
import xarray
from matplotlib import pyplot as plt
import plotly.express as px

from src.common.constants.countries import set_country_trigram
from src.common.constants.links import set_link_name, REVERSE_SIGN, get_bus_from_link_name
from src.common.constants.prod_types import STOCK_LIKE_PROD_TYPES, ProdTypeNames, add_suffix_to_storage_unit_col, \
    get_prod_type_from_unit_name, set_gen_unit_name, get_country_from_unit_name, PT_ORDER_IN_DATA_OUTPUT, \
    STORAGE_COL_SUFFIX, rm_country_from_prod_unit_name
from src.common.constants.pypsa_params import PypsaOptimConstNames
from src.common.logger import format_relative_path_for_log
from src.common.long_term_uc_io import FigNamesPrefix, get_output_figure, get_figure_file_named, get_opt_power_file, \
    get_storage_opt_dec_file, get_marginal_prices_file, get_link_flow_opt_dec_file, get_uc_summary_file, \
    get_link_capa_dual_vars_file
from src.common.plot_params import PlotParams
from src.utils.basic_utils import format_with_spaces, get_default_values, dict_to_str
from src.utils.df_utils import rename_df_columns, sort_out_cols_with_zero_values, is_column_constant, \
    create_df_from_dict
from src.utils.plot import get_similar_color

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


def set_stock_color(prod_type: str, per_pt_colors: dict[str, str], used_colors: list[str] = None) -> str:
    if used_colors is None:
        used_colors = []

    color = per_pt_colors[prod_type]
    while color in used_colors:
        color = get_similar_color(color=color, hue_shift=0.1)
    used_colors.append(color)
    return color, used_colors


@dataclass
class UCSummaryMetrics:
    ENERGY_UNIT = 'GWh'
    COST_UNIT = 'M€'
    CO2_EMIS_UNIT = 'ktCO2eq.'
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
            (f'{metric_sep}Energy Not Served ({self.ENERGY_UNIT}): '
             f'{dict_to_str(d=self.per_country_ens, nbers_with_spaces=True)}')
        uc_summary_metrics_str += \
            (f'{metric_sep}Number of failure hours: '
             f'{dict_to_str(d=self.per_country_n_failure_hours, nbers_with_spaces=True)}')
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
        logging.info(f"Save UC summary metrics to file {format_relative_path_for_log(path=file)}")
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


def duals_to_link_dual_dict(duals: xarray.DataArray) -> dict[str, np.ndarray]:
    link_dim = next((dim for dim in duals.dims if dim.startswith("Link") or dim == "name"), None)
    if link_dim is None:
        link_dim = next((dim for dim in duals.dims if dim != "snapshot"), None)
    if link_dim is None:
        raise ValueError(f"Could not identify link dimension in duals with dims: {duals.dims}")

    duals = duals.transpose(link_dim, "snapshot")

    return dict(zip(duals.coords[link_dim].values, duals.values, strict=True))


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
        self.link_capa_dual: dict[str, np.array] = None  # {link name: vector with all values, one per snapshot}

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

    def get_link_capa_dual_var_opt(self, network: pypsa.Network) -> dict[str, float]:

        model = network.model

        if not hasattr(model, "dual") or len(model.dual) == 0:
            raise RuntimeError(
                "No dual values available. "
                "Make sure the optimization was run with assign_all_duals=True."
            )

        link_ub_constraint = model.constraints[PypsaOptimConstNames.link_ub]
        duals = link_ub_constraint.dual
        self.link_capa_dual = duals_to_link_dual_dict(duals)

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

    def plot_stock_soc(self, plot_params_agg_pt: PlotParams, country: str, year: int, climatic_year: int,
                       start_horizon: datetime, storage_const_soc_max: dict[str, float],
                       hydro_soc_min: xarray.DataArray = None, hydro_soc_max: xarray.DataArray = None,
                       plot_in_pu: bool = True, toy_model_output: bool = False,
                       output_tag: str = None):
        """
        Plot storage state of charge.

            Top subplot:
                Hydro reservoirs with optional min/max SOC bounds.

            Bottom subplot:
                All other storage units.

        """
        # TODO: txt with assets with SOC=0 for all t, to avoid saturating graph...
        df_soc_opt = self.storage_soc.copy()
        if country != "europe":
            country_prefix = f"{set_country_trigram(country=country)}_"
            current_country_assets = [col for col in df_soc_opt.columns if col.startswith(country_prefix)]
            df_soc_opt = df_soc_opt[current_country_assets]

        # select two different classes of stocks for two different subplots
        hydro_reservoirs = [col for col in df_soc_opt.columns
                            if get_prod_type_from_unit_name(prod_unit_name=col) == ProdTypeNames.hydro_reservoir]

        other_stocks = [col for col in df_soc_opt.columns if col not in hydro_reservoirs]

        hydro_soc_min_plot = None
        hydro_soc_max_plot = None

        # convert all values in per unit (of storage CONSTANT capacity)
        if plot_in_pu:

            for col in df_soc_opt.columns:
                if col not in storage_const_soc_max:
                    continue

                capacity = storage_const_soc_max[col]

                df_soc_opt[col] /= capacity

            if hydro_soc_min is not None:
                hydro_soc_min_plot = hydro_soc_min.copy()

                for asset in hydro_soc_min_plot:
                    hydro_soc_min_plot[asset] /= storage_const_soc_max[asset]

            if hydro_soc_max is not None:
                hydro_soc_max_plot = hydro_soc_max.copy()

                for asset in hydro_soc_max_plot:
                    hydro_soc_max_plot[asset] /= storage_const_soc_max[asset]

        else:
            hydro_soc_min_plot = hydro_soc_min
            hydro_soc_max_plot = hydro_soc_max

        # conditional def. of the axes and subplots
        hydro_res_with_nonconst_zero_soc = \
            [asset for asset in hydro_reservoirs if not is_column_constant(df=df_soc_opt, col_name=asset, value=0.0)]
        other_stocks_with_nonconst_zero_soc = \
            [asset for asset in other_stocks if not is_column_constant(df=df_soc_opt, col_name=asset, value=0.0)]
        has_hydro_res_to_plot = len(hydro_res_with_nonconst_zero_soc) > 0
        has_other_stocks_to_plot = len(other_stocks_with_nonconst_zero_soc) > 0

        if not has_hydro_res_to_plot and not has_other_stocks_to_plot:
            msg_suffix = f" for country {country}" if country != "europe" else ""
            logging.warning(f"No storage units available to plot stock SoC{msg_suffix}")
            return

        if has_hydro_res_to_plot and has_other_stocks_to_plot:

            fig, (ax_hydro_res, ax_other) = plt.subplots(2, 1, figsize=(16, 10),
                                                         sharex=True, constrained_layout=True)

        elif has_hydro_res_to_plot:

            fig, ax_hydro_res = plt.subplots(1, 1, figsize=(16, 5), constrained_layout=True)
            ax_other = None

        else:  # has_other

            fig, ax_other = plt.subplots(1, 1, figsize=(16, 5), constrained_layout=True)
            ax_hydro_res = None

        # common plot elements
        snapshots = df_soc_opt.index
        y_label = "SOC (p.u.)" if plot_in_pu else "SOC (MWh)"

        # Hydro reservoirs
        if ax_hydro_res is not None:
            used_colors = []
            for asset in hydro_res_with_nonconst_zero_soc:

                line = ax_hydro_res.plot(snapshots, df_soc_opt[asset],
                                         label=rm_country_from_prod_unit_name(asset))[0]
                color, used_colors = set_stock_color(prod_type=ProdTypeNames.hydro_reservoir,
                                                     per_pt_colors=plot_params_agg_pt.per_case_color,
                                                     used_colors=used_colors)

                if hydro_soc_min_plot is not None and asset in hydro_soc_min_plot:
                    ax_hydro_res.plot(snapshots, hydro_soc_min_plot[asset],
                                      linestyle="--", color=color, alpha=0.8)

                if hydro_soc_max_plot is not None and asset in hydro_soc_max_plot:
                    ax_hydro_res.plot(snapshots, hydro_soc_max_plot[asset],
                                      linestyle="--", color=color, alpha=0.8)

            ax_hydro_res.set_title("Hydro reservoir SOC")
            ax_hydro_res.set_ylabel(y_label)
            ax_hydro_res.grid(True, alpha=0.3)

            if hydro_reservoirs:
                ax_hydro_res.legend(fontsize=8, ncol=2)

        # Other storage units
        if ax_other is not None:
            used_colors = []

            for asset in other_stocks_with_nonconst_zero_soc:
                prod_type = get_prod_type_from_unit_name(asset)
                color, used_colors = set_stock_color(prod_type=prod_type,
                                                     per_pt_colors=plot_params_agg_pt.per_case_color,
                                                     used_colors=used_colors)

                line = ax_other.plot(snapshots, df_soc_opt[asset],
                                     label=rm_country_from_prod_unit_name(asset), color=color)[0]

                if not plot_in_pu:
                    ax_other.axhline(y=storage_const_soc_max[asset], color=color, linestyle="--", alpha=0.8)

            ax_other.set_title("Other storage SOC")
            ax_other.set_ylabel(y_label)
            ax_other.set_xlabel("Time")
            ax_other.grid(True, alpha=0.3)

            if other_stocks:
                ax_other.legend(fontsize=8, ncol=2)

        fig.suptitle(f"Storage state of charge - {country}", fontsize=14)

        # text to indicate stocks with always SoC = 0
        const_zero_soc_assets = (
                [asset for asset in hydro_reservoirs if is_column_constant(df=df_soc_opt, col_name=asset, value=0.0)]
                + [asset for asset in other_stocks if is_column_constant(df=df_soc_opt, col_name=asset, value=0.0)]
        )
        if const_zero_soc_assets:
            # remove country trigrams to simplify block of text
            const_zero_soc_assets = [rm_country_from_prod_unit_name(asset) for asset in const_zero_soc_assets]
            txt = (
                    "Assets not shown (SOC = 0 for all snapshots):\n"
                    + "\n".join(sorted(const_zero_soc_assets))
            )

            fig.text(0.99, 0.01, txt, ha="right", va="bottom", fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
                     )

        output_file = get_output_figure(
            fig_name=FigNamesPrefix.stock_soc,
            country=country,
            year=year,
            climatic_year=climatic_year,
            start_horizon=start_horizon,
            toy_model_output=toy_model_output,
            output_tag=output_tag,
        )

        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

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
            country_suffix = f" for country {country}" if not country == "europe" else ""
            logging.info(f"Plot marginal price{country_suffix}")
            sde_dual_var_opt_plot = set_col_order_for_plot(df=self.sde_dual, cols_ordered=plot_params_zone.order)
            sde_dual_var_opt_plot.plot.line(figsize=(8, 3), ylabel='Euro per MWh',
                                            color=plot_params_zone.per_case_color)
            plt.tight_layout()
            plt.savefig(get_output_figure(fig_name=FigNamesPrefix.prices, country=country, year=year,
                                          climatic_year=climatic_year, start_horizon=start_horizon,
                                          toy_model_output=toy_model_output, output_tag=output_tag)
                        )
            plt.close()

    def plot_link_capa_saturation(self, link_capacities: dict[str, float], year: int, climatic_year: int,
                                  start_horizon: datetime, output_tag: str = None, n_time_blocks: int = 4,
                                  saturation_threshold: float = 0.999,
                                  time_label_mode: Literal["blocks", "dates"] = "blocks"):
        """
        Plot a 3-level sunburst showing transmission link capacity saturation.

        Levels:
            1. Link
            2. Saturation direction
            3. Time block

        Parameters
        ----------
        link_capacities :
            Mapping {link_name: capacity}.
        year :
            Of UC simulated
        climatic_year :
            Idem
        start_horizon :
            Of UC period simulated
        output_tag :
            On output folder, to be more explicit
        n_time_blocks :
            Number of temporal aggregation blocks. Ex: if 4, then number saturation occurrences will be considered for
            blocks [0AM, 6AM], [6AM, 0PM], [0PM, 6PM], [6PM, 0AM next day]
        saturation_threshold :
            Saturation threshold as a fraction of nominal capacity.
        time_label_mode :
            "blocks" -> Block 1, ..., Block N
            "dates"  -> Explicit time interval labels
        """
        logging.info('Plot link capacity saturation (stat. aggregated view)')
        df_link_flow_opt_direct = self.link_flow_direct

        if len(df_link_flow_opt_direct) == 0:
            raise ValueError("Link flow df is empty (after UC optim.) -> plot not possible")

        records = []

        n_snapshots = len(df_link_flow_opt_direct)
        block_edges = np.linspace(
            0,
            n_snapshots,
            n_time_blocks + 1,
            dtype=int,
        )

        for link_name, capacity in link_capacities.items():

            if link_name not in df_link_flow_opt_direct.columns:
                continue

            flow = df_link_flow_opt_direct[link_name]

            origin_bus, destination_bus = get_bus_from_link_name(link_name=link_name)

            sat_pos = flow >= saturation_threshold * capacity
            sat_neg = flow <= -saturation_threshold * capacity

            directions = {
                f"{origin_bus} → {destination_bus}": sat_pos,
                f"{destination_bus} → {origin_bus}": sat_neg,
            }

            for direction, sat_mask in directions.items():

                for block_id in range(n_time_blocks):

                    start = block_edges[block_id]
                    end = block_edges[block_id + 1]

                    block_mask = sat_mask.iloc[start:end]

                    proportion_total_time = block_mask.sum() / n_snapshots

                    if proportion_total_time == 0:
                        continue

                    if time_label_mode == "blocks":
                        time_label = f"Block {block_id + 1}"

                    elif time_label_mode == "dates":

                        start_ts = df_link_flow_opt_direct.index[start]
                        end_ts = df_link_flow_opt_direct.index[end - 1]

                        time_label = (
                            f"{start_ts:%Y-%m-%d}"
                            f" → "
                            f"{end_ts:%Y-%m-%d}"
                        )

                    else:
                        raise ValueError(
                            "time_label_mode must be 'blocks' or 'dates'."
                        )

                    records.append(
                        {
                            "link": link_name,
                            "direction": direction,
                            "time_block": time_label,
                            "proportion": proportion_total_time,
                        }
                    )

        sunburst_df = pd.DataFrame(records)

        if sunburst_df.empty:
            raise ValueError("No saturation detected.")

        fig = px.sunburst(
            sunburst_df,
            path=["link", "direction", "time_block"],
            values="proportion",
        )

        fig.update_traces(
            textinfo="label+percent parent",
            branchvalues="total",
        )

        fig.update_layout(
            title="Transmission Link Saturation Breakdown",
        )

        fig.write_image(
            get_output_figure(
                fig_name=FigNamesPrefix.link_capa_saturation,
                country="europe",
                year=year,
                climatic_year=climatic_year,
                start_horizon=start_horizon,
                output_tag=output_tag,
            )
        )

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
        logging.info(f'Save - all but Storage assets - optimal dispatch decisions to csv '
                     f'file {format_relative_path_for_log(path=opt_p_csv_file)}')
        df_prod_opt = self.prod
        df_prod_opt = process_data_to_csv(data_type="prod", df=df_prod_opt, rename_snapshot_col=rename_snapshot_col,
                                          order_cols=lexico_order_prod_cols)
        df_prod_opt.to_csv(opt_p_csv_file)
        # then storage assets decisions
        storage_opt_dec_csv_file = \
            get_storage_opt_dec_file(country=country, year=year, climatic_year=climatic_year,
                                     start_horizon=start_horizon, toy_model_output=toy_model_output,
                                     output_tag=output_tag)
        logging.info(f'Save Storage optimal decisions to csv '
                     f'file {format_relative_path_for_log(path=storage_opt_dec_csv_file)}')
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
        logging.info(f'Save link flow optimal decisions to csv '
                     f'file {format_relative_path_for_log(path=link_flow_opt_dec_csv_file)}')
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
        marginal_prices_csv_file = get_marginal_prices_file(country=country, year=year,
                                                            climatic_year=climatic_year,
                                                            start_horizon=start_horizon,
                                                            toy_model_output=toy_model_output,
                                                            output_tag=output_tag)
        logging.info(f'Save marginal prices decisions to .csv file '
                     f'{format_relative_path_for_log(path=marginal_prices_csv_file)}')

        df_sde_dual_var_opt = self.sde_dual
        if rename_snapshot_col:
            df_sde_dual_var_opt.index.name = OUTPUT_DATE_COL
        # do NOT cast this df, given that price values can be accurate at some decimals
        # -> may be useful to observe the correspondence with (input) marginal cost values
        df_sde_dual_var_opt.to_csv(marginal_prices_csv_file)

    def save_link_capa_dual_vars_to_csv(self, year: int, climatic_year: int, start_horizon: datetime,
                                        output_tag: str = None):
        link_capa_duals_csv_file = get_link_capa_dual_vars_file(year=year, climatic_year=climatic_year,
                                                                start_horizon=start_horizon, output_tag=output_tag)
        logging.info(f'Save link capacity dual variables to .csv file '
                     f'{format_relative_path_for_log(path=link_capa_duals_csv_file)}')

        df_link_capa_duals = create_df_from_dict(data=self.link_capa_dual, key_col_name="link",
                                                 value_col_name="dual_variable")
        df_link_capa_duals.to_csv(link_capa_duals_csv_file, index=False)

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
