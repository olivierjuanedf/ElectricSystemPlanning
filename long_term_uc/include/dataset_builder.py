from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pypsa
import matplotlib.pyplot as plt

from long_term_uc.common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from long_term_uc.common.error_msgs import print_errors_list
from long_term_uc.common.fuel_sources import FuelSources
from long_term_uc.common.long_term_uc_io import get_marginal_prices_file, get_network_figure, \
    get_opt_power_file, get_price_figure, get_prod_figure, get_storage_opt_dec_file, \
        get_capacity_figure, get_figure_file_named
from long_term_uc.include.plotter import PlotParams
from long_term_uc.utils.basic_utils import lexico_compar_str
from long_term_uc.utils.df_utils import rename_df_columns
from long_term_uc.utils.pypsa_utils import get_network_obj_value
from long_term_uc.common.error_msgs import print_errors_list
from long_term_uc.common.fuel_sources import FuelSources
from long_term_uc.common.long_term_uc_io import get_marginal_prices_file, get_network_figure, \
    get_opt_power_file, get_price_figure
from long_term_uc.include.plotter import PlotParams
from long_term_uc.utils.basic_utils import lexico_compar_str
from long_term_uc.utils.pypsa_utils import get_network_obj_value


@dataclass
class GenUnitsPypsaParams:
    capa_factors: str = "p_max_pu" 
    power_capa: str = "p_nom"
    min_power: str = "p_min_pu"
    marginal_cost: str = "marginal_cost"
    co2_emissions: str = "co2_emissions"  # TODO: check that aligned on PyPSA generators attribute names
    committable: str = "committable"
    max_hours: str = "max_hours"
    energy_capa: str = None

GEN_UNITS_PYPSA_PARAMS = GenUnitsPypsaParams()


@dataclass
class GenerationUnitData:
    name: str
    type: str
    carrier: str = None
    p_nom: float = None
    p_min_pu: float = None
    p_max_pu: float = None
    efficiency: float = None
    marginal_cost: float = None
    committable: bool = False
    max_hours: float = None
    cyclic_state_of_charge: bool = None

    def get_non_none_attr_names(self):
        return [key for key, val in self.__dict__.items() if val is not None]


def get_val_of_agg_pt_in_df(df_data: pd.DataFrame, prod_type_agg_col: str, 
                            agg_prod_type: str, value_col: str, static_val: bool) \
                                -> Union[np.ndarray, float]:
    if static_val:
        return df_data[df_data[prod_type_agg_col] == agg_prod_type][value_col].iloc[0]
    else:
        return np.array(df_data[df_data[prod_type_agg_col] == agg_prod_type][value_col])


UNIT_NAME_SEP = '_'


def set_country_trigram(country: str) -> str:
    return f'{country[:3].lower()}'


def get_prod_type_from_unit_name(prod_unit_name: str) -> str:
    len_country_suffix = 3 + len(UNIT_NAME_SEP)
    return prod_unit_name[len_country_suffix:]


def get_prod_type_from_unit_name(prod_unit_name: str) -> str:
    len_country_suffix = 3 + len(UNIT_NAME_SEP)
    return prod_unit_name[len_country_suffix:]


def set_gen_unit_name(country: str, agg_prod_type: str) -> str:
    country_trigram = set_country_trigram(country=country)
    return f'{country_trigram}{UNIT_NAME_SEP}{agg_prod_type}'


GEN_UNITS_DATA_TYPE = Dict[str, List[GenerationUnitData]]
PYPSA_RESULT_TYPE = Tuple[str, str]


def set_col_order_for_plot(df: pd.DataFrame, cols_ordered: List[str]) -> pd.DataFrame:
    current_df_cols = list(df.columns)
    current_df_cols_ordered = [col for col in cols_ordered if col in current_df_cols]
    df = df[current_df_cols_ordered]
    return df


def set_full_coll_for_storage_df(df: pd.DataFrame, col_suffix: str) -> pd.DataFrame:
    old_cols = list(df.columns)
    new_cols = {col: f'{col}_{col_suffix}' for col in old_cols}
    df = rename_df_columns(df=df, old_to_new_cols=new_cols)
    return df


OUTPUT_DATE_COL = 'date'


@dataclass
class PypsaModel:
    name: str
    network: pypsa.Network = None
    prod_var_opt: pd.DataFrame = None  # prod opt value that will be obtained after optim.
    sde_dual_var_opt: pd.DataFrame = None  # idem for SDE constraint dual variable
    storage_prod_var_opt: pd.DataFrame = None  # idem for Storage prod -> prod (turbining)
    storage_cons_var_opt: pd.DataFrame = None  # idem for Storage prod -> cons (pumping)
    storage_soc_opt: pd.DataFrame = None  # idem for Storage prod -> SoC (State-of-Charge)

    def init_pypsa_network(self, date_idx: pd.Index, date_range: pd.DatetimeIndex = None):
        # TODO: type date_idx, date_range
        logging.info('Initialize PyPSA network')
        self.network = pypsa.Network(snapshots=date_idx)
        if date_range is not None:
            self.network.set_snapshots(date_range[:-1])

    def add_gps_coordinates(self, countries_gps_coords: Dict[str, Tuple[float, float]]):
        logging.info('Add GPS coordinates') 
        for country, gps_coords in countries_gps_coords.items():
            country_bus_name = get_country_bus_name(country=country)
            self.network.add('Bus', name=f'{country_bus_name}', x=gps_coords[0], y=gps_coords[1])

    def add_energy_carrier(self, fuel_sources: Dict[str, FuelSources]):
        logging.info('Add energy carriers')
        for carrier in list(fuel_sources.keys()):
            self.network.add('Carrier', name=carrier, co2_emissions=fuel_sources[carrier].co2_emissions/1000)

    def add_generators(self, generators_data: Dict[str, List[GenerationUnitData]]):
        logging.info('Add generators - associated to their respective buses')
        for country, gen_units_data in generators_data.items():
            country_bus_name = get_country_bus_name(country=country)
            for gen_unit_data in gen_units_data:
                pypsa_gen_unit_dict = gen_unit_data.__dict__
                logging.debug(f'{country}, {pypsa_gen_unit_dict}')
                if pypsa_gen_unit_dict.get(GEN_UNITS_PYPSA_PARAMS.max_hours, None) is not None:
                    self.network.add('StorageUnit', bus=f'{country_bus_name}', **pypsa_gen_unit_dict, state_of_charge_initial = pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.power_capa] * pypsa_gen_unit_dict[GEN_UNITS_PYPSA_PARAMS.max_hours] * 0.8
    )
                else:
                    self.network.add('Generator', bus=f'{country_bus_name}', **pypsa_gen_unit_dict)
        # TODO: better msg with per bus list of generators/stocks (easier to be read) 
        logging.info(f'Considered generators: {list(self.network.generators.index)}')
        logging.info(f'Considered storage units: {list(self.network.storage_units.index)}')

    def add_loads(self, demand: Dict[str, pd.DataFrame]):
        logging.info('Add loads - associated to their respective buses')
        for country in demand:
            country_bus_name = get_country_bus_name(country=country)
            load_data = {'name': f'{country_bus_name}-load', 'bus': f'{country_bus_name}',
                        'carrier': 'AC', 'p_set': demand[country]['value'].values}
            self.network.add('Load', **load_data)

    def add_interco_links(self, countries: List[str], interco_capas: Dict[Tuple[str, str], float]):
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
                    links.append({'name': f'{country_origin_bus_name}-{country_dest_bus_name}_ac',
                                  'bus0': f'{country_origin_bus_name}', 'bus1': f'{country_dest_bus_name}',
                                  'p_nom': current_interco_capa, 'p_min_pu': p_min_pu, 'p_max_pu' : p_max_pu}
                                )
        if len(links_wo_capa_msg) > 0:
            print_errors_list(error_name='-> interco. links without capacity data', 
                            errors_list=links_wo_capa_msg)
        
        # add to PyPSA network
        for link in links:
            if link[GEN_UNITS_PYPSA_PARAMS.power_capa] > 0:
                self.network.add('Link', **link)

    def plot_network(self):
        self.network.plot(title=f'{self.name.capitalize()} network', color_geomap=True, jitter=0.3)
        plt.savefig(get_network_figure())
        plt.close()

    def optimize_network(self, year: int, n_countries: int, period_start: datetime, save_lp_file: bool = True) -> PYPSA_RESULT_TYPE:
        """
        Solve the optimization UC problem associated to current network
        :returns a tuple (xxx, status of resolution)
        """
        logging.info('Optimize "network" - i.e. solve associated UC problem')
        result = self.network.optimize(solver_name='highs')
        logging.info(result)
        if save_lp_file:
            save_lp_model(self.network, year=year, n_countries=n_countries, period_start=period_start)
        return result
    
    def get_prod_var_opt(self):
        self.prod_var_opt = self.network.generators_t.p
    
    def get_storage_vars_opt(self):
        self.storage_prod_var_opt = self.network.storage_units_t.p
        self.storage_cons_var_opt = self.network.storage_units_t.p_store
        self.storage_soc_opt = self.network.storage_units_t.state_of_charge

    def get_sde_dual_var_opt(self):
        self.sde_dual_var_opt = self.network.buses_t.marginal_price

    def get_opt_value(self, pypsa_resol_status: str) -> float: 
        objective_value = get_network_obj_value(network=self.network)
        logging.info(f'Optimisation resolution status is {pypsa_resol_status} with objective value (cost) = {objective_value:.2f} -> output data (resp. figures) can be generated')
        return objective_value
    
    def plot_installed_capas(self, country: str, year: int):
        country_trigram = set_country_trigram(country=country)
        # N.B. p_nom_opt is the optimized capacity (that can be also a variable in PyPSA but here...
        # not optimized - only UC problem -> values plotted correspond to the ones that can be found in input data)
        # all but failure asset capacity will be used in plot
        self.network.generators.p_nom_opt.drop(f'{country_trigram}_failure').div(1e3).plot.bar(ylabel='GW', figsize=(8, 3))
        # [Coding trick] Matplotlib can directly adapt size of figure to fit with values plotted
        plt.tight_layout()
        plt.savefig(get_capacity_figure(country=country, year=year))
        plt.close()

    def plot_opt_prod_var(self, plot_params: PlotParams, country: str, year: int, 
                          climatic_year: int, start_horizon: datetime):
        """ 
        Plot 'stack' of optimized production profiles
        """
        # sort values to get only prod of given country
        country_trigram = set_country_trigram(country=country)
        country_prod_cols = [prod_unit_name for prod_unit_name in list(self.prod_var_opt) 
                             if prod_unit_name.startswith(country_trigram)]
        current_prod_var_opt = self.prod_var_opt[country_prod_cols]
        # suppress trigram from prod unit names to simplify legend in figures
        new_prod_cols = {col: col[4:] for col in country_prod_cols}
        current_prod_var_opt = rename_df_columns(df=current_prod_var_opt, old_to_new_cols=new_prod_cols)
        current_prod_var_opt = set_col_order_for_plot(df=current_prod_var_opt, 
                                                      cols_ordered=plot_params.agg_prod_type_order)
        current_prod_var_opt.div(1e3).plot.area(subplots=False, ylabel='GW', 
                                                color=plot_params.per_agg_prod_type_color)
        plt.tight_layout()
        plt.savefig(get_prod_figure(country=country, year=year, 
                                    climatic_year=climatic_year, start_horizon=start_horizon))
        plt.close()

    def plot_failure_at_opt(self, country: str, year: int, climatic_year: int, start_horizon: datetime):
        failure_unit_name = set_gen_unit_name(country=country, agg_prod_type='failure')
        self.network.generators_t.p.div(1e3)[failure_unit_name].plot.line(subplots=False, ylabel='GW')
        plt.tight_layout()
        plt.savefig(get_figure_file_named('failure', country=country, year=year, climatic_year=climatic_year,
                                          start_horizon=start_horizon)
                            )
        plt.close()

    def plot_marginal_price(self, plot_params: PlotParams, year: int, climatic_year: int, start_horizon: datetime):
        sde_dual_var_opt_plot = set_col_order_for_plot(df=self.sde_dual_var_opt,
                                                       cols_ordered=plot_params.zone_order)
        sde_dual_var_opt_plot.plot.line(figsize=(8, 3), ylabel='Euro per MWh',
                                        color=plot_params.per_zone_color)
        plt.tight_layout()
        plt.savefig(get_price_figure(country='europe', year=year, climatic_year=climatic_year,
                                     start_horizon=start_horizon)
                                     )
        plt.close()

    def save_opt_decisions_to_csv(self, year: int, climatic_year: int, start_horizon: datetime,
                                  rename_snapshot_col: bool = True):
        # TODO: check if unique country and in this case (i) suppress country prefix in asset names; 
        # (ii) rename file with country suffix instead of europe one
        country = 'europe'
        # opt prod decisions for all but Storage assets
        opt_p_csv_file = get_opt_power_file(country=country, year=year, climatic_year=climatic_year,
                                            start_horizon=start_horizon)
        logging.info(f'Save - all but Storage assets - optimal dispatch decisions to csv file {opt_p_csv_file}')
        df_prod_opt = self.prod_var_opt
        if rename_snapshot_col:
            df_prod_opt.index.name = OUTPUT_DATE_COL
        df_prod_opt.to_csv(opt_p_csv_file)
        # then storage assets decisions
        storage_opt_dec_csv_file = \
            get_storage_opt_dec_file(country=country, year=year, climatic_year=climatic_year,
                                     start_horizon=start_horizon)
        logging.info(f'Save Storage optimal decisions to csv file {storage_opt_dec_csv_file}')
        # join the 3 Storage result dfs
        df_prod_opt = self.storage_prod_var_opt
        df_cons_opt = self.storage_cons_var_opt
        df_soc_opt = self.storage_soc_opt
        # rename first the different columns -> adding prod/cons/soc suffixes
        df_prod_opt = set_full_coll_for_storage_df(df=df_prod_opt, col_suffix='prod')
        df_cons_opt = set_full_coll_for_storage_df(df=df_cons_opt, col_suffix='cons')
        df_soc_opt = set_full_coll_for_storage_df(df=df_soc_opt, col_suffix='soc')
        df_storage_all_decs = df_prod_opt.join(df_cons_opt).join(df_soc_opt)
        if rename_snapshot_col:
            df_storage_all_decs.index.name = OUTPUT_DATE_COL
        df_storage_all_decs.to_csv(storage_opt_dec_csv_file)

    def save_marginal_prices_to_csv(self, year: int, climatic_year: int, start_horizon: datetime,
                                    rename_snapshot_col: bool = True):
        logging.info('Save marginal prices decisions to .csv file')
        marginal_prices_csv_file = get_marginal_prices_file(country='europe', year=year, 
                                                            climatic_year=climatic_year,
                                                            start_horizon=start_horizon)
        df_sde_dual_var_opt = self.sde_dual_var_opt
        if rename_snapshot_col:
            df_sde_dual_var_opt.index.name = OUTPUT_DATE_COL
        df_sde_dual_var_opt.to_csv(marginal_prices_csv_file)

    
def overwrite_gen_units_fuel_src_params(generation_units_data: GEN_UNITS_DATA_TYPE, updated_fuel_sources_params: Dict[str, Dict[str, float]]) -> GEN_UNITS_DATA_TYPE:
    for _, units_data in generation_units_data.items():
        # loop over all units in current country
        for indiv_unit_data in units_data:
            current_prod_type = get_prod_type_from_unit_name(prod_unit_name=indiv_unit_data.name)
            if current_prod_type in updated_fuel_sources_params:
                # TODO: add CO2 emissions, and merge both case? Q2OJ: how-to properly?
                if GEN_UNITS_PYPSA_PARAMS.marginal_cost in updated_fuel_sources_params[current_prod_type]:
                    indiv_unit_data.marginal_cost = updated_fuel_sources_params[current_prod_type][GEN_UNITS_PYPSA_PARAMS.marginal_cost]

        # TODO: from units data info on fuel source extract and apply updated params values
        updated_fuel_sources_params = None


def get_country_bus_name(country: str) -> str:
    return country.lower()[:3]


STORAGE_LIKE_UNITS = ['batteries', 'flexibility', 'hydro']


from itertools import product


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


def save_lp_model(network: pypsa.Network, year: int, n_countries: int, period_start: datetime):
    logging.info('Save lp model')
    import pypsa.optimization as opt
    from long_term_uc.common.long_term_uc_io import OUTPUT_DATA_FOLDER

    m = opt.create_model(network)
    # to avoid suppressing previous runs results
    run_id = np.random.randint(99)
    period_start_file = set_period_start_file(year=year, period_start=period_start)
    file_suffix = f'{n_countries}-countries_{period_start_file}_{run_id}'
    m.to_file(Path(f'{OUTPUT_DATA_FOLDER}/model_{file_suffix}.lp'))


def get_stationary_batt_opt_dec(network: pypsa.Network, countries: List[str]):
    stationary_batt_opt_dec = {}
    # for all but storages
    # network.generators_t.p
    # for storages
    # network.storage_units_t.p_dispatch
    # network.generators.loc['fra_coal'] -> info given asset
    # network.generators_t.p_set
    for generator in network.generators:
        if generator.carrier == 'flexibility':
            bus_name = generator.bus
            current_country = [country for country in countries if country.startswith(bus_name)][0]
            stationary_batt_opt_dec[current_country] = generator.p_nom_opt


def plot_uc_run_figs(network: pypsa.Network, countries: List[str], year: int, uc_period_start: datetime):
    # TODO: use this function
    import matplotlib.pyplot as plt
    logging.info('Plot generation and prices figures')

    # p_nom_opt is the optimized capacity (that can be also a variable in PyPSA...
    # but here not optimized -> values in input data plotted)
    for country in countries:
        country_bus_name = get_country_bus_name(country=country)
        network.generators.p_nom_opt.drop(f'Failure_{country_bus_name}').div(1e3).plot.bar(ylabel='GW', figsize=(8, 3))
    # [Coding trick] Matplotlib can directly adapt size of figure to fit with values plotted
    plt.tight_layout()
    plt.close()

    # # And "stack" of optimized production profiles
    # network.generators_t.p.div(1e3).plot.area(subplots=False, ylabel="GW")
    # from long_term_uc.common.long_term_uc_io import get_prod_figure, get_price_figure
    # plt.savefig(get_prod_figure(country=country, year=year, start_horizon=uc_period_start))
    # plt.tight_layout()
    # plt.close()

    # # Finally, "marginal prices" -> meaning? How can you interprete the very constant value plotted?
    # network.buses_t.marginal_price.mean(1).plot.area(figsize=(8, 3), ylabel="Euro per MWh")
    # plt.savefig(get_price_figure(country=country, year=year, start_horizon=uc_period_start))
    # plt.tight_layout()
    # plt.close()
