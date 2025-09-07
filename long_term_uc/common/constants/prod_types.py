from dataclasses import dataclass


@dataclass
class ProdTypeNames:  # by alpha. order
    batteries: str = 'batteries'
    biofuel: str = 'biofuel'
    coal: str = 'coal'
    csp_no_storage: str = 'csp_nostorage'
    demand_side_response: str = 'demand_side_response_capacity'  # TODO: why capa in this name?
    gas: str = 'gas'
    hard_coal: str = 'hard_coal'
    hydro_pondage: str = 'hydro_pondage'
    hydro_pump_storage_closed: str = 'hydro_pump_storage_closed_loop'
    hydro_pump_storage_open: str = 'hydro_pump_storage_open_loop'
    hydro_reservoir: str = 'hydro_reservoir'
    hydro_ror: str = 'hydro_run_of_river'
    lignite: str = 'lignite'
    nuclear: str = 'nuclear'
    oil: str = 'oil'
    others_non_renewable: str = 'others_non-renewable'
    others_renewable: str = 'others_renewable'
    solar_pv: str = 'solar_(photovoltaic)'
    solar_pv_in_cf_data: str = 'lfsolarpv'
    solar_thermal: str = 'solar_(thermal)'
    wind_offshore: str = 'wind_offshore'
    wind_onshore: str = 'wind_onshore'
