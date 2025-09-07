from dataclasses import dataclass

from long_term_uc.common.constants.prod_types import ProdTypeNames

@dataclass
class DatatypesNames:
    demand: str = 'demand'
    capa_factor: str = 'res_capa-factors'
    installed_capa: str = 'generation_capas'
    interco_capa: str = 'intercos_capas'


DATATYPE_NAMES = DatatypesNames()
PROD_TYPES_PER_DT = {DATATYPE_NAMES.capa_factor: 
                     [ProdTypeNames.csp_no_storage, 'lfsolarpv', ProdTypeNames.wind_offshore, ProdTypeNames.wind_onshore],
                     DATATYPE_NAMES.installed_capa: 
                     [ProdTypeNames.batteries, ProdTypeNames.biofuel, ProdTypeNames.coal, ProdTypeNames.hard_coal,
                      ProdTypeNames.lignite, ProdTypeNames.demand_side_response, ProdTypeNames.gas,
                      ProdTypeNames.hydro_pondage, ProdTypeNames.hydro_pump_storage_closed,
                      ProdTypeNames.hydro_pump_storage_open, ProdTypeNames.hydro_reservoir,
                      ProdTypeNames.hydro_ror, ProdTypeNames.nuclear, ProdTypeNames.oil,
                      ProdTypeNames.solar_pv, ProdTypeNames.solar_thermal,
                      ProdTypeNames.wind_offshore, ProdTypeNames.wind_onshore]
                      }
UNITS_PER_DT = {DATATYPE_NAMES.demand: 'mw', DATATYPE_NAMES.capa_factor: '%'}
