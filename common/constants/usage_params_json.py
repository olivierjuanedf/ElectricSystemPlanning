
from dataclasses import dataclass


@dataclass
class UsageJsonParamNames:  # by alpha. order
    allow_adding_interco_capas: str = 'allow_adding_interco_capas'
    allow_manually_adding_demand: str = 'allow_manually_adding_demand'
    allow_manually_adding_generators: str = 'allow_manually_adding_generators'
    allow_overwriting_eraa_interco_capa_vals: str = 'allow_overwriting_eraa_interco_capa_vals'
    apply_cf_techno_breakthrough: str = 'apply_cf_techno_breakthrough'
    log_level: str = 'log_level'    
    mode: str = 'mode'
    res_cf_stress_test_cy: str = 'res_cf_stress_test_cy'
    res_cf_stress_test_folder: str = 'res_cf_stress_test_folder'
    team: str = 'team'


USAGE_PARAMS_SHORT_NAMES = {
    UsageJsonParamNames.allow_adding_interco_capas: 'adding_interco_capas',
    UsageJsonParamNames.allow_manually_adding_demand: 'manually_adding_demand',
    UsageJsonParamNames.allow_manually_adding_generators: 'manually_adding_generators',
    UsageJsonParamNames.allow_overwriting_eraa_interco_capa_vals: 'overwriting_eraa_interco_capa_vals',
    UsageJsonParamNames.apply_cf_techno_breakthrough: 'apply_cf_techno_breakthrough',
    UsageJsonParamNames.log_level: 'log_level',
    UsageJsonParamNames.mode: 'mode',
    UsageJsonParamNames.res_cf_stress_test_cy: 'res_cf_stress_test_cy', 
    UsageJsonParamNames.res_cf_stress_test_folder: 'res_cf_stress_test_folder',
    UsageJsonParamNames.team: 'team'
}

