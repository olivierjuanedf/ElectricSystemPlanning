import os
from pathlib import Path
from dataclasses import dataclass

from src.utils.dir_utils import find_project_root, uniformize_path_os

# INPUT
PROJECT_ROOT_FOLDER = find_project_root(start_path=Path(__file__).resolve())
DATA_FOLDER = 'data'


@dataclass
class DtSubfolders:
    demand: str = 'demand'
    res_capa_factors: str = 'res_capa-factors'
    generation_capas: str = 'generation_capas'
    interco_capas: str = 'interco_capas'
    hydro: str = 'hydro'


@dataclass
class DtFilePrefix:
    demand: str = 'demand'
    res_capa_factors: str = 'capa_factor'
    generation_capas: str = 'generation-capa'
    interco_capas: str = 'interco-capas'


DT_FILE_PREFIX = DtFilePrefix()
DEMAND_ADDER_FILE_PREFIX = "demand-adder"
DT_SUBFOLDERS = DtSubfolders()
# TODO: more robust way to complete path from root folder
INPUT_ERAA_FOLDER = uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, DATA_FOLDER, "ERAA_2023-2"))
INPUT_FOLDER = 'input'
CODE_FOLDER = 'src'
INPUT_FUEL_SOURCES_FOLDER = uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, DATA_FOLDER, "fuel_sources"))
INPUT_ENV_PARAMS_SUBFOLDER = uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, INPUT_FOLDER, "_env_params"))
INPUT_LT_UC_SUBFOLDER = uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, INPUT_FOLDER, "long_term_uc"))
INPUT_LT_UC_COUNTRY_SUBFOLDER = uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER, "countries"))
INPUT_LT_UC_OPTIONAL_SUBFOLDER = "optional"
INPUT_FUNC_PARAMS_SUBFOLDER = (
    uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, CODE_FOLDER, "functional_params"))
)
INPUT_DATA_ANALYSIS_SUBFOLDER = f'{INPUT_FOLDER}/data_analysis'


@dataclass
class FilesFormat:
    column_sep: str = ';'
    decimal_sep: str = '.'


FILES_FORMAT = FilesFormat()


# OUTPUT
@dataclass
class OutputFolderNames:
    data: str = 'data'
    figures: str = 'figures'


# long-term UC output
OUTPUT_FOLDER = 'output'
OUTPUT_FOLDER_LT = (
    uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, OUTPUT_FOLDER, "long_term_uc"))
)
OUTPUT_FOLDER_TOY_MODEL = (
    uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, OUTPUT_FOLDER, "toy_model"))
)
OUTPUT_DATA_ANALYSIS_FOLDER = (
    uniformize_path_os(path_str=os.path.join(PROJECT_ROOT_FOLDER, OUTPUT_FOLDER, "data_analysis")))
