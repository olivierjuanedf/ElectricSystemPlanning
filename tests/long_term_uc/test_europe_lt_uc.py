import logging
import os
from pathlib import Path

import pytest

from my_little_europe_lt_uc import run
from src.common.long_term_uc_io import set_full_lt_uc_output_folder, OutputFolderNames
from src.utils.dir_utils import get_files_from_prefix, remove_folder

EUR_COUNTRY = "europe"
TEST_INPUT_DIR = os.path.join(
    os.path.dirname(__file__), "input"
)  # To keep path of input relative to file

JSON_TB_MODIF_INPUT_FILES = {"elec-eur_params_tb-modif_1.json": EUR_COUNTRY}


@pytest.mark.parametrize("json_params_tb_modif_file, country", JSON_TB_MODIF_INPUT_FILES.items())
def test_run_europe_lt_uc(json_params_tb_modif_file: str, country: str):
    uc_params_json = os.path.join(TEST_INPUT_DIR, json_params_tb_modif_file)
    logging.disable(logging.CRITICAL)
    run(json_params_filepath=uc_params_json)
    lt_uc_output_folder_data = set_full_lt_uc_output_folder(folder_type=OutputFolderNames.data, country=country,
                                                            toy_model_output=not (country == EUR_COUNTRY))
    uc_summary_prefix = f"uc-summary_{country}_"
    uc_summary_files = get_files_from_prefix(folder=lt_uc_output_folder_data, file_prefix=uc_summary_prefix)
    assert len(uc_summary_files) == 1, f"UC summary file, with prefix  {uc_summary_prefix} not created"
    # remove (recursively) LT UC country output folder, containing both output data and figs subfolders
    # -> to avoid conflict with following tests
    country_output_folder = Path(lt_uc_output_folder_data).parent
    remove_folder(folder_path=country_output_folder)
