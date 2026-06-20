import logging
import os
from pathlib import Path

import pytest

from src.common.long_term_uc_io import OUTPUT_FOLDER
from src.utils.dir_utils import find_project_root
from my_little_europe_data_analysis import run

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_FOLDER, "data_analysis")

# mapping params -> output file to be generated; no assertion if None
TEST_OUTPUT_FILES = {"plot_1.json": "demand_france_2025_cy1985.png",
                     "plot_2.json": "res_capa-factors_solar_pv_italy_2025_cy1989.png",
                     "plot_3.json": "net_demand_france_2025_cy1985.png",
                     "plot_4.json": "demand_fra-ger_2025_cy1985_duration_curve.png",
                     "plot_5.json": "demand_france_2025-33_cy1985_duration_curve.png",
                     "plot_6.json": "net_demand_france_2025-33_3-clim-years.png",
                     "plot_7.json": "net_demand_france_2025_cy1985_2-extraparams.png",
                     "plot_8.json": None,  # ??
                     "plot_9.json": None,  # ??
                     "plot_10.json": None,  # ??
                     "extract_1.json": None,
                     "extract_2.json": None,
                     "extract_3.json": None
                     }


# @pytest.mark.parametrize("json_file", TEST_FILES)
@pytest.mark.parametrize("json_param_file, expected_output", TEST_OUTPUT_FILES.items())
def test_run_data_analysis(json_param_file, expected_output):
    print(f"Run data analysis test with JSON file {json_param_file}")
    json_params_filepath = os.path.join(INPUT_DIR, json_param_file)

    assert os.path.exists(json_params_filepath), f"{json_param_file} not found"

    logging.disable(logging.CRITICAL)  # do not output all logs <= CRITICAL
    run(json_params_filepath=json_params_filepath)
    # check existence of file to be generated, and suppress it
    if expected_output is not None:
        output_path = os.path.join(OUTPUT_DIR, expected_output)
        assert os.path.exists(output_path), f"Output file {expected_output} not created"
        os.remove(output_path)
