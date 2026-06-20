import logging
import os

import pytest

from src.common.long_term_uc_io import OUTPUT_DATA_ANALYSIS_FOLDER
from my_little_europe_data_analysis import run

TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")

# mapping params -> output file to be generated; no assertion if None
TEST_OUTPUT_FILES = {"plot_1.json": "demand_france_2025_cy1985.png",
                     # 2 -> PB: name generated res_capa-factors_italy_2025_cy1989_3-aggpts.png...
                     "plot_2.json": "res_capa-factors_solar_pv_italy_2025_cy1989.png",
                     "plot_3.json": "net_demand_france_2025_cy1985.png",
                     "plot_4.json": "demand_fra-ger_2025_cy1985_duration_curve.png",
                     "plot_5.json": "demand_france_2025-33_cy1985_duration_curve.png",
                     "plot_6.json": "net_demand_france_2025-33_3-clim-years.png",
                     # TODO: Q in this case 7 extra-params are [null -> ref, low, high RES capas]
                     #  -> rename suffix as 3-extraparams?
                     "plot_7.json": "net_demand_france_2025_cy1985_2-extraparams.png",
                     "plot_8.json": "res_capa-factors_france_2025-33_cy1985-87_2-aggpts.png",
                     "plot_9.json": "net_demand_incl_2-aggpts_france_2025_cy1985-87_duration_curve.png",
                     # TODO: Q idem above for 7
                     "plot_10.json": "net_demand_incl_solar_pv_france_2025_cy1985_2-extraparams_duration_curve.png",
                     "extract_1.json": "net_demand_france_2025-33_cy1985_1-15to28.csv",
                     "extract_2.json": "res_capa-factors_france_2025-33_cy1985-87_2-aggpts_1-15to28.csv",
                     "extract_3.json": "net_demand_incl_solar_pv_france_2025-33_cy1985_1-15to28.csv"
                     }


@pytest.mark.parametrize("json_param_file, expected_output", TEST_OUTPUT_FILES.items())
def test_run_data_analysis(json_param_file, expected_output):
    json_params_filepath = os.path.join(TEST_INPUT_DIR, json_param_file)

    assert os.path.exists(json_params_filepath), f"{json_param_file} not found"

    logging.disable(logging.CRITICAL)  # do not output all logs <= CRITICAL
    run(json_params_filepath=json_params_filepath)
    # check existence of file to be generated, and suppress it
    if expected_output is not None:
        output_path = os.path.join(OUTPUT_DATA_ANALYSIS_FOLDER, expected_output)
        assert os.path.exists(output_path), f"Output file {expected_output} not created"
        os.remove(output_path)
