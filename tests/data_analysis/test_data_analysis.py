# import os
#
# from my_little_europe_data_analysis import run
#
#
# INPUT_DIR = os.path.join(
#     os.path.dirname(__file__), "input"
# )  # To keep path of input relative to file


# def test_given_da_case(json_params_filename: str):
#     json_params_filepath = os.path.join(INPUT_DIR, json_params_filename)
#     run(json_param_filepath=json_params_filepath)
#
#
# def test_data_analysis():
#     # corresp. {test idx: params for plot test (plot, data type, country, year, climatic year, aggreg. prod type,
#     # period_start, period_end, extra_params)}
#     # ATTENTION: test idx is to find the plot_{idx}.json file used for test
#     # N.B. Params are just for info, to have infos on tested cases. If last ones are not provided they are None
#     plot_cases = {1: ("plot", "demand", "france", 2025, 1985),  # demand 1 ty * cy
#                   2: ("plot", "res_capa-factors", "italy", 2025, 1989),  # idem RES CF
#                   # idem, with net demand and specific period
#                   3: ("plot", "net_demand", "france", 2025, 1985, "1900/1/15", "1900/1/29"),
#                   # duration curve, on two countries
#                   4: ("plot_duration_curve", "demand", ["france", "germany"], 2025, 1985),
#                   # idem, on two years and specific temporal period
#                   5: ("plot_duration_curve", "demand", "france", [2025, 2033], 1985, "1900/1/15", "1900/1/29"),
#                   6: ("plot", "net_demand", "france", [2025, 2033], [1989, 2010, 2016]),
#                   7: ("plot", "net_demand", "france", 2025, 1985, "1900/1/15", "1900/1/29", None,
#                       [None,
#                        {"label": "res_low",
#                         "values":
#                             {"capas_aggreg_pt_with_cf":
#                                  {"wind_onshore": 10000,
#                                   "wind_offshore": 500,
#                                   "solar_pv": 10000
#                                   }
#                              }
#                         },
#                        {"label": "res_high",
#                         "values": {"capas_aggreg_pt_with_cf":
#                                        {"wind_onshore": 40000,
#                                         "wind_offshore": 10000,
#                                         "solar_pv": 40000
#                                         }
#                                    }
#                         }]),
#                   8: ("plot", "res_capa-factors", "france", [2025, 2033], [1985, 1987], ["solar_pv", "wind_offshore"],
#                       "1900/1/15", "1900/1/29"),
#                   9: ("plot_duration_curve", "net_demand", "france", 2025, [1985, 1987], ["solar_pv", "wind_onshore"],
#                       "1900/1/15", "1900/1/29"),
#                   10: ("plot_duration_curve", "net_demand", "france", 2025, 1985, "1900/1/15", "1900/1/29", "solar_pv",
#                       [None,
#                        {"label": "res_low", "values": {"capas_aggreg_pt_with_cf": {"solar_pv": 10000}}},
#                        {"label": "res_high", "values": {"capas_aggreg_pt_with_cf": {"solar_pv": 40000}}}])
#                   }
#     extract_cases = {1: ("extract", "net_demand", "france", [2025, 2033], 1985, "1900/1/15", "1900/1/29"),
#                      2: ("extract", "res_capa-factors", "france", [2025, 2033], [1985, 1987],
#                          ["solar_pv", "wind_offshore"], "1900/1/15", "1900/1/29"),
#                      3: ("extract", "net_demand", "france", [2025, 2033], 1985, "solar_pv", "1900/1/15", "1900/1/29")
#                      }
#     # concatenate different types of tests
#     tests_cases = {"plot": plot_cases, "extract": extract_cases}
#     for da_type, test_cases in tests_cases.items():
#         for case_idx in test_cases:
#             test_given_da_case(json_params_filename=f"{da_type}_{case_idx}.json")


import os
from pathlib import Path

import pytest

from code.common.long_term_uc_io import OUTPUT_FOLDER
from code.utils.dir_utils import find_project_root
from my_little_europe_data_analysis import run

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
# TODO: get data_analysis from global constants...
OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_FOLDER, "data_analysis")

# TEST_FILES = [
#     *[f"plot_{i}.json" for i in range(1, 11)],
#     *[f"extract_{i}.json" for i in range(1, 4)],
# ]
# mapping params -> output file to be generated; no assertion if None
TEST_OUTPUT_FILES = {"plot_1.json": "demand_france_2025_cy1985.png",
                     "plot_2.json": "res_capa-factors_solar_pv_italy_2025_cy1989_1-aggpts.png",
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
def test_run_smoke(json_param_file, expected_output):
    json_params_filepath = os.path.join(INPUT_DIR, json_param_file)

    assert os.path.exists(json_params_filepath), f"{json_param_file} not found"

    run(json_params_filepath=json_params_filepath)
    # check existence of file to be generated, and suppress it
    if expected_output is not None:
        output_path = os.path.join(OUTPUT_DIR, expected_output)
        assert os.path.exists(output_path), f"Output file {expected_output} not created"
        os.remove(output_path)
