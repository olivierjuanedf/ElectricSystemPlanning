import os
from my_little_europe_data_analysis import run


def test_given_da_case(json_params_filename: str):
    # TODO: get path of tests as global constant
    tests_folderpath = None
    run(json_param_filepath=os.path.join(tests_folderpath, "data_analysis", "input", json_params_filename))


def test_data_analysis():
    # corresp. {test idx: params for plot test (plot, data type, country, year, climatic year)}
    # ATTENTION: test idx is to find the plot_{idx}.json file used for test
    # N.B. Params are just for info, to have infos on tested cases
    plot_cases = {1: ("plot", "demand", "france", 2025, 1985),
                  2: ("plot", "res_capa-factors", "italy", 2025, 1989),
                  3: ("plot", "net_demand", "france", 2025, 1985, "1900/1/15", "1900/1/29"),
                  4: ("plot_duration_curve", "demand", ["france", "germany"], 2025, 1985),
                  5: ("plot_duration_curve", "demand", "france", [2025, 2033], 1985, "1900/1/15", "1900/1/29"),
                  6: ("plot", "net_demand", "france", [2025, 2033], [1989, 2010, 2016]),
                  7: ("plot", "net_demand", "france", 2025, 1985, "1900/1/15", "1900/1/29",
                      [None,
                       {"label": "res_low",
                        "values":
                            {"capas_aggreg_pt_with_cf":
                                 {"wind_onshore": 10000,
                                  "wind_offshore": 500,
                                  "solar_pv": 10000
                                  }
                             }
                        },
                       {"label": "res_high",
                        "values": {"capas_aggreg_pt_with_cf":
                                       {"wind_onshore": 40000,
                                        "wind_offshore": 10000,
                                        "solar_pv": 40000
                                        }
                                   }
                        }])
                  }
    extract_cases = {1: ("extract", "net_demand", "france", [2025, 2033], 1985, "1900/1/15", "1900/1/29")}
    # concatenate different types of tests
    tests_cases = {"plot": plot_cases, "extract": extract_cases}
    for da_type, test_cases in tests_cases.items():
        for case_idx in test_cases:
            test_given_da_case(json_params_filename=f"{da_type}_{case_idx}.json")
