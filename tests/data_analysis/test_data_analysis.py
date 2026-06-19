from code.common.constants.data_analysis_types import AnalysisTypes

from code.common.constants.datatypes import DatatypesNames

from code.my_little_europe_data_analysis import run


def test_plot_unique_case(analysis_type: str = AnalysisTypes.plot, data_type: str = DatatypesNames.demand,
                          country: str = "france", year: int = 2025, climatic_year: int = 1985):
    data_analysis_json = "_".join([analysis_type, data_type, country, f"ty{year}", f"cy{climatic_year}"]) + ".json"
    run(json_param_filepath=data_analysis_json)


def test_data_analysis():
    # cases with plot test (plot, data type, country, year, climatic year)
    plot_cases = [("plot", "demand", "france", 2025, 1985), ("plot", "res_capa-factors", "italy", 2025, 1989)]
    for case in plot_cases:
        test_plot_unique_case(*case)
