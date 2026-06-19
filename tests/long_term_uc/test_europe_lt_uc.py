import os

from my_little_europe_lt_uc import run


INPUT_DIR = os.path.join(
    os.path.dirname(__file__), "input"
)  # To keep path of input relative to file


def test_europe_lt_uc():
    uc_params_json = os.path.join(INPUT_DIR, "elec-eur_params_tb-modif_1.json")
    run(json_params_filepath=uc_params_json)
