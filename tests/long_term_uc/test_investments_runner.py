from datetime import datetime
import json
import os
import tempfile
import pandas as pd
import pytest

from src.common.constants.usage_params_json import EnvPhaseNames
from src.include_runner.investments_runner import (
    AVAILABLE_COUNTRIES,
    normalize_country_name,
    normalize_asset_name,
    parse_capacity_value,
    parse_investment_file,
    parse_country_json_file,
    load_country_capacities,
    collect_investment_files,
    write_country_json,
    update_europe_params,
    display_multi_country_summary,
    display_summary,
    calc_crf,
    calc_annualized_capex_per_mw,
    compute_country_annualized_capex,
)
from src.utils.read import read_and_check_uc_run_params, read_usage_params


class TestInvestmentsRunnerNormalization:
    def test_normalize_country_name_valid_and_aliases(self):
        assert normalize_country_name("france") == "france"
        assert normalize_country_name("FR") == "france"
        assert normalize_country_name("allemagne") == "germany"
        assert normalize_country_name("de") == "germany"
        assert normalize_country_name("italia") == "italy"
        assert normalize_country_name("espagne") == "iberian-peninsula"
        assert normalize_country_name("spain") == "iberian-peninsula"
        assert normalize_country_name("pologne") == "poland"
        assert normalize_country_name("belgique") == "benelux"
        assert normalize_country_name("suede") == "scandinavia"

    def test_normalize_country_name_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown country"):
            normalize_country_name("atlantis")

    def test_normalize_asset_name_valid_and_aliases(self):
        assert normalize_asset_name("nuclear") == "nuclear"
        assert normalize_asset_name("nucleaire") == "nuclear"
        assert normalize_asset_name("nuc") == "nuclear"
        assert normalize_asset_name("wind_onshore") == "wind_onshore"
        assert normalize_asset_name("eolien_terrestre") == "wind_onshore"
        assert normalize_asset_name("solar_pv") == "solar_pv"
        assert normalize_asset_name("pv") == "solar_pv"
        assert normalize_asset_name("batteries") == "batteries"
        assert normalize_asset_name("battery") == "batteries"
        assert normalize_asset_name("gaz") == "gas"

    def test_normalize_asset_name_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown asset type"):
            normalize_asset_name("antimatter_reactor")

    def test_parse_capacity_value(self):
        assert parse_capacity_value("60000") == 60000.0
        assert parse_capacity_value("60000 MW") == 60000.0
        assert parse_capacity_value("1234.5") == 1234.5

        with pytest.raises(ValueError, match="Capacity cannot be negative"):
            parse_capacity_value("-500")

        with pytest.raises(ValueError, match="Invalid capacity value"):
            parse_capacity_value("abc")


class TestInvestmentsRunnerParsing:
    def test_parse_investment_file_with_comments_and_separators(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
            tmp.write("# Test country file\n")
            tmp.write("// Another comment\n")
            tmp.write("country: france\n")
            tmp.write("nuclear: 61000\n")
            tmp.write("solar_pv = 35000\n")
            tmp.write("wind_onshore 25000 MW\n")
            tmp_path = tmp.name

        try:
            country, capacities = parse_investment_file(tmp_path)
            assert country == "france"
            assert capacities["nuclear"] == 61000.0
            assert capacities["solar_pv"] == 35000.0
            assert capacities["wind_onshore"] == 25000.0
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_collect_investment_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "france.txt")
            f2 = os.path.join(tmpdir, "germany.txt")
            f_other = os.path.join(tmpdir, "notes.pdf")
            for f in (f1, f2, f_other):
                with open(f, "w") as fp:
                    fp.write("dummy")

            collected = collect_investment_files(tmpdir)
            assert f1 in collected
            assert f2 in collected
            assert f_other not in collected

    def test_parse_country_json_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            data = {
                "team": "france",
                "capacities_tb_overwritten": {
                    "france": {
                        "nuclear": 61000.0,
                        "solar_pv": 35000.0
                    }
                }
            }
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            country, capacities = parse_country_json_file(tmp_path)
            assert country == "france"
            assert capacities["nuclear"] == 61000.0
            assert capacities["solar_pv"] == 35000.0
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_parse_country_json_file_fallback_to_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "germany.json")
            data = {
                "capacities_tb_overwritten": {
                    "germany": {
                        "wind_onshore": 65000.0,
                        "gas": 25000.0
                    }
                }
            }
            with open(tmp_path, "w") as fp:
                json.dump(data, fp)

            country, capacities = parse_country_json_file(tmp_path)
            assert country == "germany"
            assert capacities["wind_onshore"] == 65000.0
            assert capacities["gas"] == 25000.0

    def test_load_country_capacities_routes_correctly(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as t1:
            t1.write("country: poland\ncoal: 20000\n")
            txt_path = t1.name

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as t2:
            json.dump({
                "team": "italy",
                "capacities_tb_overwritten": {
                    "italy": {"solar_pv": 30000.0}
                }
            }, t2)
            json_path = t2.name

        try:
            c1, cap1 = load_country_capacities(txt_path)
            assert c1 == "poland"
            assert cap1["coal"] == 20000.0

            c2, cap2 = load_country_capacities(json_path)
            assert c2 == "italy"
            assert cap2["solar_pv"] == 30000.0
        finally:
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            if os.path.isfile(json_path):
                os.remove(json_path)

    def test_collect_investment_files_with_json_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f_txt = os.path.join(tmpdir, "france.txt")
            f_json = os.path.join(tmpdir, "germany.json")
            for f in (f_txt, f_json):
                with open(f, "w") as fp:
                    fp.write("{}")

            txt_files = collect_investment_files(tmpdir, extension=".txt")
            assert f_txt in txt_files
            assert f_json not in txt_files

            json_files = collect_investment_files(tmpdir, extension=".json")
            assert f_json in json_files
            assert f_txt not in json_files


class TestInvestmentsRunnerEuropeModeExclusivity:
    def test_update_europe_params_enforces_europe_mode(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json.dump({"mode": "solo", "team": "france"}, tmp)
            tmp_path = tmp.name

        try:
            update_europe_params(mode="europe", team="france", europe_params_path=tmp_path)
            with open(tmp_path, "r") as fp:
                data = json.load(fp)
            assert data["mode"] == "europe"
            assert data["uc_period_start"] == "1900/1/1"
            assert data["uc_period_end"] == "1901/1/1"
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_update_europe_params_rejects_solo_mode(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json.dump({"mode": "europe", "team": "france"}, tmp)
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="The runner exclusively operates in 'europe' mode"):
                update_europe_params(mode="solo", europe_params_path=tmp_path)
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_write_country_json_sets_proper_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = write_country_json(
                country="france",
                capacities={"nuclear": 60000.0, "gas": 5000.0},
                countries_dir=tmpdir,
            )
            assert os.path.isfile(json_path)
            with open(json_path, "r") as fp:
                data = json.load(fp)
            assert data["team"] == "france"
            assert "nuclear" in data["selected_prod_types"]["france"]
            assert "gas" in data["selected_prod_types"]["france"]
            assert "failure" in data["selected_prod_types"]["france"]
            assert data["capacities_tb_overwritten"]["france"]["nuclear"] == 60000.0

    def test_display_multi_country_summary_with_trigrams_and_updated_marker(self, capsys):
        summary_mock = {
            "total_cost": 12500.0,
            "total_operational_cost": 9800.0,
            "total_co2_emissions": 45.0,
            "per_country_ens": {"fra": 0.0, "ger": 150.0},
            "per_country_n_failure_hours": {"fra": 0.0, "ger": 2.5},
            "per_country_total_cost": {"fra": 3200.0, "ger": 4100.0},
            "per_country_co2_emissions": {"fra": 1.5, "ger": 18.0},
            "per_country_production": {"fra": 400000.0, "ger": 450000.0},
        }

        display_multi_country_summary(
            summary_data=summary_mock,
            countries=["france", "germany"],
            updated_countries=["france"],
        )

        captured = capsys.readouterr().out
        assert "RESULTS OF THE EUROPEAN INTERCONNECTED SIMULATION" in captured
        assert "COST(€/MWh)" in captured
        assert "france *" in captured
        assert "germany" in captured
        assert "TOTAL EUROPE" in captured
        assert "LOSS OF LOAD / FAILURE DETECTED in: germany" in captured
        assert "Countries updated by the runner" in captured
        assert "Average electricity cost" in captured

    def test_display_summary_delegates_to_multi_country(self, capsys):
        summary_mock = {
            "total_cost": 1000.0,
            "total_operational_cost": 800.0,
            "total_co2_emissions": 10.0,
            "per_country_ens": {"fra": 0.0},
            "per_country_n_failure_hours": {"fra": 0.0},
            "per_country_total_cost": {"fra": 1000.0},
            "per_country_co2_emissions": {"fra": 10.0},
        }

        display_summary(summary_data=summary_mock, country="france")
        captured = capsys.readouterr().out
        assert "RESULTS OF THE EUROPEAN INTERCONNECTED SIMULATION" in captured
        assert "france" in captured

    def test_read_and_check_uc_run_params_in_europe_mode(self):
        usage_params = read_usage_params()
        eraa_descr, uc_params = read_and_check_uc_run_params(
            phase_name=EnvPhaseNames.multizones_uc_model,
            usage_params=usage_params,
        )
        assert uc_params is not None
        assert uc_params.mode == "europe"

    def test_full_year_horizon_date_range(self):
        start = datetime(1900, 1, 1)
        end = datetime(1901, 1, 1)
        target_year = 2033
        end_year = target_year + (end.year - start.year)
        horizon = pd.date_range(
            start=start.replace(year=target_year),
            end=end.replace(year=end_year),
            freq="h",
        )
        # 365 days * 24 hours = 8760 snapshots
        assert len(horizon[:-1]) == 8760
        assert horizon[0] == pd.Timestamp("2033-01-01 00:00:00")
        assert horizon[-2] == pd.Timestamp("2033-12-31 23:00:00")


class TestInvestmentsRunnerCapex:
    def test_calc_crf_annuity_factor(self):
        # 4% WACC, 60 years (nuclear)
        crf_nuclear = calc_crf(wacc=0.04, lifetime=60)
        assert 0.0441 < crf_nuclear < 0.0443

        # 5% WACC, 20 years (solar PV)
        crf_solar = calc_crf(wacc=0.05, lifetime=20)
        assert 0.0801 < crf_solar < 0.0804

        # Zero WACC edge case
        assert calc_crf(wacc=0.0, lifetime=20) == 0.05

    def test_compute_country_annualized_capex(self):
        capacities = {
            "nuclear": 1000.0,     # 1000 MW * 4.5 M€/MW = 4500 M€ overnight
            "solar_pv": 2000.0,    # 2000 MW * 0.5 M€/MW = 1000 M€ overnight
            "failure": 100000.0,   # failure should have 0 CAPEX
        }
        overnight_eur, annualized_eur = compute_country_annualized_capex(capacities=capacities)
        assert overnight_eur == 5500e6  # 5.5 B€

        # Annualized: nuclear (1000 * 4500e3 * ~0.0442) + solar (2000 * 500e3 * ~0.0802)
        annualized_m_yr = annualized_eur / 1e6
        assert 270.0 < annualized_m_yr < 290.0


class TestMainRunnerCli:
    def test_main_runner_dry_run_txt_mode(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "txt", "--dry-run", "--no-summary"])
        # Should execute successfully without throwing exceptions
        main()

    def test_main_runner_dry_run_json_mode(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "json", "--dry-run", "--no-summary"])
        main()

    def test_main_runner_dry_run_source_json(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "--source", "json", "--dry-run", "--no-summary"])
        main()


