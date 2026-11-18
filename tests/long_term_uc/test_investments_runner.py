from datetime import datetime
import json
import os
import tempfile
import pandas as pd
import pytest

from src.common.constants.usage_params_json import EnvPhaseNames
from src.include_runner.investments_runner import (
    AVAILABLE_COUNTRIES,
    DEFAULT_PERIOD_START,
    DEFAULT_PERIOD_END,
    TEST_PERIOD_START,
    TEST_PERIOD_END,
    resolve_simulation_period,
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
    extract_europe_id,
    extract_country_and_europe,
    group_files_by_europe,
    group_files_by_europe_with_sources,
    find_new_summary_file,
    extract_simulation_metrics,
    compute_europe_comparison,
    format_comparison_table,
    display_europe_comparison,
    save_comparison_results,
    load_simulation_history,
    compute_simulation_evolutions,
    format_simulation_evolutions_table,
    display_simulation_evolutions,
    save_simulation_evolutions,
    clear_runner_outputs,
    load_eraa_country_capacities,
    format_installed_capacities_table,
    display_installed_capacities,
    compute_capacity_comparison_with_eraa,
    format_capacity_comparison_table,
    get_selected_climatic_year,
    get_period_tag,
    get_eraa_baseline_cache_path,
    get_or_run_eraa_baseline,
    compute_four_way_comparison,
    format_four_way_comparison_table,
    load_all_simulation_runs,
    format_incremental_capacities_trajectory_table,
    format_incremental_capacities_summary_table,
    save_incremental_capacities_trajectory,
    get_tech_display_name,
    sync_country_json_files,
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
        update_europe_params(mode="europe")
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

    def test_main_runner_dry_run_dual_europe(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "--dry-run", "--no-summary"])
        main()

    def test_main_runner_filter_europe1(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "--europe", "europe1", "--dry-run", "--no-summary"])
        main()

    def test_main_runner_filter_europe2(self, monkeypatch):
        import sys
        from main_runner import main

        monkeypatch.setattr(sys, "argv", ["main_runner.py", "--europe", "europe2", "--dry-run", "--no-summary"])
        main()


class TestDualEuropeParsing:
    def test_extract_europe_id_from_filename(self):
        assert extract_europe_id("france_europe1.txt") == "europe1"
        assert extract_europe_id("germany_europe2.txt") == "europe2"
        assert extract_europe_id("italy_europe_1.json") == "europe1"
        assert extract_europe_id("poland_europe-2.json") == "europe2"
        assert extract_europe_id("benelux_eur1.txt") == "europe1"
        assert extract_europe_id("scandinavia_eur2.txt") == "europe2"
        assert extract_europe_id("france.txt", default_europe="europe1") == "europe1"

    def test_extract_europe_id_from_file_content(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
            tmp.write("country: france\neurope: europe2\nnuclear: 50000\n")
            tmp_path = tmp.name

        try:
            assert extract_europe_id(tmp_path) == "europe2"
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_extract_country_and_europe(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
            tmp.write("country: germany\ngas: 20000\n")
            tmp_path = tmp.name

        try:
            country, eur = extract_country_and_europe(tmp_path, default_europe="europe1")
            assert country == "germany"
            assert eur == "europe1"
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_group_files_by_europe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "france_europe1.txt")
            f2 = os.path.join(tmpdir, "france_europe2.txt")
            f3 = os.path.join(tmpdir, "germany_europe1.txt")

            with open(f1, "w") as fp:
                fp.write("country: france\nnuclear: 60000\n")
            with open(f2, "w") as fp:
                fp.write("country: france\nnuclear: 50000\n")
            with open(f3, "w") as fp:
                fp.write("country: germany\nwind_onshore: 65000\n")

            grouped = group_files_by_europe([f1, f2, f3])
            assert "europe1" in grouped
            assert "europe2" in grouped
            assert "france" in grouped["europe1"]
            assert "germany" in grouped["europe1"]
            assert "france" in grouped["europe2"]
            assert grouped["europe1"]["france"]["nuclear"] == 60000.0
            assert grouped["europe2"]["france"]["nuclear"] == 50000.0


class TestComparativeAnalysisAndReporting:
    def test_extract_simulation_metrics(self):
        mock_summary = {
            "total_cost": 10000.0,
            "total_operational_cost": 8000.0,
            "total_co2_emissions": 40.0,
            "per_country_ens": {"fra": 0.0, "ger": 0.0},
            "per_country_n_failure_hours": {"fra": 0.0, "ger": 0.0},
            "per_country_total_cost": {"fra": 4000.0, "ger": 6000.0},
            "per_country_total_operational_cost": {"fra": 3200.0, "ger": 4800.0},
            "per_country_co2_emissions": {"fra": 5.0, "ger": 35.0},
            "per_country_production": {"fra": 350000.0, "ger": 450000.0},
        }
        mock_capas = {
            "france": {"nuclear": 60000.0},
            "germany": {"wind_onshore": 60000.0},
        }

        metrics = extract_simulation_metrics(
            summary_data=mock_summary,
            per_country_capacities=mock_capas,
            countries=["france", "germany"],
        )
        assert metrics["total_operational_cost_m_yr"] == 8000.0
        assert metrics["total_co2_emissions_mt"] == 40.0
        assert metrics["total_production_twh"] == 800.0
        assert "france" in metrics["per_country"]
        assert "germany" in metrics["per_country"]

    def test_compute_europe_comparison(self):
        m1 = {
            "total_cost_m_yr": 12000.0,
            "total_operational_cost_m_yr": 8000.0,
            "total_annualized_capex_m_yr": 4000.0,
            "total_overnight_capex_b": 80.0,
            "avg_cost_per_mwh": 75.0,
            "total_co2_emissions_mt": 40.0,
            "total_production_twh": 160.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 5000.0,
                    "opex_m_yr": 3000.0,
                    "capex_m_yr": 2000.0,
                    "co2_mt": 5.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                    "production_gwh": 400000.0,
                    "cost_per_mwh": 12.5,
                }
            },
        }
        m2 = {
            "total_cost_m_yr": 10800.0,
            "total_operational_cost_m_yr": 7000.0,
            "total_annualized_capex_m_yr": 3800.0,
            "total_overnight_capex_b": 75.0,
            "avg_cost_per_mwh": 67.5,
            "total_co2_emissions_mt": 30.0,
            "total_production_twh": 160.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 4500.0,
                    "opex_m_yr": 2600.0,
                    "capex_m_yr": 1900.0,
                    "co2_mt": 4.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                    "production_gwh": 400000.0,
                    "cost_per_mwh": 11.25,
                }
            },
        }

        comparison = compute_europe_comparison(m1, m2)
        assert comparison["global"]["total_cost_m_yr"]["delta"] == -1200.0
        assert comparison["global"]["total_cost_m_yr"]["pct_change"] == -10.0
        assert comparison["global"]["total_co2_emissions_mt"]["delta"] == -10.0
        assert comparison["per_country"]["france"]["total_cost_m_yr"]["delta"] == -500.0

    def test_format_comparison_table_and_display(self, capsys):
        m1 = {
            "total_cost_m_yr": 10000.0,
            "total_operational_cost_m_yr": 6000.0,
            "total_annualized_capex_m_yr": 4000.0,
            "total_overnight_capex_b": 80.0,
            "avg_cost_per_mwh": 70.0,
            "total_co2_emissions_mt": 50.0,
            "total_production_twh": 150.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 4000.0,
                    "opex_m_yr": 2500.0,
                    "capex_m_yr": 1500.0,
                    "co2_mt": 5.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                    "production_gwh": 300000.0,
                    "cost_per_mwh": 13.3,
                }
            },
        }
        m2 = {
            "total_cost_m_yr": 9500.0,
            "total_operational_cost_m_yr": 5800.0,
            "total_annualized_capex_m_yr": 3700.0,
            "total_overnight_capex_b": 75.0,
            "avg_cost_per_mwh": 65.0,
            "total_co2_emissions_mt": 40.0,
            "total_production_twh": 150.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 3800.0,
                    "opex_m_yr": 2400.0,
                    "capex_m_yr": 1400.0,
                    "co2_mt": 4.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                    "production_gwh": 300000.0,
                    "cost_per_mwh": 12.6,
                }
            },
        }
        comparison = compute_europe_comparison(m1, m2)
        report = format_comparison_table(m1, m2, comparison)
        display_europe_comparison(report)

        captured = capsys.readouterr().out
        assert "COMPARATIVE DASHBOARD: EUROPE 1 vs EUROPE 2" in captured
        assert "Total System Cost (OPEX+CAPEX)" in captured
        assert "PER-COUNTRY COMPARISON" in captured
        assert "france" in captured

    def test_save_comparison_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp_mock = {"global": {"total_cost_m_yr": {"delta": -500.0}}}
            txt_report = "Mock Comparative Report Content"
            txt_p, json_p = save_comparison_results(
                comparison_data=comp_mock,
                formatted_report=txt_report,
                output_dir=tmpdir,
                filename_prefix="test_comp",
            )
            assert os.path.isfile(txt_p)
            assert os.path.isfile(json_p)
            with open(txt_p, "r", encoding="utf-8") as f:
                assert f.read() == txt_report
            with open(json_p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                assert loaded["global"]["total_cost_m_yr"]["delta"] == -500.0
                assert "timestamp" in loaded
                assert "timestamp_str" in loaded

            # Check that _latest copies were created as well
            latest_txt = os.path.join(tmpdir, "test_comp_latest.txt")
            latest_json = os.path.join(tmpdir, "test_comp_latest.json")
            assert os.path.isfile(latest_txt)
            assert os.path.isfile(latest_json)


class TestSimulationHistoryAndEvolutions:
    def test_load_simulation_history_chronological_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 mock runs with timestamps
            run1 = {
                "timestamp_str": "2026-11-15 10:00:00",
                "europe1_metrics": {
                    "total_cost_m_yr": 10000.0,
                    "per_country": {
                        "france": {"total_cost_m_yr": 4000.0, "co2_mt": 5.0, "ens_mwh": 0.0, "failure_hours": 0.0}
                    }
                }
            }
            run2 = {
                "timestamp_str": "2026-11-15 12:00:00",
                "europe1_metrics": {
                    "total_cost_m_yr": 9500.0,
                    "per_country": {
                        "france": {"total_cost_m_yr": 3800.0, "co2_mt": 4.0, "ens_mwh": 0.0, "failure_hours": 0.0}
                    }
                }
            }
            # Put them out of order in filenames
            with open(os.path.join(tmpdir, "comp_20261115_120000.json"), "w") as f:
                json.dump(run2, f)
            with open(os.path.join(tmpdir, "comp_20261115_100000.json"), "w") as f:
                json.dump(run1, f)
            # Add latest file (should be ignored)
            with open(os.path.join(tmpdir, "comp_latest.json"), "w") as f:
                json.dump(run2, f)

            history = load_simulation_history(output_dir=tmpdir, target_europe="europe1")
            assert len(history) == 2
            assert history[0]["run_index"] == 1
            assert history[0]["timestamp"] == "2026-11-15 10:00:00"
            assert history[1]["run_index"] == 2
            assert history[1]["timestamp"] == "2026-11-15 12:00:00"

    def test_compute_simulation_evolutions_and_format(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            run1 = {
                "timestamp_str": "2026-11-15 10:00:00",
                "europe1_metrics": {
                    "total_cost_m_yr": 10000.0,
                    "total_operational_cost_m_yr": 6000.0,
                    "total_annualized_capex_m_yr": 4000.0,
                    "total_co2_emissions_mt": 50.0,
                    "total_ens_mwh": 100.0,
                    "max_failure_hours": 2.0,
                    "per_country": {
                        "france": {"total_cost_m_yr": 4000.0, "co2_mt": 5.0, "ens_mwh": 0.0, "failure_hours": 0.0},
                        "germany": {"total_cost_m_yr": 6000.0, "co2_mt": 45.0, "ens_mwh": 100.0, "failure_hours": 2.0},
                    }
                }
            }
            run2 = {
                "timestamp_str": "2026-11-15 14:00:00",
                "europe1_metrics": {
                    "total_cost_m_yr": 9200.0,
                    "total_operational_cost_m_yr": 5500.0,
                    "total_annualized_capex_m_yr": 3700.0,
                    "total_co2_emissions_mt": 42.0,
                    "total_ens_mwh": 0.0,
                    "max_failure_hours": 0.0,
                    "per_country": {
                        "france": {"total_cost_m_yr": 3800.0, "co2_mt": 4.0, "ens_mwh": 0.0, "failure_hours": 0.0},
                        "germany": {"total_cost_m_yr": 5400.0, "co2_mt": 38.0, "ens_mwh": 0.0, "failure_hours": 0.0},
                    }
                }
            }
            with open(os.path.join(tmpdir, "run1.json"), "w") as f:
                json.dump(run1, f)
            with open(os.path.join(tmpdir, "run2.json"), "w") as f:
                json.dump(run2, f)

            history = load_simulation_history(output_dir=tmpdir, target_europe="europe1")
            assert len(history) == 2

            evol = compute_simulation_evolutions(history)
            assert evol["has_history"] is True
            assert evol["global_net_evolution"]["total_cost_m_yr"]["delta"] == -800.0
            assert evol["global_net_evolution"]["total_cost_m_yr"]["pct_change"] == -8.0
            assert evol["country_evolutions"]["germany"]["cost_delta"] == -600.0
            assert evol["country_evolutions"]["germany"]["ens_delta"] == -100.0

            table_text = format_simulation_evolutions_table(evol, target_europe="Europe 1")
            display_simulation_evolutions(table_text)
            captured = capsys.readouterr().out

            assert "SIMULATION EVOLUTION HISTORY: EUROPE 1" in captured
            assert "GLOBAL SYSTEM EVOLUTION ACROSS SIMULATIONS" in captured
            assert "PER-COUNTRY TOTAL SYSTEM COST (M€/yr) EVOLUTION" in captured
            assert "ENERGY NOT SERVED - ENS (MWh)" in captured
            assert "LOSS OF LOAD - FAILURE (h)" in captured
            assert "germany" in captured

            # Test saving evolution report
            txt_p, json_p = save_simulation_evolutions(
                evolution_data=evol,
                report_text=table_text,
                output_dir=tmpdir,
                target_europe="europe1",
            )
            assert os.path.isfile(txt_p)
            assert os.path.isfile(json_p)


class TestInvestmentsRunnerSimulationPeriod:
    def test_resolve_simulation_period_default_full_year(self):
        start, end = resolve_simulation_period(is_test=False)
        assert start == "1900/1/1"
        assert end == "1901/1/1"

    def test_resolve_simulation_period_test_mode(self):
        start, end = resolve_simulation_period(is_test=True)
        assert start == "1900/1/1"
        assert end == "1900/1/15"

    def test_resolve_simulation_period_test_mode_preserves_custom_dates(self):
        # Custom start date with test mode
        start, end = resolve_simulation_period(is_test=True, period_start="1900/2/1")
        assert start == "1900/2/1"
        assert end == "1900/1/15"

        # Custom end date with test mode
        start, end = resolve_simulation_period(is_test=True, period_end="1900/1/20")
        assert start == "1900/1/1"
        assert end == "1900/1/20"

    def test_update_europe_params_with_test_period(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json.dump({"mode": "europe", "team": "france"}, tmp)
            tmp_path = tmp.name

        try:
            test_start, test_end = resolve_simulation_period(is_test=True)
            update_europe_params(
                mode="europe",
                team="france",
                period_start=test_start,
                period_end=test_end,
                europe_params_path=tmp_path,
            )
            with open(tmp_path, "r") as fp:
                data = json.load(fp)
            assert data["mode"] == "europe"
            assert data["uc_period_start"] == "1900/1/1"
            assert data["uc_period_end"] == "1900/1/15"
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_main_runner_cli_test_flag(self):
        from main_runner import build_parser

        parser = build_parser()

        # Without --test
        args_default = parser.parse_args([])
        assert args_default.test is False
        start, end = resolve_simulation_period(
            is_test=args_default.test,
            period_start=args_default.period_start,
            period_end=args_default.period_end,
        )
        assert start == "1900/1/1"
        assert end == "1901/1/1"

        # With --test
        args_test = parser.parse_args(["--test"])
        assert args_test.test is True
        start, end = resolve_simulation_period(
            is_test=args_test.test,
            period_start=args_test.period_start,
            period_end=args_test.period_end,
        )
        assert start == "1900/1/1"
        assert end == "1900/1/15"


class TestInvestmentsRunnerCleanOutputs:
    def test_clear_runner_outputs_removes_files_and_preserves_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_names = [
                "comparison_europe1_vs_europe2_20261116_112515.json",
                "comparison_europe1_vs_europe2_20261116_112515.txt",
                "comparison_europe1_vs_europe2_latest.json",
                "comparison_europe1_vs_europe2_latest.txt",
                "history_evolutions_europe1.json",
                "history_evolutions_europe1.txt",
            ]
            for fname in file_names:
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "w") as fp:
                    fp.write("{}")

            # Hidden file that must be preserved
            keep_file = os.path.join(tmpdir, ".gitkeep")
            with open(keep_file, "w") as fp:
                fp.write("")

            removed = clear_runner_outputs(output_dir=tmpdir)
            assert len(removed) == 6
            assert os.path.exists(keep_file)
            remaining = os.listdir(tmpdir)
            assert remaining == [".gitkeep"]

            # Verify history load is now completely clean
            history = load_simulation_history(output_dir=tmpdir, target_europe="europe1")
            assert history == []

    def test_clear_runner_outputs_nonexistent_directory(self):
        removed = clear_runner_outputs(output_dir="/non/existent/path/for/clear/runner/test")
        assert removed == []

    def test_clear_runner_outputs_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            removed = clear_runner_outputs(output_dir=tmpdir)
            assert removed == []

    def test_main_runner_cli_clean_flags(self):
        from main_runner import build_parser

        parser = build_parser()
        args_clean = parser.parse_args(["--clean"])
        assert args_clean.clean is True

        args_clear = parser.parse_args(["--clear"])
        assert args_clear.clean is True

        args_default = parser.parse_args([])
        assert args_default.clean is False

    def test_main_runner_clean_standalone_execution(self, monkeypatch):
        from main_runner import main

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "comparison_europe1_vs_europe2_20261116_112515.json")
            with open(test_file, "w") as fp:
                fp.write("{}")

            monkeypatch.setattr("sys.argv", ["main_runner.py", "--output-dir", tmpdir, "--clean"])

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0
            assert not os.path.exists(test_file)

    def test_main_runner_clean_keyword_execution(self, monkeypatch):
        from main_runner import main

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "history_evolutions_europe1.json")
            with open(test_file, "w") as fp:
                fp.write("{}")

            monkeypatch.setattr("sys.argv", ["main_runner.py", "--output-dir", tmpdir, "clean"])

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0
            assert not os.path.exists(test_file)


class TestInvestmentsRunnerCapacityAndEraaComparison:
    """Tests for installed capacity display, ERAA references loading and capacity comparison."""

    def test_load_eraa_country_capacities_numeric_values(self):
        # 2025 France
        capas_2025 = load_eraa_country_capacities(year=2025)
        assert "france" in capas_2025
        assert "germany" in capas_2025
        fr_25 = capas_2025["france"]
        assert fr_25["nuclear"] == 61761.0
        assert fr_25["solar_pv"] == 18184.99
        assert fr_25["wind_onshore"] == 24058.99
        assert fr_25["wind_offshore"] == 2015.0
        assert fr_25["batteries"] == 470.0
        assert fr_25["hydro_reservoir"] == 9539.0
        assert fr_25["hydro_run_of_river"] == 13600.0
        assert fr_25["gas"] == 7189.0

        # 2033 France
        capas_2033 = load_eraa_country_capacities(year=2033)
        fr_33 = capas_2033["france"]
        assert fr_33["nuclear"] == 61761.0
        assert fr_33["solar_pv"] == 54282.16
        assert fr_33["wind_offshore"] == 11575.0
        assert fr_33["wind_onshore"] == 36100.03

        # 2025 Germany coal aggregation (hard_coal + lignite)
        de_25 = capas_2025["germany"]
        expected_coal = 12786.24 + 14687.0
        assert pytest.approx(de_25["coal"], rel=1e-4) == expected_coal

    def test_load_eraa_country_capacities_missing_directory(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            caps = load_eraa_country_capacities(year=2025, data_dir=empty_dir)
            assert isinstance(caps, dict)
            for c in AVAILABLE_COUNTRIES:
                assert c in caps
                assert caps[c] == {}

    def test_format_installed_capacities_table_and_display(self, capsys):
        sample_caps = {
            "france": {
                "nuclear": 50000.0,
                "solar_pv": 40000.0,
                "wind_onshore": 30000.0,
                "wind_offshore": 10000.0,
                "gas": 5000.0,
            },
            "germany": {
                "solar_pv": 60000.0,
                "wind_onshore": 50000.0,
                "gas": 20000.0,
            },
        }

        table_str = format_installed_capacities_table(sample_caps, title="TEST CAPACITIES")
        assert "TEST CAPACITIES" in table_str
        assert "Country" in table_str
        assert "Nuc" in table_str
        assert "Solar" in table_str
        assert "TOTAL EUROPE" in table_str
        assert "france" in table_str
        assert "germany" in table_str
        assert "LowCarb%" in table_str
        assert "Overnight" in table_str
        assert "Ann.CAPEX" in table_str

        # Verify console display
        display_installed_capacities(table_str)
        captured = capsys.readouterr()
        assert "TEST CAPACITIES" in captured.out
        assert "TOTAL EUROPE" in captured.out

    def test_compute_capacity_comparison_with_eraa(self):
        current = {
            "france": {
                "nuclear": 50000.0,
                "solar_pv": 25000.0,
            }
        }
        eraa_ref = {
            "france": {
                "nuclear": 60000.0,
                "solar_pv": 20000.0,
            }
        }

        comp = compute_capacity_comparison_with_eraa(current, eraa_ref)
        fr_comp = comp["per_country"]["france"]

        assert fr_comp["nuclear"]["current_mw"] == 50000.0
        assert fr_comp["nuclear"]["eraa_mw"] == 60000.0
        assert fr_comp["nuclear"]["diff_mw"] == -10000.0
        assert fr_comp["nuclear"]["diff_gw"] == -10.0
        assert pytest.approx(fr_comp["nuclear"]["diff_pct"], rel=1e-3) == -16.6666

        assert fr_comp["solar_pv"]["diff_mw"] == 5000.0
        assert fr_comp["solar_pv"]["diff_gw"] == 5.0
        assert pytest.approx(fr_comp["solar_pv"]["diff_pct"], rel=1e-3) == 25.0

        # Total for country
        assert fr_comp["total"]["current_mw"] == 75000.0
        assert fr_comp["total"]["eraa_mw"] == 80000.0
        assert fr_comp["total"]["diff_mw"] == -5000.0

        # Total Europe
        eur_comp = comp["total_europe"]
        assert eur_comp["total"]["current_mw"] == 75000.0
        assert eur_comp["total"]["eraa_mw"] == 80000.0

    def test_format_capacity_comparison_table(self):
        current = {
            "france": {
                "nuclear": 50000.0,
                "solar_pv": 30000.0,
            }
        }
        eraa_2025 = {
            "france": {
                "nuclear": 61761.0,
                "solar_pv": 18185.0,
            }
        }
        eraa_2033 = {
            "france": {
                "nuclear": 61761.0,
                "solar_pv": 54282.0,
            }
        }

        report = format_capacity_comparison_table(
            current_capacities=current,
            eraa_2025_capas=eraa_2025,
            eraa_2033_capas=eraa_2033,
            europe_name="Europe 1",
        )

        assert "CAPACITY COMPARISON vs ERAA REFERENCES - EUROPE 1 (GW)" in report
        assert "[FRANCE]" in report
        assert "nuclear" in report
        assert "solar_pv" in report
        assert "ERAA 2025" in report
        assert "ERAA 2033" in report
        assert "TOTAL EUROPE" in report

    def test_main_runner_dry_run_displays_capacities_and_eraa_comparison(self, monkeypatch, capsys):
        from main_runner import main

        monkeypatch.setattr(
            "sys.argv",
            ["main_runner.py", "input/runner/france_europe1.txt", "input/runner/france_europe2.txt", "--dry-run"],
        )
        main()

        captured = capsys.readouterr()
        assert "INSTALLED CAPACITIES - EUROPE1 (GW)" in captured.out
        assert "INSTALLED CAPACITIES - EUROPE2 (GW)" in captured.out
        assert "INCREMENTAL CAPACITY TRAJECTORY: ERAA REFERENCES vs HISTORICAL RUNS (GW)" in captured.out

    def test_extract_simulation_metrics_includes_capacities(self):
        summary_mock = {
            "total_operational_cost": 15000.0,
            "total_co2_emissions": 85.0,
            "per_country_total_cost": {"FRA": 8000.0, "DEU": 7000.0},
        }
        test_capas = {
            "france": {"nuclear": 50000.0, "solar_pv": 40000.0},
            "germany": {"wind_onshore": 60000.0, "gas": 20000.0},
        }

        metrics = extract_simulation_metrics(
            summary_data=summary_mock,
            per_country_capacities=test_capas,
            countries=["france", "germany"],
        )

        assert "per_country_capacities" in metrics
        assert metrics["per_country_capacities"]["france"] == test_capas["france"]
        assert metrics["per_country"]["france"]["capacities"] == test_capas["france"]
        assert metrics["per_country"]["germany"]["capacities"] == test_capas["germany"]

    def test_save_and_load_simulation_history_with_capacities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_e1 = {
                "total_cost_m_yr": 25000.0,
                "total_operational_cost_m_yr": 15000.0,
                "total_annualized_capex_m_yr": 10000.0,
                "total_co2_emissions_mt": 50.0,
                "total_ens_mwh": 0.0,
                "max_failure_hours": 0.0,
                "per_country": {
                    "france": {
                        "total_cost_m_yr": 12000.0,
                        "co2_mt": 10.0,
                        "ens_mwh": 0.0,
                        "failure_hours": 0.0,
                        "capacities": {"nuclear": 50000.0, "solar_pv": 40000.0},
                    }
                },
                "per_country_capacities": {
                    "france": {"nuclear": 50000.0, "solar_pv": 40000.0}
                },
            }
            comparison_dummy = {"global": {}, "per_country": {}}

            save_comparison_results(
                comparison_data=comparison_dummy,
                formatted_report="Report",
                output_dir=tmpdir,
                metrics_eur1=metrics_e1,
                timestamp=datetime(2026, 11, 17, 10, 0, 0),
            )

            hist = load_simulation_history(output_dir=tmpdir, target_europe="europe1")
            assert len(hist) == 1
            loaded_m = hist[0]["metrics"]
            assert "per_country" in loaded_m
            assert loaded_m["per_country"]["france"]["capacities"] == {"nuclear": 50000.0, "solar_pv": 40000.0}

    def test_compute_and_format_simulation_evolutions_with_capacities(self):
        history = [
            {
                "run_index": 1,
                "filename": "run1.json",
                "timestamp": "2026-11-17 10:00:00",
                "metrics": {
                    "total_cost_m_yr": 30000.0,
                    "total_operational_cost_m_yr": 20000.0,
                    "total_annualized_capex_m_yr": 10000.0,
                    "total_co2_emissions_mt": 80.0,
                    "total_ens_mwh": 500.0,
                    "max_failure_hours": 12.0,
                    "per_country": {
                        "france": {
                            "total_cost_m_yr": 15000.0,
                            "co2_mt": 20.0,
                            "ens_mwh": 500.0,
                            "failure_hours": 12.0,
                            "capacities": {"nuclear": 50000.0, "solar_pv": 40000.0, "gas": 5000.0},
                        }
                    },
                },
            },
            {
                "run_index": 2,
                "filename": "run2.json",
                "timestamp": "2026-11-17 11:00:00",
                "metrics": {
                    "total_cost_m_yr": 28000.0,
                    "total_operational_cost_m_yr": 17000.0,
                    "total_annualized_capex_m_yr": 11000.0,
                    "total_co2_emissions_mt": 65.0,
                    "total_ens_mwh": 0.0,
                    "max_failure_hours": 0.0,
                    "per_country": {
                        "france": {
                            "total_cost_m_yr": 14000.0,
                            "co2_mt": 15.0,
                            "ens_mwh": 0.0,
                            "failure_hours": 0.0,
                            "capacities": {"nuclear": 50000.0, "solar_pv": 45000.0, "gas": 3000.0},
                        }
                    },
                },
            },
        ]

        evol_data = compute_simulation_evolutions(history)
        assert evol_data["has_history"] is True
        cap_evol = evol_data["capacity_evolutions"]
        assert cap_evol["has_capacity_data"] is True

        fr_cap = cap_evol["per_country"]["france"]
        assert fr_cap["solar_pv"]["delta_prev_gw"] == 5.0
        assert fr_cap["solar_pv"]["delta_net_gw"] == 5.0
        assert pytest.approx(fr_cap["solar_pv"]["pct_net"], rel=1e-3) == 12.5

        assert fr_cap["gas"]["delta_prev_gw"] == -2.0
        assert fr_cap["gas"]["delta_net_gw"] == -2.0
        assert pytest.approx(fr_cap["gas"]["pct_net"], rel=1e-3) == -40.0

        # Check formatting
        report = format_simulation_evolutions_table(evol_data, target_europe="Europe 1")
        assert "[5] CAPACITY EVOLUTION ACROSS SUCCESSIVE RUNS vs ERAA REFERENCES (GW)" in report
        assert "ERAA 2025" in report
        assert "ERAA 2033" in report
        assert "Δ vs 2025" in report
        assert "Δ vs 2033" in report
        assert "[FRANCE]" in report
        assert "solar_pv" in report
        assert "gas" in report
        assert "+5.0" in report
        assert "-2.0" in report
        assert "[TOTAL EUROPE]" in report


class TestInvestmentsRunnerEraaBaselinesAndFourWayComparison:
    """Tests for pure ERAA baselines execution/caching and 4-way comparative dashboard."""

    def test_get_period_tag_and_cache_path(self):
        assert get_period_tag(TEST_PERIOD_START, TEST_PERIOD_END) == "test_2w"
        assert get_period_tag(DEFAULT_PERIOD_START, DEFAULT_PERIOD_END) == "full_year"
        assert get_period_tag("1900/01/01", "1900/01/15") == "19000101_19000115"

        path_25 = get_eraa_baseline_cache_path(
            eraa_year=2025,
            period_start=TEST_PERIOD_START,
            period_end=TEST_PERIOD_END,
            target_year=2033,
            climatic_year=1989,
            output_dir="output/runner",
        )
        assert "output/runner/baselines" in path_25
        assert "eraa_baseline_2025_target2033_cy1989_test_2w.json" in path_25

    def test_get_or_run_eraa_baseline_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = get_eraa_baseline_cache_path(
                eraa_year=2025,
                period_start=TEST_PERIOD_START,
                period_end=TEST_PERIOD_END,
                target_year=2033,
                climatic_year=1989,
                output_dir=tmpdir,
            )
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            mock_metrics = {
                "total_cost_m_yr": 60000.0,
                "total_operational_cost_m_yr": 40000.0,
                "total_annualized_capex_m_yr": 20000.0,
                "total_co2_emissions_mt": 75.0,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(mock_metrics, f)

            loaded = get_or_run_eraa_baseline(
                eraa_year=2025,
                period_start=TEST_PERIOD_START,
                period_end=TEST_PERIOD_END,
                target_year=2033,
                climatic_year=1989,
                output_dir=tmpdir,
                dry_run=False,
            )
            assert loaded is not None
            assert loaded["total_cost_m_yr"] == 60000.0

    def test_get_or_run_eraa_baseline_dry_run_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = get_or_run_eraa_baseline(
                eraa_year=2025,
                period_start=TEST_PERIOD_START,
                period_end=TEST_PERIOD_END,
                target_year=2033,
                climatic_year=1989,
                output_dir=tmpdir,
                dry_run=True,
            )
            assert res is None

    def test_compute_four_way_comparison(self):
        m_e1 = {
            "total_cost_m_yr": 55000.0,
            "total_operational_cost_m_yr": 35000.0,
            "total_annualized_capex_m_yr": 20000.0,
            "total_overnight_capex_b": 300.0,
            "avg_cost_per_mwh": 60.0,
            "total_co2_emissions_mt": 40.0,
            "total_production_twh": 900.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 20000.0,
                    "opex_m_yr": 10000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 10.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                }
            },
        }
        m_e2 = {
            "total_cost_m_yr": 53000.0,
            "total_operational_cost_m_yr": 33000.0,
            "total_annualized_capex_m_yr": 20000.0,
            "total_overnight_capex_b": 300.0,
            "avg_cost_per_mwh": 58.0,
            "total_co2_emissions_mt": 38.0,
            "total_production_twh": 900.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 19000.0,
                    "opex_m_yr": 9000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 9.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                }
            },
        }
        m_25 = {
            "total_cost_m_yr": 65000.0,
            "total_operational_cost_m_yr": 50000.0,
            "total_annualized_capex_m_yr": 15000.0,
            "total_overnight_capex_b": 220.0,
            "avg_cost_per_mwh": 75.0,
            "total_co2_emissions_mt": 85.0,
            "total_production_twh": 850.0,
            "total_ens_mwh": 500.0,
            "max_failure_hours": 6.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 25000.0,
                    "opex_m_yr": 15000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 20.0,
                    "ens_mwh": 100.0,
                    "failure_hours": 3.0,
                }
            },
        }
        m_33 = {
            "total_cost_m_yr": 50000.0,
            "total_operational_cost_m_yr": 28000.0,
            "total_annualized_capex_m_yr": 22000.0,
            "total_overnight_capex_b": 350.0,
            "avg_cost_per_mwh": 55.0,
            "total_co2_emissions_mt": 30.0,
            "total_production_twh": 910.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 18000.0,
                    "opex_m_yr": 8000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 7.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                }
            },
        }

        four_way = compute_four_way_comparison(
            metrics_eur1=m_e1,
            metrics_eur2=m_e2,
            metrics_eraa2025=m_25,
            metrics_eraa2033=m_33,
        )

        assert four_way["has_eraa_baselines"] is True
        glob = four_way["global"]

        # Total cost: E1 vs ERAA 2033 (55000 - 50000 = +5000, +10.0%)
        assert glob["total_cost_m_yr"]["delta_eur1_vs_33"] == 5000.0
        assert pytest.approx(glob["total_cost_m_yr"]["pct_eur1_vs_33"], rel=1e-3) == 10.0

        # E2 vs ERAA 2033 (53000 - 50000 = +3000, +6.0%)
        assert glob["total_cost_m_yr"]["delta_eur2_vs_33"] == 3000.0
        assert pytest.approx(glob["total_cost_m_yr"]["pct_eur2_vs_33"], rel=1e-3) == 6.0

        # CO2 emissions: E1 vs 33 (40 - 30 = +10 Mt, +33.3%)
        assert glob["total_co2_emissions_mt"]["delta_eur1_vs_33"] == 10.0
        assert pytest.approx(glob["total_co2_emissions_mt"]["pct_eur1_vs_33"], rel=1e-3) == 33.333

        # Per-country France
        fr_comp = four_way["per_country"]["france"]
        assert fr_comp["total_cost_m_yr"]["delta_eur1_vs_33"] == 2000.0

    def test_format_four_way_comparison_table(self):
        m_e1 = {
            "total_cost_m_yr": 55000.0,
            "total_operational_cost_m_yr": 35000.0,
            "total_annualized_capex_m_yr": 20000.0,
            "total_overnight_capex_b": 300.0,
            "avg_cost_per_mwh": 60.0,
            "total_co2_emissions_mt": 40.0,
            "total_production_twh": 900.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 20000.0,
                    "opex_m_yr": 10000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 10.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                }
            },
        }
        m_33 = {
            "total_cost_m_yr": 50000.0,
            "total_operational_cost_m_yr": 28000.0,
            "total_annualized_capex_m_yr": 22000.0,
            "total_overnight_capex_b": 350.0,
            "avg_cost_per_mwh": 55.0,
            "total_co2_emissions_mt": 30.0,
            "total_production_twh": 910.0,
            "total_ens_mwh": 0.0,
            "max_failure_hours": 0.0,
            "per_country": {
                "france": {
                    "total_cost_m_yr": 18000.0,
                    "opex_m_yr": 8000.0,
                    "capex_m_yr": 10000.0,
                    "co2_mt": 7.0,
                    "ens_mwh": 0.0,
                    "failure_hours": 0.0,
                }
            },
        }

        four_way = compute_four_way_comparison(metrics_eur1=m_e1, metrics_eraa2033=m_33)
        table_str = format_four_way_comparison_table(four_way)

        assert "4-WAY COMPARISON: STUDENT MIXES vs OFFICIAL ERAA BASELINES" in table_str
        assert "GLOBAL METRIC" in table_str
        assert "EUROPE 1" in table_str
        assert "ERAA 2033" in table_str
        assert "Δ(E1 - ERAA33)" in table_str
        assert "Total system cost" in table_str
        assert "[FRANCE]" in table_str

    def test_save_comparison_results_with_four_way(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            comp_dummy = {"global": {}, "per_country": {}}
            m_e1 = {"total_cost_m_yr": 50000.0}
            m_33 = {"total_cost_m_yr": 45000.0}
            four_way_data = {"global": {}, "per_country": {}}
            four_way_rep = "=== FOUR WAY REPORT ==="

            txt_file, json_file = save_comparison_results(
                comparison_data=comp_dummy,
                formatted_report="Standard Report",
                output_dir=tmpdir,
                metrics_eur1=m_e1,
                metrics_eraa2033=m_33,
                four_way_comparison=four_way_data,
                four_way_report=four_way_rep,
                timestamp=datetime(2026, 11, 17, 12, 0, 0),
            )

            assert os.path.isfile(txt_file)
            assert os.path.isfile(json_file)

            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Standard Report" in content
            assert "=== FOUR WAY REPORT ===" in content

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "eraa2033_metrics" in data
            assert "four_way_comparison" in data

    def test_main_runner_cli_eraa_flags(self):
        from main_runner import build_parser

        parser = build_parser()
        args_no_eraa = parser.parse_args(["--no-compare-eraa"])
        assert args_no_eraa.no_compare_eraa is True

        args_force_eraa = parser.parse_args(["--force-eraa-run"])
        assert args_force_eraa.force_eraa_run is True

        args_default = parser.parse_args([])
        assert args_default.no_compare_eraa is False
        assert args_default.force_eraa_run is False
        assert args_default.history_europe == "all"

        args_e2 = parser.parse_args(["--history-europe", "europe2"])
        assert args_e2.history_europe == "europe2"

    def test_load_and_compute_simulation_evolutions_europe2(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            run1 = {
                "timestamp_str": "2026-11-17 10:00:00",
                "europe2_metrics": {
                    "total_cost_m_yr": 10000.0,
                    "total_operational_cost_m_yr": 6000.0,
                    "total_annualized_capex_m_yr": 4000.0,
                    "total_co2_emissions_mt": 30.0,
                    "total_ens_mwh": 50.0,
                    "max_failure_hours": 2.0,
                    "per_country": {
                        "france": {
                            "total_cost_m_yr": 5000.0,
                            "co2_mt": 15.0,
                            "ens_mwh": 50.0,
                            "failure_hours": 2.0,
                            "capacities": {"nuclear": 50000.0},
                        }
                    },
                },
            }
            run2 = {
                "timestamp_str": "2026-11-17 11:00:00",
                "europe2_metrics": {
                    "total_cost_m_yr": 9200.0,
                    "total_operational_cost_m_yr": 5200.0,
                    "total_annualized_capex_m_yr": 4000.0,
                    "total_co2_emissions_mt": 25.0,
                    "total_ens_mwh": 0.0,
                    "max_failure_hours": 0.0,
                    "per_country": {
                        "france": {
                            "total_cost_m_yr": 4600.0,
                            "co2_mt": 12.0,
                            "ens_mwh": 0.0,
                            "failure_hours": 0.0,
                            "capacities": {"nuclear": 50000.0},
                        }
                    },
                },
            }
            with open(os.path.join(tmpdir, "comparison_europe1_vs_europe2_1.json"), "w") as fp:
                json.dump(run1, fp)
            with open(os.path.join(tmpdir, "comparison_europe1_vs_europe2_2.json"), "w") as fp:
                json.dump(run2, fp)

            history_e2 = load_simulation_history(output_dir=tmpdir, target_europe="europe2")
            assert len(history_e2) == 2
            evol_e2 = compute_simulation_evolutions(history_e2)
            assert evol_e2["has_history"] is True
            assert evol_e2["global_net_evolution"]["total_cost_m_yr"]["delta"] == -800.0

            table_e2 = format_simulation_evolutions_table(evol_e2, target_europe="europe2")
            assert "SIMULATION EVOLUTION HISTORY: EUROPE2" in table_e2
            assert "france" in table_e2


class TestInvestmentsRunnerIncrementalCapacityTrajectory:
    """Tests for incremental capacity trajectory across ERAA references and successive runs."""

    def test_load_all_simulation_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run1 = {
                "timestamp_str": "2026-11-17 10:00:00",
                "europe1_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 40000.0}},
                "europe2_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 35000.0}},
            }
            run2 = {
                "timestamp_str": "2026-11-17 11:00:00",
                "europe1_capacities": {"france": {"nuclear": 55000.0, "solar_pv": 45000.0}},
                "europe2_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 38000.0}},
            }
            with open(os.path.join(tmpdir, "comparison_europe1_vs_europe2_1.json"), "w") as fp:
                json.dump(run1, fp)
            with open(os.path.join(tmpdir, "comparison_europe1_vs_europe2_2.json"), "w") as fp:
                json.dump(run2, fp)

            loaded = load_all_simulation_runs(output_dir=tmpdir)
            assert len(loaded) == 2
            assert loaded[0]["run_index"] == 1
            assert loaded[0]["europe1_capacities"]["france"]["nuclear"] == 50000.0
            assert loaded[1]["run_index"] == 2
            assert loaded[1]["europe1_capacities"]["france"]["nuclear"] == 55000.0

    def test_format_incremental_capacities_trajectory_table(self):
        eraa_25 = {"france": {"nuclear": 61761.0, "solar_pv": 18185.0}}
        eraa_33 = {"france": {"nuclear": 61761.0, "solar_pv": 54282.0}}

        history = [
            {
                "run_index": 1,
                "europe1_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 40000.0}},
                "europe2_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 35000.0}},
            },
            {
                "run_index": 2,
                "europe1_capacities": {"france": {"nuclear": 55000.0, "solar_pv": 45000.0}},
                "europe2_capacities": {"france": {"nuclear": 50000.0, "solar_pv": 40000.0}},
            },
        ]

        table = format_incremental_capacities_trajectory_table(
            eraa_2025_capas=eraa_25,
            eraa_2033_capas=eraa_33,
            history_runs=history,
            displayed_countries=["france"],
        )

        assert "INCREMENTAL CAPACITY TRAJECTORY: ERAA REFERENCES vs HISTORICAL RUNS (GW)" in table
        assert "ERAA 2025" in table
        assert "ERAA 2033" in table
        assert "R1 (E1)" in table
        assert "R1 (E2)" in table
        assert "R2 (E1)" in table
        assert "R2 (E2)" in table
        assert "[FRANCE]" in table
        assert "nuclear" in table
        assert "solar_pv" in table
        assert "Δ Net (E1)" in table
        assert "Δ Net (E2)" in table
        assert "[TOTAL EUROPE]" in table

        # Verify that Europe 1 runs are grouped together before Europe 2 runs
        pos_e1_1 = table.find("R1 (E1)")
        pos_e1_2 = table.find("R2 (E1)")
        pos_e2_1 = table.find("R1 (E2)")
        pos_e2_2 = table.find("R2 (E2)")
        assert pos_e1_1 < pos_e1_2 < pos_e2_1 < pos_e2_2

        # Verify that column headers are repeated at each country section
        assert table.count("Country / Technology") >= 2

    def test_format_incremental_capacities_summary_table(self):
        eraa_25 = {"france": {"nuclear": 60000.0, "gas": 10000.0}}
        eraa_33 = {"france": {"nuclear": 60000.0, "gas": 5000.0}}

        history = [
            {
                "run_index": 1,
                "europe1_capacities": {"france": {"nuclear": 50000.0, "gas": 5000.0}},
                "europe2_capacities": {"france": {"nuclear": 50000.0, "gas": 8000.0}},
            }
        ]

        sum_table = format_incremental_capacities_summary_table(
            eraa_2025_capas=eraa_25,
            eraa_2033_capas=eraa_33,
            history_runs=history,
            displayed_countries=["france"],
        )

        assert "TOTAL CAPACITY & LOW-CARBON SHARE TRAJECTORY (GW / %)" in sum_table
        assert "france" in sum_table
        assert "TOTAL EUROPE" in sum_table

    def test_save_incremental_capacities_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = save_incremental_capacities_trajectory(
                report_text="Sample Trajectory",
                output_dir=tmpdir,
            )
            assert os.path.isfile(txt_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == "Sample Trajectory"

    def test_tech_display_name_and_column_alignment(self):
        # Long hydro names are shortened
        assert get_tech_display_name("hydro_pump_storage_closed_loop") == "hydro_phs_closed"
        assert get_tech_display_name("hydro_pump_storage_open_loop") == "hydro_phs_open"
        assert get_tech_display_name("nuclear") == "nuclear"

        # Check alignment in format_simulation_evolutions_table
        history = [
            {
                "run_index": 1,
                "timestamp": "2026-11-17 10:00:00",
                "metrics": {
                    "total_cost_m_yr": 1000.0,
                    "per_country": {
                        "france": {
                            "capacities": {
                                "hydro_pump_storage_closed_loop": 5000.0,
                                "hydro_pump_storage_open_loop": 2000.0,
                                "nuclear": 50000.0,
                            }
                        }
                    },
                },
            },
            {
                "run_index": 2,
                "timestamp": "2026-11-17 11:00:00",
                "metrics": {
                    "total_cost_m_yr": 950.0,
                    "per_country": {
                        "france": {
                            "capacities": {
                                "hydro_pump_storage_closed_loop": 6000.0,
                                "hydro_pump_storage_open_loop": 2500.0,
                                "nuclear": 55000.0,
                            }
                        }
                    },
                },
            },
        ]
        evol = compute_simulation_evolutions(history)
        table = format_simulation_evolutions_table(evol, target_europe="Europe 1")

        # In section [5], verify the pipe separator is perfectly aligned across all lines
        sec5_lines = [l for l in table.splitlines() if "|" in l and ("hydro" in l or "nuclear" in l or "COUNTRY / TECH" in l)]
        first_pipe_indices = [l.index("|") for l in sec5_lines]
        assert len(set(first_pipe_indices)) == 1, f"Pipe misaligned in section [5]: {first_pipe_indices}"

    def test_sync_country_json_files_cleans_inactive_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create inactive country json (e.g. benelux)
            benelux_file = os.path.join(tmpdir, "benelux.json")
            with open(benelux_file, "w") as fp:
                fp.write("{}")

            active_countries = ["france", "germany"]
            caps = {
                "france": {"nuclear": 50000.0},
                "germany": {"wind_onshore": 60000.0},
            }

            sync_country_json_files(
                active_countries=active_countries,
                country_capacities=caps,
                countries_dir=tmpdir,
            )

            # Active files must exist
            assert os.path.isfile(os.path.join(tmpdir, "france.json"))
            assert os.path.isfile(os.path.join(tmpdir, "germany.json"))

            # Inactive file must be cleaned up to prevent PyPSA mismatch errors
            assert not os.path.isfile(benelux_file)












