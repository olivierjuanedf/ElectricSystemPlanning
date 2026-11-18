# Plan d'implémentation : Capacités installées, Évolution incrémentale et Comparaison ERAA (Runner)

Ce document décrit le plan technique détaillé et validé pour implémenter les 3 fonctionnalités demandées dans le fichier `TODO` pour le Runner de simulation collective (`main_runner.py` / `src/include_runner/investments_runner.py`).

---

## 1. Contexte & Motivation

Le runner actuel permet de collecter les fichiers de choix d'investissements étudiants (Europe 1 et Europe 2), de lancer les simulations séquentielles PyPSA sur le réseau européen interconnecté, de comparer les coûts opérationnels (OPEX), coûts d'investissement (CAPEX) et émissions CO2, et de suivre l'évolution des métriques globales au fil des runs.

Cependant, plusieurs informations clés manquent pour guider les étudiants et enseignants :
1. **Visibilité des capacités installées par pays** : Les capacités réelles par technologie ne sont pas affichées (seul le nombre d'actifs apparaît).
2. **Évolution incrémentale des capacités** : Les runs successifs ne tracent pas l'évolution des capacités investies, et il n'existe pas de comparaison directe avec les points de référence ENTSO-E ERAA 2025 (mix de départ) et ERAA 2033 (trajectoire de référence).
3. **Comparaison globale avec les mix ERAA purs** : Les étudiants ne peuvent pas situer leurs résultats (Europe 1 et Europe 2) par rapport à ce que donneraient les capacités pures ERAA 2025 et ERAA 2033 simulées sur la même année de test, avec la même météo et le même horizon temporel.

---

## 2. Périmètre & Décisions de Conception

Conformément à la consultation et aux choix validés :
- **Démarche incrémentale** : L'implémentation est structurée en 3 étapes opérationnelles et testables :
  - **Étape 1** : Capacités installées par pays & Comparaison aux références ERAA 2025 / 2033 (Phases 1 & 2).
  - **Étape 2** : Suivi incrémental des capacités dans l'historique des runs (Phase 3).
  - **Étape 3** : Baselines ERAA pures & Comparaison globale à 4 voies (Phases 4 & 5).
- **Niveau d'agrégation des capacités** : Respect strict du niveau d'agrégation des paramètres de `input/long_term_uc/countries/{country}.json` (`capacities_tb_overwritten`) :
  - `nuclear`
  - `solar_pv`
  - `wind_onshore`
  - `wind_offshore`
  - `hydro_reservoir`
  - `hydro_run_of_river`
  - `hydro_pump_storage_closed_loop`
  - `hydro_pump_storage_open_loop`
  - `batteries`
  - `dsr`
  - `gas`
  - (et filières thermiques secondaires si présentes : `coal`, `oil`, `others_fatal`, `biofuel`).
- **Exécution des baselines ERAA pures** : Mode **automatique avec mise en cache**.
  - Une première exécution génère et met en cache les résultats des simulations pures ERAA 2025 et ERAA 2033 dans `output/runner/baselines/` pour les conditions données (`target_year`, `climatic_year`, `period`).
  - Les runs suivants rechargent instantanément le cache (zéro temps de calcul additionnel pour PyPSA).
  - Un drapeau CLI `--no-compare-eraa` permet de désactiver la comparaison ERAA si désiré, et `--force-eraa-run` permet de forcer le recalcul du cache.

---

## 3. Architecture Technique et Solutions Proposées

### 3.1. Affichage des infos de capacité installée de chaque pays
- **Lecture des capacités** : Extraction des capacités pour chaque pays depuis les dictionnaires chargés (`country_capacities`) ou depuis les fichiers `.json` de chaque pays.
- **Tableau des capacités installées (`format_installed_capacities_table`)** :
  - Tableau ASCII aligné présentant les pays en lignes et les filières au niveau d'agrégation `input/countries` en colonnes (exprimées en GW avec 1 décimale).
  - Ligne de total européen `TOTAL EUROPE` pour chaque filière.
  - Colonnes de synthèse :
    - `Total Capa (GW)`
    - `Part Décarbonée (%)` (part des filières non fossiles : nucléaire, EnR, stockage, flexibilité, bioénergies)
    - `Overnight CAPEX (B€)`
    - `CAPEX Annualisé (M€/an)`
- **Intégration** :
  - Affiché dès le chargement des fichiers (visible immédiatement même avec `--dry-run`).
  - Intégré dans le rapport de synthèse de chaque Europe.

### 3.2. Évolution incrémentale des capacités
#### A. Comparaison aux investissements ERAA 2025 et ERAA 2033
- **Chargement des données ERAA de référence** :
  - Création de la fonction `load_eraa_country_capacities(year: int) -> Dict[str, Dict[str, float]]` lisant les fichiers `data/ERAA_2023-2/generation_capas/generation-capa_{year}_{country}.csv` pour les 7 pays disponibles (`benelux`, `france`, `germany`, `iberian-peninsula`, `italy`, `poland`, `scandinavia`).
  - Mappage des types de production bruts vers les noms de filières agrégées utilisés dans `countries/{country}.json` (`aggreg_prod_types_def.generation_capas`).
  - Sélection de la puissance installée appropriée pour chaque type d'actif : `power_capacity`, `power_capacity_turbine`, ou `power_capacity_injection`.
- **Calcul des deltas d'investissement** :
  - $\Delta_{2025} = \text{Capa}_{\text{actuelle}} - \text{Capa}_{\text{ERAA 2025}}$ (investissements nets réalisés depuis le point de départ 2025).
  - $\Delta_{2033} = \text{Capa}_{\text{actuelle}} - \text{Capa}_{\text{ERAA 2033}}$ (écart à la feuille de route officielle 2033).
- **Formatage du tableau comparatif des capacités (`format_capacity_comparison_table`)** :
  - Affichage comparatif par pays et filière : `Actuel` vs `ERAA 2025` vs `ERAA 2033` avec écarts absolus (GW) et relatifs (%).

#### B. Évolution à travers les runs successifs (historique incrémental)
- **Stockage des capacités** : Enrichissement des dictionnaires de métriques et des fichiers JSON d'historique (`comparison_europe1_vs_europe2_*.json`) avec le dictionnaire complet des capacités de chaque pays pour Europe 1 et Europe 2.
- **Grand tableau unifié de trajectoire incrémentale (`format_incremental_capacities_trajectory_table`)** :
  - Confrontation directe multi-colonnes des capacités installées :
    `Country / Technology | ERAA 2025 | ERAA 2033 | Run 1 (E1) | Run 1 (E2) | Run 2 (E1) | Run 2 (E2) | ... | Δ Net (E1) | Δ Net (E2)`
  - Affichage détaillé par filière pour chaque pays et pour `TOTAL EUROPE`.
  - Résumé synthétique par pays (`format_incremental_capacities_summary_table`) : capacités totales (GW) et part décarbonée (%) au fil des jalons ERAA et des runs successifs.
  - Sauvegarde automatique sous `output/runner/history_capacities_trajectory.txt`.
- **Calcul de l'évolution des capacités dans l'historique de simulation** :
  - Section dédiée `[CAPACITY EVOLUTION ACROSS SUCCESSIVE RUNS]` dans `format_simulation_evolutions_table` affichant pour chaque pays les modifications incrémentales de capacités décidées par les étudiants entre chaque itération.

### 3.3. Comparaison à 4 voies des runs aux capacités ERAA pures (ERAA 2025 et ERAA 2033)
- **Conditions de simulation identiques ("sur la même année de test")** :
  - La simulation de test s'exécute avec les paramètres communs : même `selected_target_year` (ex. 2033), même `selected_climatic_year` (ex. 1989), et même période (`uc_period_start` à `uc_period_end`).
  - Baseline pure ERAA 2025 : capacités injectées = capacités pures ERAA 2025 de tous les pays.
  - Baseline pure ERAA 2033 : capacités injectées = capacités pures ERAA 2033 de tous les pays.
- **Moteur de cache des baselines** :
  - Fichier de cache : `output/runner/baselines/eraa_baseline_{year}_target{target_year}_cy{climatic_year}_{period_tag}.json`.
  - Si le fichier de cache existe, chargement direct des métriques.
  - Si le fichier n'existe pas (et si solver actif / non `--dry-run`), exécution de la simulation PyPSA pour la baseline et écriture du cache.
- **Dashboard comparatif à 4 voies** :
  - Tableau global comparant : **EUROPE 1 | EUROPE 2 | ERAA 2025 | ERAA 2033 | DELTA (E1-ERAA33) | DELTA (E2-ERAA33)**.
  - Métriques comparées :
    - Coût total du système (M€/an)
    - Coût opérationnel OPEX (M€/an)
    - CAPEX annualisé (M€/an)
    - CAPEX overnight (B€)
    - Coût moyen (€/MWh)
    - Émissions CO2 (Mt)
    - Production totale (TWh)
    - Défaillance / ENS (MWh)
    - Heures de défaillance maximales (h)
  - Tableau par pays comparant les coûts, le CO2 et l'ENS entre les 4 mix.
- **Sauvegarde** :
  - Export texte complet dans `output/runner/comparison_europe1_vs_europe2_*.txt`.
  - Export JSON enrichi avec les 4 jeux de métriques et les capacités dans `output/runner/comparison_europe1_vs_europe2_*.json`.

---

## 4. Fichiers impactés & Modifications

| Fichier | Nature de la modification |
| :--- | :--- |
| `src/include_runner/investments_runner.py` | • Ajout de `load_eraa_country_capacities(year)`<br>• Ajout de `format_installed_capacities_table` et `display_installed_capacities`<br>• Ajout de `compute_capacity_comparison_with_eraa` et `format_capacity_comparison_table`<br>• Ajout de `get_or_run_eraa_baseline` (avec gestion de cache)<br>• Ajout de `compute_four_way_comparison` et `format_four_way_comparison_table`<br>• Mise à jour de `save_comparison_results`, `load_simulation_history`, `compute_simulation_evolutions`, `format_simulation_evolutions_table` |
| `main_runner.py` | • Ajout des options CLI `--no-compare-eraa`, `--force-eraa-run`<br>• Affichage des capacités installées dès le chargement<br>• Intégration de l'exécution/chargement des baselines ERAA<br>• Affichage du dashboard comparatif à 4 voies et de l'évolution incrémentale des capacités |
| `tests/long_term_uc/test_investments_runner.py` | • Nouveaux tests pour le chargement des capacités ERAA 2025/2033<br>• Tests de formatage du tableau des capacités installées<br>• Tests de comparaison des capacités vs ERAA 2025/2033<br>• Tests du cache des baselines ERAA et comparaison 4 voies<br>• Tests de l'évolution incrémentale des capacités dans l'historique |
| `doc/teachers/runner_tutorial.md` | • Mise à jour de la documentation pour expliquer les tableaux de capacités installées, la comparaison avec ERAA 2025/2033 et les options de cache |
| `TODO` | • Mise à jour des tâches complétées |

---

## 5. Déroulement par Étapes Incrémentales

### Étape 1 : Capacités Installées par Pays & Comparaison aux Références ERAA (Phases 1 & 2)
1. **Implémentation de `load_eraa_country_capacities(year)`** :
   - Chargement des CSV `generation-capa_{year}_{country}.csv`.
   - Agrégation par pays et filière selon `aggreg_prod_types_def`.
2. **Implémentation de `format_installed_capacities_table` & `display_installed_capacities`** :
   - Tableau ASCII des capacités installées en GW par pays et total Europe.
   - Calcul part décarbonée et CAPEX overnight / annualisé.
3. **Implémentation de `compute_capacity_comparison_with_eraa` & `format_capacity_comparison_table`** :
   - Calcul des écarts $\Delta_{2025}$ et $\Delta_{2033}$ par filière et pays.
   - Tableau comparatif imprimable.
4. **Intégration dans `main_runner.py`** :
   - Affichage immédiat dès le chargement des configurations d'investissements, y compris avec `--dry-run`.
5. **Validation par tests unitaires** dans `tests/long_term_uc/test_investments_runner.py`.

### Étape 2 : Suivi Incrémental des Capacités dans l'Historique des Runs (Phase 3)
1. **Sauvegarde des capacités par pays** dans `comparison_europe1_vs_europe2_*.json`.
2. **Extension de `compute_simulation_evolutions`** pour extraire les capacités au fil des runs et calculer $\Delta_{\text{prev}}$ et $\Delta_{\text{net}}$.
3. **Mise à jour de `format_simulation_evolutions_table`** avec la section `[CAPACITY EVOLUTION ACROSS SUCCESSIVE RUNS]`.
4. **Validation par tests unitaires** de l'évolution incrémentale.

### Étape 3 : Baselines ERAA Pures & Comparaison à 4 Voies (Phases 4 & 5)
1. **Implémentation de `get_or_run_eraa_baseline`** avec mise en cache dans `output/runner/baselines/`.
2. **Implémentation de `compute_four_way_comparison` et `format_four_way_comparison_table`**.
3. **Ajout des arguments CLI `--no-compare-eraa` et `--force-eraa-run` dans `main_runner.py`**.
4. **Persistance des résultats 4 voies** dans `save_comparison_results`.
5. **Validation par tests complets**, mise à jour de `doc/teachers/runner_tutorial.md` et de `TODO`.

---

## 6. Stratégie de Vérification et Tests

1. **Validation unitaire** :
   - Vérifier que `load_eraa_country_capacities(2025)` et `(2033)` retournent les bonnes valeurs numériques (ex. France nucléaire 61 761 MW, solaire 2025 18 185 MW, solaire 2033 54 282 MW).
   - Vérifier la robustesse des calculs de deltas face aux divisions par zéro et valeurs manquantes.
   - Tester la sérialisation/désérialisation du cache des baselines.
2. **Validation d'intégration (`pytest tests/long_term_uc/test_investments_runner.py`)** :
   - Tester l'exécution complète du pipeline avec `--test` et `--dry-run`.
   - Vérifier que `--clean` supprime correctement les fichiers sans casser le fonctionnement.
3. **Validation de non-régression** :
   - Exécuter la suite existante `pytest tests/` pour s'assurer qu'aucun autre composant n'est altéré.
