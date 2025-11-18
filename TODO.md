
TODO

MAIN actions (M)
M0) Update readme.md with new names for project etc...
MObis) Check marginal cost/efficiency values in elec-europe_params_fixed.json
M1) See "TODO[debug]"
M2) [CR] Voir "TODO[CR]"
M4) Prévoir appui (doc/mini-script ?) pour aider les étudiants à gérer les infaisabilités ? (bcp au début... surtout si on leur fait passer les embûches pédagos - ne pas mettre d'actif défaillance par ex !)
M7) Tester avec des dates start/end sans hh:mm
M8) Sortir/tracer les émissions CO2
M11) Version de Python ok?
M13) Vérif cohérence FuelNames avec ProdTypeNames -> utilité des 2 ?
M14) Virer les gitignore qui traînent...
M16) Tests is_stress_test case...
M19) Ajouter graphe camembert à la DJ -> to show congestion of intercos in an "aggreg. view"
M20) Save output graphs in html to allow interactive discussions
#################### LATER  ################
M5) Trier/simplifier JSON visibles des élèves -> pour que cela soit facile pour eux de rentrer dedans (ne leur laisser voir que les params utilisateurs). Et adapter doc en fonction
M17 (later)) Introduce aggreg. prod types -> "all_res". To avoid typing lists of all RES types for selection...
M18) Reformat data files description with file objects (folder, separators, column names...)

DATA (D)
D1) Add ERAA ed. 2024, with climatic modelling...

DATA ANALYSIS (DA) - before 1st UC run, to get an intuition of the pbs - my_little_europe_data_analysis.py
#################### LATER  ################
DA3) (improve code quality) Avoid creating Dataset object once per data analysis - getting once all data needed (however it should be done 
on the product of data needs -> more than needed in general)
DA5) Allow capacity plot/extract - over multiple years and dts?
DA6) Take into account fatal prod (ror) for net demand case
DA7) Replace [-2] by an adaptive index to refer to extra-params idx at some stages

TOY EX (TE) - my_toy_ex_italy.py
TE5) Fullfill long_term_uc/toy_model_params/ex_italy-complem_parameters.py with complem exs in Ita case (hydro, batteries)
TE7) Do NOT mention diff of PV key between capa and CF data -> confusing for the students...
#################### LATER  ################
TE2) Keep FUEL_SOURCES or too complex for the students?

MAIN EUROPE SIMUS (MAS) (my_little_europe_lt_uc.py)
MAS2) Integrate hydraulic storages... with inflows/min SOC data from ERAA
MAS3) Usage param auto fulfill interco capa missing -> ??
MAS4) Add possibility to set Stock (additional to ERAA data) in JSON tb modif input file
#################### LATER  ################
MAS5) Add possibility to provide additional fatal demand -> for iterations between UC and imperfect disaggreg of an aggregate DSR flex (EV for ex) modeled as a Stock for ex! (cf. OMCEP course)
MAS6) Reformat/simplify JSON params file (in input/long_term_uc/elec-europe_params_to-be-modif.json 
-> suppress "selec" prefix implicit for some params?

RUNNER (R) (main_runner.py)
R1) Finish v1 runner
-> automatically over multiple projects cloned?
R2) Script to generate some graphs/ppt(s) after launching the runner
-> for archive/discussing live with the students
R3) "Stress tests": blackout on some countries * pts; issue on some intercos; techno. breakthrough on some pts

PLOTS
P1) Eco2mix colors TB completed -> coal; and markets to distinguish agg_prod_type with same colors
P3) Check case with unique curve -> plot_dims obtained from UC ts name (call of def get_dims_from_uc_ts_name)

OTHERS
O1) Doc basic use of codespace out of the repot?
O3) / by efficiency in FuelSources and not * for primary cost?
O4) Iberian-peninsula -> Iberia
O7) Check multiple links between two zones possible. Cf. ger-scandinavia AC+DC in CentraleSupélec students hypothesis. 
And interco types (hvdc/hvac) ok? Q2Emmanuel NEAU and Jean-Yves BOURMAUD
#################### LATER  ################
O2) Scripts avec qques exemples de base Python ? "[coding tricks]"
O5) Subpart of git with distinguished access-rights between students / TA for docs and data available?
(to avoid conflicts; changes leading to "un-necessary" bugs)
O6) Finish and connect type checker for JSON file values -> using map(func, [val]) and all([true])
-> OK excepting UsageParameters
