
TODO

MAIN actions (M)
M2) [CR] Voir "TODO[CR]"
M3) Restructurer l'arborescence en enlevant le sous-dossier long_term_uc (présent à l'origine car il y avait aussi un autre sous-dossier)
M4) Prévoir appui (doc/mini-script ?) pour aider les étudiants à gérer les infaisabilités ? (bcp au début... surtout si on leur fait passer les embûches pédagos - ne pas mettre d'actif défaillance par ex !)
M5) Trier/simplifier JSON visibles des élèves -> pour que cela soit facile pour eux de rentrer dedans (ne leur laisser voir que les params utilisateurs). Et adapter doc en fonction
M6) Ajouter les carriers dans le modèle Pypsa pour faire disparaître les msgs d'erreur "The following storage_units/prod/etc. have carriers which are not defined"

DATA (D)
D1) Solar pv key to be aligned in capacity ("solar_pv") and CF data ("lfsolarpv") to avoid troubles/confusions...

DATA ANALYSIS (DA) - before 1st UC run, to get an intuition of the pbs - my_little_europe_data_analysis.py
DA1) Rendre fonctionnelle v1 de ce script !
-> (OB) OK commit 1351d06
DA2) L'enrichir... pour la séance de data crunch ? Ou bien laisser faire les étudiants sur Excel ?

TOY EX (TE) - my_toy_ex_italy.py
TE1) voir "XXX" (notamment les coding tricks)
TE2) conserver FUEL_SOURCES ou bien trop compliqué pour les étudiants ?
TE3) voir si certains warning sont de notre fait... même si en ligne il semble que PyPSA en génère pas mal - notamment en lien avec Linopy
TE4) doc/toy-model_tutorial.md to be completed/improved
TE5) Remplir long_term_uc/toy_model_params/ex_italy-complem_parameters.py avec des exs complémentaires au cas italien (hydrau, batteries)

MAIN EUROPE SIMUS (MAS) (my_little_europe_lt_uc.py)
MAS1) Doc doc/... pour clarifier les choses et permettre utilisation autonome
-> 1 doc par jour (session) pour ne pas "faire peur" au début avec un doc trop conséquent ?
MAS2) Integrate hydraulic storages... cf. CS student code
MAS3) Usage param auto fulfill interco capa missing -> ??
MAS4) Add possibility to set Stock (additional to ERAA data) in JSON tb modif input file
MAS5) Add possibility to provide additional fatal demand -> for iterations between UC and imperfect disaggreg of an aggregate DSR flex (EV for ex) modeled as a Stock for ex! (cf. OMCEP course)
MAS6) Reformat/simplify JSON params file (in input/long_term_uc/)
* elec-europe_params_to-be-modif.json -> suppress "selec" prefix implicit for some params?

PLOTS
P1) Eco2mix colors TB completed -> coal; and markets to distinguish agg_prod_type with same colors
P2) Add plot functions to get demand/cf/capas values for the selected values of params (and selected period) -> useful for 1st stage of data analysis (better with some graphs easy tb obtained)
P3) Set up per country colors so that all groups will obtain directly comparable graphs

OTHERS
O1) Doc b.a.-ba utilisation codespace en dehors du repot ?
O2) Scripts avec qques exemples de base Python ? "[coding tricks]"
O3) / by efficiency in FuelSources and not * for primary cost?
O4) Iberian-peninsula -> Iberia
O5) Sous-partie git avec accès différencié élèves / TA pour docs et données diff ?
(pour éviter conflits ; chgts menant à des bogues "non-nécessaires")
O6) Finish and connect type checker for JSON file values -> using map(func, [val]) and all([true])
-> OK excepting UsageParameters
O7) Check multiple links between two zones possible. Cf. ger-scandinavia AC+DC in CentraleSupélec students hypothesis. And interco types (hvdc/hvac) ok? Q2Emmanuel NEAU and Jean-Yves BOURMAUD
O8) See with OJ if regular runner possible (run all github with current state of code and plot current scores) -> script to calculate scores and plot results
