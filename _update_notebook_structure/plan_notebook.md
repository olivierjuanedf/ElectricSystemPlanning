# Commentaires suite à la lecture de la première version du notebook

### Remarques préliminaires sur le format de ce document
En premier nous avons mis des remarques générales. En second nous nous sommes notés des remarques pour harmoniser la suite du cours avec le notebook. Enfin nous avons déroulé le script .ipynb et fait des remarques parties par partie. Les titres des (sous-)sections et/ou leur n° sont indiqués pour s’y repérer !
Pour les remarques partie par partie :
Les passages surlignés en jaune sont des blocs de txt extraits du notebook
En rouge des questions (Qx), mais parfois aussi juste commentaires/suggestions

### Remarques générales sur le notebook 
Possible de faire des “chapitres” (et/ou différents notebooks) ?

 -> pour structurer l’ensemble ; bien montrer les différents éléments aux élèves, faire un découpage partie par partie pour faciliter notre gestion en classe du rythme auquel les élèves le parcourent, et… ne pas leur faire trop peur au début (s’ils font défiler le script jusqu’en bas !)

-> pour différencier des chapitres obligatoires, traités ensemble en classe et des chapitres optionnels pour les élèves rapides

TODO[Ca] : voir avec l’extension notebook -> vérif ok sur le codespace
Idée de structuration par chapitre

## Restructuration proposée

### Chapitres du cours - à traiter en classe avec tous les élèves

Imports -> idéalement géré au préalable à l’install/ouverture de l’env ? Ou bien intéressant de montrer aux élèves quelles dépendances seront utilisées ? TODO : ajouter dans les requirements pour que cela soit fait automatiquement (de manière invisible)

1. Power demand

2. Generation - Controllable generation technologies

3. Boiteux stacking method (1/2) - only with controllable generation technologies

4. Generation - Renewable generation technologies

5. Boiteux stacking method (2/2) - with controllable and renewable generation technologies

N.B. Ce découpage en chaps pourrait aussi mieux montrer la logique genrale, via entrées/sorties entre étapes

### Annexes - à ne pas traiter en classe avec tous les élèves, à garder pour les plus curieux / rapides

A. Controllable generation technologies - Advanced 

B. Detailed economic and environnemental characteristics of these technologies 
Renewable generation technologies - Advanced 

C. Detailed economic and environnemental characteristics of these technologies 
Storage (partie 5 actuelle). Q : plutôt stocks CT en fait ?

 * Principe du stockage et gestion heuristique 

 -> déformation de la LDC

Point tout de même important pour introduire les élèves aux besoins et difficultés de gestion des stockages à différentes échelles temporelles (stocks courts ou saisonniers) ? 

Prévoir de finir la partie Boiteux en milieu d’aprem, pour avoir un peu de temps sur ça en fin d’aprem ?

D. Optimization models (partie 6 et 7 actuelles) 

* Optim écrite en PuLP côté Kamil ; ferait double dose de découverte modeler dans la sem. (PuLP & PyPSA)

* Même sans regarder le code, possibilité d’obtenir des sorties plus cohérentes avec ce module - en ajoutant des contraintes “politiques” notamment (CO2 par ex.). NB LP paramétrable : (0) Monozone ? (i) investMin/Max par techno ; (ii) réduction émis CO2 vs. situation de réf. ; (iv) Pas de modélisation des stocks (en fait si dans la Sec. 6 du notebook). Pb d’optim EOD simplifié - se rapprochant de la “modélisation” par monotone ; en ajoutant des contraintes pour améliorer réalisme décisions invest

## Proposition : Ne pas parler des années climatiques (CY), et garder ce concept pour plus tard dans la semaine 
Actuellement plusieurs parties du notebook intègrent plusieurs CY. Nous proposons de ne garder qu’une seule CY dans le tronc commun du notebook, et de leur expliquer le concept de CY le mercredi ou le jeudi.
	L’idée serait de choisir une année climatique pour faire le notebook, au lieu d’utiliser une moyenne des CY (concept de moyenne de CY qu’il serait d’ailleurs intéressant de discuter avec les élèves dans la semaine si on a le temps)
TODO : enlever la moyenne ; prendre plutôt une année moyenne
On peut aussi prévoir de garder cette partie dans une annexe pour les curieux / élèves ayant bien avancé ?

WS vs CY ? TODO : basculer en CY pour ê cohérent avec terminologie éd. ERAA2023 (notebook conçu initialement en ERAA2024)

Proposition : Utiliser l’année (TY) 2033 dans la majeure partie du notebook
Pour alléger et simplifier, nous proposons de ne considérer que l’année cible 2033 dans la majeure partie du notebook (en particulier pour la théorie de Boiteux).
La comparaison entre 2025 et 2033 reste intéressante pour la première partie qui étudie la demande, mais pour le reste cela permettrait d’alléger et de garder le fil conducteur pour les élèves.

Penser à réutiliser certaines parties du notebook dans les jours suivants du cours ?

## Sauvegarder certaines figures dans un dossier ? 
Sauver certaines données et figs, même manuellement, dans le dossier de sortie “data analysis” (-> pour rapport/prez finale des élèves)

TODO : 
Figs -> ajouter fait de sauver les figs dans un dossier ? Ou bien cliquer sur les figs suffit - montrer aux élèves qui ne sont pas habitués
Data -> le coder

## Autres questions 
* (théorique) Possible de réfléchir “en delta” avec la méthode. de Boiteux - partant du parc 2025 et choisissant les invests 2033 (quid usage du parc 2025 pour construire la LDC - profils optimisés des actifs dispatchables)
Si module optim gardé dispatch connu en sortie
(fonctionnel) Possible de n’afficher que les X premières lignes des blocs de code - à n’ouvrir que par les élèves intéressés (sinon exécution seule)
-> Dispo, en double cliquant à gauche d’un bloc (seul commentaire devrait rester). TODO : vérif dispo dans le codespace

* (fonctionnel) Possible d’avoir des cases blanches pour bien identifier les zones où remplissage attendu ?

TODO : créer cellule md. Voir si possible de lui donner un style particulier -> sinon tag (“answers”) repérable dans le menu déroulant “outlines” (via sous-section)
(impact participation DSR) Bcp plus forte dans les données ERAA ed.2024 2035 vs. 2026 -> décisions d’invest (EnR) notamment impactées



## Remarques pour harmoniser le reste du cours avec le notebook (Camille / Cécile / Olivier)

* Préciser le fil rouge de la semaine le lundi 
    -   Préciser le déroulé de la semaine, et l’aspect fil rouge : objectif, trouver les capacités à installer pour répondre à la demande en 2033
    -    Le cadre : les différents pays
    -    Les données : expliquer les données ERAA + le fait que ce sont des Projections
    -    Les années :

        o   Année de départ : 2025 (capacités installées pour 2025 en point de départ)

        o   Année cible : 2033 (on cherche à définir les capacités à installer pour 2033)

* Préparer les éléments de cours associés au notebook
Quelles pauses de cours ? (base théorique nécessaire pour faire la suite du notebook en comprenant bien les choses)

* Conserver un temps d’échange à la fin de la journée sur le notebook
Pour que les élèves puissent se présenter ce qu’ils ont fait pendant la journée
Penser à des cas particuliers “troublants” pour les élèves
Scandinavie sans hydro./avec hydro. mal gérée. Si un groupe d’élèves avance bien possible d’explorer avec un placement heuristique de l’hydro a priori
Prochaines étapes
Propal reformatage notebook pour avoir un tronc commun plus simple/court (chaps)
Valider avec Kamil
Le faire
Tester sur nos pays en se partageant le boulot entre Cé, Ca et O (2 pays/pers.). Et Alex ?
5. (Idées pour plus tard)
Décomposition de Fourier demande (résiduelle) pour identifier les différents cycles ?

## Remarques étape par étape lors du déroulé du Data Crunch Notebook
 Data Crunch Notebook
Goals of this notebook:
- Sandbox to discover your study case country's profile
- Discover key indicators that will help shape the system
- Discover new concepts & methods: load duration curves, controllable/fatal generation, baseload/peakload, Raw/Net load, Boiteux stacking
- Understand main power generation technologies' characteristics (technical, environmental & economic)
- Simulate various power mix
Expected Output:
- Choice of power mix for your study case country -> Input for PyPSA and the rest of the week
Q1 : préciser explicitement qu’il s’agira nécessairement d’un 1er mix « Mix n1 », nécessairement naïf vs ceux qui suivront lors de la semaine ? (pour faire passer dès le début ce côté progressif/fil rouge de la sem)
1 - Imports
Q2 : (pour dev., test en local par ex.) prévoir requirements_data-crunch.txt ? (pour les élèves) gérer cela auparavant, au moment de charger + « paramétrer » le codespace ?
è Dépendances (Py 3.10) numpy, pandas, matplotlib, seaborn, plotly
2- Choice of your study case
Choose the country (or group zone) that was assigned to your group
Q3 : « country (or group zone) » -> « (meta-)country », pour essayer d’unifier les termes tout au long de la semaine ?
country_choice = "iberian-peninsula"
#List of years considered for this practical work
ERAA_years = [2025,2033]
Q4 : aligner noms de vars / params sur ceux utilisés dans le reste du projet de code ? (pour faciliter prise en main par les élèves tout au long de la sem)
·        country_choice -> country
·        ERAA_years -> target_years (cf. fichier elec-europe_eraa-available-values.json dans le dossier input)
Q4bis : idéalement faire comprendre aux élèves dès le data crunch que les valeurs dispos sont celles indiquées dans le fichier fichier elec-europe_eraa-available-values.json :
·        country à choisir dans la liste associée au champ countries
·        TY obtenues à partir de ce fichier json, en les affichant pour que les élèves en prennent connaissance
3 – Power Demand
## 3.1 - Download of gross demand data
Q5 : (i) Unifier les termes tout au long de la sem. : « gross demand » pas utilisé dans le reste du code a priori -> « demand » à la place ? (et « X demand » si demande spécifique à évoquer) ; (ii) (Mineur !) « Download » -> « Read/get » pour ê tout à fait précis (car données déjà dans l’env.)
Q6 : variable « pivoted » à supprimer, en réduisant le bloc de code à 2 lignes – pour « minimiser » taille du code seulement fonctionnel (pas le cœur du cours) ?
Renaming the climatic years for clarity
Q6 : point épineux selon moi, car (de mémoire !) dans ERAA <= 2023 :
·        Le terme « CY » (Climatic Year) est bien employé pour cette « dimension » des données, et « WS » (Weather Scen.) n’est utilisé qu’à partir de l’ERAA >= 2024….
·        Mais les données « climatiques » sont générées en reproduisant des météos passées donc en fait correspondent plutôt à l’acronyme « WS » (Weather Scen.)
ð  Mettre « CYx » dans ce cas comme les données 2023.2 sont utilisées – pour rester alignés sur la déf. des données ERAA ? Ou bien les modifier – sur l’ensemble de notre env… - pour coller à leur nature ?
***WARNING***: the weather scenarios are not equally probable! The "average whether scenario" is therefore not really the average observed climate
Q7 : idéalement il faudrait qu’on accorde nos violons entre séance optim (Cécile) / climat (Boutheina) et les échanges autour des simus – dont data crunch. Pour moi jusqu’à l’édition ERAA2024 l’usage qui y fait de ces scénarios :
·        Phase de Capa Planning (Eco Viability Assessment, EVA) -> 3 CYs avec des probas non uniformes (1 très dominante d’ailleurs, ~85%)
·        Phase UC(ED) -> simu de Monte-Carlo avec probas uniformes sur ces scénarios
3.2 – Data Analysis (Average weather scenario)
***WARNING***: the weather scenarios are not equally probable! The "average whether scenario" is therefore not really the average observed climate
Q8 : éviter de répéter ce WARNING ? Le donner + discuter 1 bonne fois pour toutes
*(Run this command in terminal if needed : pip install --upgrade nbformat)*
Q9 : gérer toutes les Qs d’install dès le début pour les élèves
common dummy year (e.g., 2000)
Q10 : prendre plutôt 1900 pour ê plus « carricatural » dans le fait de faire comprendre qu’il ne s’agit pas d’un vrai calendrier ? (1900 retenu dans le dossier « data » de l’env.)
[Figure « Hourly electricity demand »]
Q11 : (pour les figures de demande et les suivantes) idéalement utiliser les couleurs par année (TY) définies dans le fichier src/functional_params/plot_params.json (champ « year ») -> déf. unique centralisée dans l’ensemble de l’env pour les élèves
Q12 : (détails de forme) (i) Améliorer labels des x « MarCH », « JulY », etc. (comme il y a la place en plus) ; (ii) Passer les données en GW pour éviter les facteurs « k » sur l’axe des ordonnées (d’avoir à faire un « calcul » pour comprendre ce qui est tracé)
Questions : (…) (après cette figure)
Q13 : les n° pour pouvoir y faire réf./échanger plus facilement dessus avec les élèves ?
3.3.2 - Usefull indicators for profile analysis
Peak Load
Q14 : préciser -> annuelle ?
Seasonal Variation
Q15 : max – min -> « spread » plus standard/précis ? Idem Daily Variation. Si chgt de nom, modifier aussi le nom des vars dans le code
results.append({
        "Year": year,
        "Total_Demand_TWh": total_demand_mwh,
        "Peak_Load_GW": peak_load,
        "Seasonal_Variation_GW": seasonal_variation,
        "Max_Daily_Variation_GW": daily_variation,
    })
Q16 : delta entre les unités dans les clés de ce dict et dans les noms de vars (ainsi que dans les calculs ?)
fig.add_trace(
go.Bar(
x=results_df.index,
y=results_df["Peak_Load_GW"]/1e3,
text=(results_df["Peak_Load_GW"]/1e3).round(0),
textposition="auto",
name="Peak Load (GW)"
),
row=1, col=2
)
Q17 : une petite boucle avec un dict pour factoriser ces 4 bouts de code ? (synthétiser le code ; montrer sa structure « essentielle » si jamais qques élèves voulaient y jeter un œil)
[Fig. « Elec demand in {country}…]
Q18 : (i) Utiliser couleurs de plot_params.json pour les CY ; (ii) Centrer les xticks, en ne conservant que les (2) années tracées ;
Distribuiton of weekly variation
[Fig. « Daily and Weekly Variations…]
Q19 : (i) Utiliser couleurs de plot_params.json pour les CY (et différencier daily/weekly par subplot horizontaux ?)
Q20 : typo à “Distribuiton”
[Fig. « Distribution of Daily and Weekly Demand Variation]
N.B. Trop bien le graphe interactif 😊 !
Q21 : (i) Passer en GW ; (ii) Pas sûr de comprendre l’info supplémentaire apportée par les triangles en pointillé ; (iii) Diminuer un peu le nbre de chiffres signifs ; (iv) « Weekly variation » est la distribution, sur les sems, des écarts entre la variation moyenne sur les jours de sem. et celle sur les jours de we (au sein d’une sem. donnée) ? Intérêt de cette métrique ?
3.2.3 - Load Duration Curve
[Fig. « Load duration curves…]
Q22 : préciser un peu le xlabel, pour que les élèves percoivent bien que les idx ont été ordonnés selon demande décroissante
Interpret this graph
3.3.3 Load duraiton curve
Questions:
(…)
- Do it echoes the description of the climatic years presented at the beginning of the section?
Q23 : caracts des CY données (chaud/inter./froid) ? Ou autre sens à cette Q ?
3.4 Risk management and unsupplied energy
Q24 : penser le lien avec prez de Cécile (rapide aparté sur les façons de prendre en compte cela dans les pbs d’optim UC -> (i) pondération via VoLL ; (ii) volume max de défaillance si en Energy Not Served ? (iii) ouverture sur des « épisodes de défaillance » spécifiques… cf Dunkelfläute…)
The VOLL considerd by ERAA (in technology.json) is set at **20000€/MWh**
Q25 (pour CaCéO !) : récupérer / intégrer ce technology.json, qui comprend pas mal de params que nous n’avions pas récupérés pour l’instant – dont les CAPEX
Q26 : « 20000 » -> ajouter esp. des milliers pour faciliter lecture ?
(Ces concepts sont bien utilisés par la suite dans le notebook ?)
4 – Generation
4.1 - Notion of controllable and variable generation
**Notebook Break**: in class discussion on the concepts of controllable and variable generation, gross and net load
Q27 : pour « bonne progressivité » de cette session, avoir fait plus tôt un break sur la demande (nette) ? (concepts nécessaires pour bien comprendre certains des graphes donnés auparavant)
4.2
As the power mix was not built including vRES, the
Q28 : (i) préciser qu’on parle du mix « originel » (ou d’une époque révolue) ; que les élèves ne s’y trompent pas ! (ii) Bien déf. vRES au 1er usage, et si possible il faudrait qu’on unifie dans l’ensemble de l’env. (de notre côté on est sur VRE dans glossary.md pour l’instant)
4.2.1
Q29 [@CaCéO] : intégrer base load vs peak load PP au glossaire de l’env.
Q30 (sur la description des actifs) : (i) « PP » -> pas forcément utile ? (factorisé en tête de liste) ; (ii) « typical » ou « typically » ?
Q30bis [@CaCéO] : intégrer ce descriptif ailleurs dans l’env. ?
4.2.2
Q31 (envir.) : « CO2eq.  emissions » -> pas forcément évident pour les élèves ; le déf. à l’écrit ? (a minima en dire 2 mots à l’oral !)
Q32 (eco.) : « fixed cost » vs. « variable costS” -> raison du pluriel non partagé ? De même bien prendre le tps de déf. (oralement) ces notions, à mon avis peu connues d’une part des élèves
Q32bis [@CaCéO] : acronymes dans le glossaire ?
def vpm(taux, nper, va, vc=0, type=0):
Q33 : (i) la passer en anglais ; (ii) « vpm » ? ; (iii) Bien prendre le tps d’expliquer ce concept… pas évident/nouveau pour une majorité d’élèves a priori ; (iv) Idéalement mettre en .tex les fules dans le header de la fonction -> pourrait permettre aux élèves les plus matheux de capter tout de suite leur « substance »
Q34 : var. « exclude_techs » -> plutôt « excluded_techs » (pas à 1 lettre près !) ?
Q35 : une petite boucle pour factoriser les 3 blocs de code
fig.add_trace(go.Bar(
    name="Overnight Investment",
    x=techs,
    y=invest_overnight
)) ?
Q36 [Fig. Annualized fixed cost by techno.] : (i) Passer en k€/MW*year pour le ylabel ; (ii) Logique ordre retenu – en abscisse – pour le tracé ?
Q37 [Fig. « Variation of fixed costs with WACC”] (et sur les autres) : « WACC » à déf. au 1er usage. Q37bis [@CaCéO] : dans le glossaire
Q38 : (i) Pointer sur les couleurs par techno. de plot_params.json ; (ii) Ordre, idem Q36 (ii)
Q39 : « but also on the co2 price” -> “CO2”
Q40 :
“# Parameter to change at will
co2_price = 100 # €/tCO2”
è Fournir un pointeur (site Internet) aux élèves pour qu’ils puissent observer des valeurs récentes réalistes ? (et/ou projections futures)
Q41 : factoriser 3 appels au code
fig.add_trace(go.Bar(
    x=technologies,
    y=fuel_costs,
    name="Fuel cost",
))
?
Q42 [Fig. « Variable costs répartition »] -> redéf. « VOM » sur la fig. ? Même si cela aura été fait auparavant, vu le nbre de nouveaux acronymes que les étudiants vont découvrir lors de cette session, le fait de rendre les figs autoportantes me semblerait un plus
Qx : quelques typos
·        « interpretE » (e final manquant)
·        « breifly »
·        (3.3.3) « How if each » -> « iS each »
·        (3.3.3) « Do it echoes » -> « Does it echo »
·        (4.1) « Questions: (…) - Do you think we could run a power based on 100% renewables? » -> « power system » 
·         

