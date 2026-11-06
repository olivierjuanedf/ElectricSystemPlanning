# Prétraitement d'Élargissement Minimal des Bornes Hydroélectriques ($V_{\min}/V_{\max}$) par Optimisation Linéaire

## 1. Contexte et Problématique

Dans le cadre des simulations de Unit Commitment (UC) européen long terme (`my_little_europe_lt_uc.py`), l'activation des contraintes de stocks hydrauliques cibles (issues de `data/ERAA_2023-2/hydro/PECD-hydro-weekly-reservoir-min-max-levels.csv`) conduisait systématiquement le solveur d'optimisation (HiGHS) à un statut d'**infaisabilité stricte** (`infeasible`).

### Causes Racines Identifiées :

1. **Incompatibilité physique au démarrage ($SoC_{\text{init}}$) :**
   - Par défaut, les réservoirs sont initialisés à $80\%$ de leur capacité ($SoC_{\text{init}} = 0.80 \times S^{\max}$).
   - Or, en semaine 1, les données cibles imposent une borne supérieure plus basse : $V_{\max} = 79.0\%$ pour la France et $V_{\max} = 72.5\%$ pour la péninsule Ibérique.
   - Comme la contrainte s'applique dès l'heure 0 ($t=0$), le réservoir ne peut physiquement pas évacuer $1\%$ à $7.5\%$ de sa capacité énergétique en une seule heure (la puissance maximale de turbinage horaire représente moins de $0.09\%$ de la capacité pour la France).

2. **Anomalie d'échelle sur les données de Scandinavie :**
   - Dans le fichier ERAA, les profils de Scandinavie oscillent entre $0.25$ et $4.79$ (échelle sur $\approx 5$ au lieu d'un per-unit $[0, 1]$ standard).
   - Le clipping naïf précédent à $[0, 1]$ forçait $V_{\min} = V_{\max} = 100\%$ constant pour toutes les semaines, ce qui exigeait de maintenir le réservoir scandinave saturé en permanence depuis un niveau initial de $80\%$, en contradiction avec les apports naturels limités en hiver ($\sim 2.7\%$ par semaine).

3. **Infaisabilité inter-périodes :**
   - La vitesse de variation du volume d'eau entre deux périodes est physiquement bornée par les apports naturels entrants $I_w$ (remplissage maximal sans production) et la capacité maximale de turbinage $P_{\text{nom}} \times 168\text{ h}$ (vidange maximale à pleine puissance).
   - Si les cibles imposent des pentes plus raides que ces limites physiques, le problème devient mathématiquement sans solution.

---

## 2. Principe de la Solution : Micro-LP de Relaxation Minimale

Pour résoudre ces infaisabilités tout en restant le plus fidèle possible aux profils de gestion saisonnière ERAA, un **prétraitement par optimisation linéaire auxiliaire (micro-LP)** a été développé.

Avant l'optimisation UC globale, un petit problème linéaire est résolu de manière indépendante pour chaque réservoir hydraulique sur l'horizon de simulation.

### Formulation Mathématique du Micro-LP :

Pour un réservoir $u$, sur un horizon de $N$ snapshots ($t = 0, \dots, N-1$) avec $\mathcal{T}_{\text{target}}$ l'ensemble des instants contraints par les données hebdomadaires :

#### Variables de décision :
- $S_t \in [0, S^{\max}]$ : niveau de stock à l'heure $t$ (MWh)
- $p_t \in [0, P_{\text{nom}}]$ : puissance turbinée à l'heure $t$ (MW)
- $\text{spill}_t \in [0, I_t]$ : déversement éventuel (limité aux apports entrants naturels)
- $\delta^{\min}_k \ge 0$ : abaissement de la borne inférieure pour le $k$-ième instant cible
- $\delta^{\max}_k \ge 0$ : élévation de la borne supérieure pour le $k$-ième instant cible

#### Fonction Objectif :
$$\min \sum_{k \in \mathcal{T}_{\text{target}}} \left( \delta^{\min}_k + \delta^{\max}_k \right) + 10^{-5} \sum_{t=0}^{N-1} \text{spill}_t$$

- La norme $L_1$ garantit d'élargir les bornes **juste ce qu'il faut** pour qu'au moins une trajectoire physiquement réalisable existe.
- Si le couloir d'origine est déjà réalisable, $\delta^{\min}_k = \delta^{\max}_k = 0$, et les bornes cibles restent strictement intactes.
- La pénalité infinitésimale sur le déversement empêche de vider inutilement de l'eau.

#### Contraintes Physiques :
1. **Équilibre massique hydraulique horaire :**
   $$S_0 + \frac{1}{\eta} p_0 + \text{spill}_0 = SoC_{\text{init}} + I_0$$
   $$S_t - S_{t-1} + \frac{1}{\eta} p_t + \text{spill}_t = I_t \quad \forall t \in \{1, \dots, N-1\}$$
2. **Encadrement assoupli aux instants cibles :**
   $$S_{t_k} \ge V^{\min}_{t_k} - \delta^{\min}_k \iff -S_{t_k} - \delta^{\min}_k \le -V^{\min}_{t_k}$$
   $$S_{t_k} \le V^{\max}_{t_k} + \delta^{\max}_k \iff S_{t_k} - \delta^{\max}_k \le V^{\max}_{t_k}$$
3. **Bornes de relaxation :**
   $$0 \le \delta^{\min}_k \le V^{\min}_{t_k}, \quad 0 \le \delta^{\max}_k \le S^{\max} - V^{\max}_{t_k}$$

#### Sortie et Bande Passante Minimale :
Les nouvelles bornes relâchées sont calculées par :
$$\widetilde{V}^{\min}_{t_k} = \max\left(0, V^{\min}_{t_k} - \delta^{\min*}_k\right)$$
$$\widetilde{V}^{\max}_{t_k} = \min\left(S^{\max}, V^{\max}_{t_k} + \delta^{\max*}_k\right)$$

Pour garantir que le solveur UC principal conserve de la souplesse pour arbitrer la production au meilleur coût économique, une **bande passante minimale** ($\Delta_{\min} = 5\% \times S^{\max}$) est systématiquement maintenue entre $\widetilde{V}^{\min}$ et $\widetilde{V}^{\max}$.

---

## 3. Architecture Logicielle et Fichiers Modifiés

```
ElectricSystemPlanning/
├── Makefile                                    # [CRÉÉ] Cibles d'automatisation (test-hydro, run-europe, test)
├── src/
│   └── include/
│       ├── hydro_bounds_preprocessor.py        # [CRÉÉ] Module de relaxation LP et normalisation
│       └── dataset_builder.py                  # [MODIFIÉ] Intégration dans add_hydro_extreme_levels_constraint
├── tests/
│   └── long_term_uc/
│       └── test_hydro_bounds_slack.py          # [CRÉÉ] 4 tests unitaires du prétraitement
└── doc/
    └── long_term_uc/
        └── hydro_bounds_preprocessing.md       # [CRÉÉ] Présente documentation
```

### Détail des composants :
- **`src/include/hydro_bounds_preprocessor.py` :**
  - `relax_single_hydro_reservoir_bounds_lp` : Résolution du micro-LP pour une unité hydroélectrique avec SciPy (`method='highs'`) et matrices creuses CSR.
  - `preprocess_all_hydro_soc_bounds` : Itérateur sur l'ensemble des réservoirs du réseau PyPSA, extrayant automatiquement les puissances, stocks initiaux et séries temporelles d'apports.
- **`src/include/dataset_builder.py` :**
  - Dans `add_hydro_extreme_levels_constraint`, les bornes en per-unit sont filtrées par le prétraitement avant d'être converties en MWh et injectées dans le modèle d'optimisation Linopy.

---

## 4. Résultats et Validation

### 1. Tests Unitaires (`make test-hydro`)
Quatre scénarios de validation ont été testés et validés en **0.62 seconde** :
1. `test_hydro_bounds_slack_initial_soc_conflict` : Rehaussement exact de $V_{\max}$ à $t=0$ de $70\%$ à $\ge 79\%$ pour rendre réalisable un $SoC_{\text{init}} = 80\%$ avec une capacité de turbinage horaire de $1\%$.
2. `test_hydro_bounds_slack_out_of_scale_scandinavia` : Normalisation automatique d'une série $[0.25, 4.79]$ ramenée dans $[0, 1]$ et vérification de la faisabilité.
3. `test_hydro_bounds_slack_already_feasible_unchanged` : Conservation stricte des bornes déjà réalisables ($\delta = 0$).
4. `test_hydro_bounds_slack_bandwidth_enforcement` : Élargissement symétrique pour garantir $\Delta_{\min} \ge 5\%$.

### 2. Validation sur la Simulation Européenne Complète (`make run-europe`)
Lors de l'exécution de `my_little_europe_lt_uc.py` :
- **Ajustements automatiques observés :**
  - France (`fra_hydro_reservoir`) : relaxation de $V_{\max}$ de $+90.4\text{ GWh}$ au démarrage.
  - Espagne (`ibe_hydro_reservoir`) : relaxation de $V_{\max}$ de $+1\,189.7\text{ GWh}$ pour respecter la pente de vidange.
  - Scandinavie (`sca_hydro_reservoir`) : détection de l'échelle hors-norme ($4.397 > 1.0$) et normalisation par $5.0$.
- **Convergence de l'optimisation :**
  - Solveur HiGHS : statut `Optimal` obtenu en **17.88 secondes** (76 608 variables, 281 794 contraintes).
  - Génération complète des résultats dans `output/long_term_uc/data_preliminar_9/` (décisions de dispatch, flux d'interconnexion, métriques UC `uc-summary_europe_*.json`).

### 3. Suite Globale de Tests (`make test`)
- `tests/long_term_uc/test_hydro_bounds_slack.py` : **4 passed** (0.74s)
- `tests/long_term_uc/test_europe_lt_uc.py` : **1 passed** (19.32s)
