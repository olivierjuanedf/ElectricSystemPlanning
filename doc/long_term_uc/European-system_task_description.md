# European Long Term Unit Commitment Problem - Practical Session

## 1. Discovering the code environment

1.a Read [doc/PracticalSession3-European-System-Planning/European-system_tutorial.md](../PracticalSession3-European-System-Planning/European-system_tutorial.md)

1.b Check parameters in [input/long_term_uc/countries/](../../input/long_term_uc/countries/)*{**country**}.json* with "**country**" the name of your considered country 

1.c Check parameters in [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json). **For the moment you should be in “solo” mode. Put your country name in the field “team”.**

1.d Run [my_little_europe_lt_uc.py](../../my_little_europe_lt_uc.py) 
    - First to check that it works correctly.
    - What is the result of the optimization? Why?


## 2. Modeling a simple 1-country (the one you are responsible for) Unit Commitment problem 

2.1 Adjust the installed capacity in your country to obtain a feasible problem :
    - Using the results of the data analysis' session, update file  [input/long_term_uc/countries/](../../input/long_term_uc/countries/)*{**country**}.json* to add production and storage capacities to your country. 
    - Try different electricity mixes **until you obtain a feasible UC problem.**
    - Once your UC problem is feasible, check the results : data in csv files are available in [output/long_term_uc/multizones_eur/data](../../output/long_term_uc/multizones_eur/data) and plotted figures in [output/long_term_uc/multizones_eur/figures](../../output/long_term_uc/multizones_eur/figures). 

2.2 Adjust these capacities by seeking, while continuing to meet demand, to: 
    - Prevent any failure
    - Minimize costs (question: both operation and investments?)
    - Minimize CO2 emissions
    - Any other objective?
    What electricity mix do you get? Do these criteria lead to the same choices? 

N.B. The  simulations are done in a PyPSA-based dedicated environment . You can find more information on PyPSA in the following website: PyPSA documentation: https://pypsa.readthedocs.io/en/latest/. Note: we use the 0.35.1 version in this code environment and not version 1.0 (which has just been released).

## 3. Team Discussion: What objective(s) would you like to set for Europe in 2033? 

3.1 Discuss within your Europe group to decide on the objective you want to set together. 
The **goal for the rest of the week will be to gradually align your individual decisions (country by country) to achieve this objective**.

Once you have set your objective, prepare to present it to the rest of the class (10–15-minute presentation).


## 4. Modeling the European Union Unit Commitment problem 

### 4.1 Adjust the capacities the different countries to achieve the European objective
->WORK IN PROGRESS - TODO : change this section depending on how we want the students to work together
4.1.1 Experiment by playing with the parameters in files [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json) and [input/long_term_uc/countries/](../../input/long_term_uc/countries/){country}.json

4.1.2 Observe the impact on the optimal solution and on the prices when running [my_little_europe_lt_uc.py](../../my_little_europe_lt_uc.py).


### 4.2 Robustness Checks - Adapt your decisions to have a resilient European electicity system

Change the climatic year considered for the optimization in [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json) by changing **"selected_climatic_year"**

4.2.1 How would you construct an electricity mix resilient to the different climatic years? 

4.2.2 How much do the investment choices related to hedging against all potential weather risks cost? 

4.2.3 What other types of uncertainties might you want to hedge against? 


### 4.3 Cost sharing and cooperative games

4.3.1 What individual and collective choices have you made to achieve your objective? 

4.3.2 Was it easy to achieve it? Is it more difficult for some countries than for others? Did some countries have to make compromises in order to achieve it?


## 5. Presentation of the results obtained
Create a poster to present the work done during the week. It should include: 
    - The context (demand in your country, existing capacities from 2025…)
    - The objective pursued and the methodology to achieve it
    - The main results, including:
        - The investments made in your country 
        - The resulting energy mix in your country
    - A discussion on the main opportunities / difficulties for your country

