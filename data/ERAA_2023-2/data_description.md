**About ERAA - raw - data**
* Made available **by ENTSO-E**, an entity coordinating all European Transmission Network Operators 
(among them, RTE in France)
* **Source** https://www.entsoe.eu/outlooks/eraa/
* **Edition 2023-2** (last ed. currently available is 2025)

**Dates**
* All data share same year (1900) given that a "fictive" calendar is considered in this data
* 365 days for all considered years (even for 2028... with a February 29th!)

**Units**
* **Powers** in MW
* **Capacity factors** in % 
N.B. For Renewable Energy Source, to represent how much these units can produce relatively 
to their installed capacity - and depending on the (climate and) weather conditions

**"Geographical" aggregation**
* Raw data are generally provided **at the scale of *market nodes*** - typically with a few ones per country
* **Here aggregated** at the scale of **(meta-)countries**, with following operation per datatype:
- demand -> sum
- capacity factors -> mean over all market nodes in a country (to average climatic/weather effects in different
regions - nodes - of this country)

**Others**
* Regarding **hydro inflows data for Scandinavia**, the data have been **taken from ERAA2024 edition** as it was missing
in the 2023.2 one. Given that there is no possible matching between the (historical) climatic years (CY) of 
ERAA2023.2 and the (climatic-simulation based) weather scenarios (WS) of ERAA2024, the average over WS has been
calculated and assigned to each CY of the data provided here 