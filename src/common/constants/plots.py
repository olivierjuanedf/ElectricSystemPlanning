from dataclasses import dataclass


@dataclass
class PlotNames:
    prod_stacked: str = 'prod_stacked'
    prod_stacked_with_stock: str = 'prod_stacked_with_stock'
    link_flows: str = 'link_flows'
    link_capa_saturation: str = 'link_capa_saturation'
    marginal_price: str = 'marginal_price'
    stock_soc: str = 'stock_soc'


# list of plot names that will be done country per country
PER_COUNTRY_PLOT = [PlotNames.prod_stacked, PlotNames.prod_stacked_with_stock, PlotNames.link_flows,
                    PlotNames.stock_soc]
