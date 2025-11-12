import warnings

import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

from utils.basic_utils import get_key_of_val


def cast_df_col_as_date(df: pd.DataFrame, date_col: str, date_format: str) -> pd.DataFrame:
    df[date_col] = df[date_col].apply(lambda x: datetime.strptime(x, date_format))
    return df


def selec_in_df_based_on_list(df: pd.DataFrame, selec_col, selec_vals: list, rm_selec_col=False) -> pd.DataFrame:
    val = df.loc[df[selec_col].isin(selec_vals)]
    if rm_selec_col:
        val = val.drop(columns=[selec_col])
    return val


def concatenate_dfs(dfs: List[pd.DataFrame], reset_index: bool = True) -> pd.DataFrame:
    df_concat = pd.concat(dfs, axis=0)
    if reset_index:
        df_concat = df_concat.reset_index(drop=True)
    return df_concat


def set_aggreg_col_based_on_corresp(df: pd.DataFrame, col_name: str, created_agg_col_name: str, val_cols: List[str],
                                    agg_corresp: Dict[str, List[str]], common_aggreg_ope: str,
                                    other_col_for_agg: str = None) -> pd.DataFrame:
    """
    Set aggreg. column based on a correspondence {aggreg. value: list of corresp. (indiv.) values}
    :param df
    :param col_name: of the (indiv.) keys
    :param created_agg_col_name: of the aggreg. keys
    :param val_cols: list of value columns
    :param agg_corresp
    :param common_aggreg_ope: name of aggreg. operation to be applied on value columns in considered df
    :param other_col_for_agg
    :returns: df after having applied aggreg. operation
    """
    df[created_agg_col_name] = df[col_name].apply(get_key_of_val, args=(agg_corresp,))
    agg_operations = {col: common_aggreg_ope for col in val_cols}
    if other_col_for_agg is not None:
        gpby_cols = [created_agg_col_name, other_col_for_agg]
    else:
        gpby_cols = created_agg_col_name
    df = df.groupby(gpby_cols).agg(agg_operations).reset_index()
    return df


def get_subdf_from_date_range(df: pd.DataFrame, date_col: str, date_min: datetime, date_max: datetime) -> pd.DataFrame:
    """
    Get values in a dataframe from a date range
    """
    df_range = df[(date_min <= df[date_col]) & (df[date_col] < date_max)]
    return df_range


def create_dict_from_cols_in_df(df: pd.DataFrame, key_col, val_col) -> dict:
    df_to_dict = df[[key_col, val_col]]
    return dict(pd.MultiIndex.from_frame(df_to_dict))


def create_dict_from_df_row(df: pd.DataFrame, col_and_val_for_selec: tuple = None, key_cols: list = None,
                            rm_col_for_selec: bool = True) -> dict:
    col_for_selec = None
    if col_and_val_for_selec is not None:
        col_for_selec = col_and_val_for_selec[0]
        val_for_selec = col_and_val_for_selec[1]
        df = df.loc[df[col_for_selec] == val_for_selec]

    # columns used as key -> all as default
    if key_cols is None:
        key_cols = list(df.columns)

    dict_from_df = {col: df[col].iloc[0] for col in key_cols}

    # remove column used for row selection?
    if col_and_val_for_selec is not None and rm_col_for_selec:
        del dict_from_df[col_for_selec]

    return dict_from_df


def rename_df_columns(df: pd.DataFrame, old_to_new_cols: dict) -> pd.DataFrame:
    # catch SettingWithCopyWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df.rename(columns=old_to_new_cols, inplace=True)
    return df


def set_key_columns(col_names: list, tuple_values: List[tuple], n_repeat: int = None) -> pd.DataFrame:
    """
    :param col_names: list of key column names
    :param tuple_values
    :param n_repeat: number of repetition of each tuple in the df e.g., when dates are commonly used per
    each tuple value
    """
    if n_repeat is None:
        n_repeat = 1
    n_keys = len(tuple_values[0])
    concat_keys = np.concatenate([np.array(elt).reshape(1, n_keys) for elt in tuple_values], axis=0)
    concat_keys = np.repeat(concat_keys, n_repeat, axis=0)
    return pd.DataFrame(data=concat_keys, columns=col_names)
