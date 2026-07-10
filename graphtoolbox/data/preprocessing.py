import numpy as np
import pandas as pd
from typing import Any, Optional

def extract_dataframe(df: pd.DataFrame, day_inf: Optional[str] = None, day_sup: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts a subset of a DataFrame based on a date range.

    This function filters the rows of the input DataFrame `df` to include only those
    within the specified date range `[day_inf, day_sup)`. The date column in the DataFrame
    must be named 'date' and should be in the 'YYYY-MM-DD' format.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'date' column.
        day_inf (str, optional): The start date of the range (inclusive) in 'YYYY-MM-DD' format.
            If None, it defaults to the earliest date in the DataFrame.
        day_sup (str, optional): The end date of the range (exclusive) in 'YYYY-MM-DD' format.
            If None, it defaults to the latest date in the DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows within the specified date range.
    """
    assert 'date' in df.columns, "Column 'date' is not in the DataFrame!"
    new_df = df.copy()
    if day_inf is None:
        day_inf = new_df.date.to_numpy()[0]
    if day_sup is None:
        delta_t = new_df.date.to_numpy()[1] - new_df.date.to_numpy()[0]
        day_sup = new_df.date.to_numpy()[-1] + delta_t
    mask = ((new_df.date >= day_inf) & (new_df.date < day_sup))
    new_df = new_df[mask].reset_index(drop=True)
    return new_df

def create_variable(df: pd.DataFrame, var_name: str, val: np.ndarray) -> pd.DataFrame:
    """
    Adds a new column to a DataFrame with specified values.

    This function creates a new column in the input DataFrame `df` with the name `var_name`
    and populates it with the values provided in the `val` array. The length of the `val`
    array must match the number of rows in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to which the new column will be added.
        var_name (str): The name of the new column to be created.
        val (np.ndarray): A NumPy array containing the values to be added to the new column.
            The length of this array must match the number of rows in the DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the additional column `var_name` containing the values from `val`.
    """
    assert len(df) == len(val), 'DataFrame and variables lengths do not match!'
    new_df = df.copy()
    new_df[var_name] = val
    return new_df

def sub_df(df: pd.DataFrame, var_name: str, val: Any) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows where a specified column matches a given value.

    This function creates a new DataFrame containing only the rows from the input DataFrame `df`
    where the values in the column `var_name` are equal to `val`.

    Args:
        df (pd.DataFrame): The input DataFrame to be filtered.
        var_name (str): The name of the column to be filtered on.
        val (Any): The value that the column `var_name` should match to be included in the output DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where `df[var_name]` equals `val`.
    """
    assert var_name in df.columns, f"Column '{var_name}' is not in the DataFrame!"
    new_df = df.copy()
    new_df = new_df[new_df[var_name] == val].reset_index(drop=True)
    return new_df

def extract_dummies(df: pd.DataFrame, var_names: np.ndarray) -> pd.DataFrame:
    """
    Converts specified categorical variables in a DataFrame to dummy/indicator variables.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        var_names (np.ndarray): An array of column names to be converted to dummy variables.

    Returns:
        pd.DataFrame: A new DataFrame with the original columns and the added dummy variables.
    """
    assert all(var_name in df.columns for var_name in var_names), "One or more column names in var_names are not in the DataFrame!"
    new_df = df.copy()
    list_to_concat = [new_df] + [pd.get_dummies(new_df[var_name], prefix=var_name, drop_first=True, dtype=float) for var_name in var_names]
    new_df = pd.concat(list_to_concat, axis=1)
    new_df = new_df.drop(columns=var_names, axis=1)
    return new_df
