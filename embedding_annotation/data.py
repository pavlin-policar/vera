import pandas as pd

from embedding_annotation.variables import Variable, DiscreteVariable, ContinuousVariable, \
    DerivedVariable


def _pd_dtype_to_variable(col_name: str | Variable, col_type) -> Variable:
    """Convert a column from a pandas DataFrame to a Variable instance.

    Parameters
    ----------
    col_name: str
    col_type: dtype

    Returns
    -------
    Variable

    """
    if isinstance(col_name, Variable):
        return col_name

    if pd.api.types.is_categorical_dtype(col_type):
        variable = DiscreteVariable(
            col_name,
            categories=col_type.categories.tolist(),
            ordered=col_type.ordered,
        )
    elif pd.api.types.is_numeric_dtype(col_type):
        variable = ContinuousVariable(col_name)
    else:
        raise ValueError(
            f"Only categorical and numeric dtypes supported! Got " f"`{col_type.name}`."
        )

    return variable


def ingest(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a pandas DataFrame to a DataFrame the library can understand.

    This really just creates a copy of the data frame, but swaps out the columns
    for instances of our `Variable` objects, so we know which derived
    variables can be merged later on.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    df_new = df.copy()
    new_index = map(lambda p: _pd_dtype_to_variable(*p), zip(df.columns, df.dtypes))
    df_new.columns = pd.Index(list(new_index))
    return df_new


def ingested_to_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df_new = pd.DataFrame(index=df.index)

    for column in df.columns:
        if isinstance(column, DerivedVariable):
            df_new[column.name] = pd.Categorical(df[column])
        elif isinstance(column, DiscreteVariable):
            col = pd.Categorical(
                df[column], ordered=column.ordered, categories=column.categories
            )
            df_new[column.name] = col
        elif isinstance(column, ContinuousVariable):
            df_new[column.name] = df[column]
        else:  # probably an uningested df column
            df_new[column] = df[column]

    return df_new
