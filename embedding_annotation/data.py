import numpy as np
import pandas as pd


class Variable:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    @property
    def is_discrete(self):
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self):
        return isinstance(self, ContinuousVariable)

    def __str__(self):
        return f"{self.name} ({'d' if self.is_discrete else 'c'})"


class DiscreteVariable(Variable):
    def __init__(self, name, values: list, ordered: bool = False):
        super().__init__(name)
        self.categories = values
        self.ordered = ordered

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, values={repr(self.categories)} ordered={self.ordered})"


class ContinuousVariable(Variable):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"


class Rule:
    pass


class IntervalRule(Rule):
    def __init__(
        self,
        lower: float = None,
        upper: float = None,
        value_name: str = "x",
    ):
        if lower is None and upper is None:
            raise ValueError("`lower` and `upper` can't both be `None`!")
        self.lower = lower
        self.upper = upper
        self.value_name = value_name

    def __str__(self):
        if self.lower is not None and self.upper is not None:
            return f"{self.lower:.2f} < {self.value_name} < {self.upper:.2f}"
        elif self.lower is not None and self.upper is None:
            return f"{self.lower:.2f} < {self.value_name}"
        elif self.lower is None and self.upper is not None:
            return f"x < {self.upper:.2f}"


class EqualityRule(Rule):
    def __init__(self, value, value_name: str = "x"):
        self.value = value
        self.value_name = value_name

    def __str__(self):
        return f"{self.value_name} == {repr(self.value)}"


class ExplanatoryVariable(DiscreteVariable):
    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
        discretization_indices: list[int] = None,
    ):
        self.base_variable = base_variable
        self.rule = rule
        self.discretization_indices = discretization_indices

    @property
    def name(self):
        return str(self.rule)

    def __str__(self):
        return f"{self.rule} (x)"

    def __repr__(self):
        attrs = ["base_variable", "rule", "discretization_indices"]
        attrs_str = ", ".join(
            f"{attr}={repr(getattr(self, attr))}" for attr in attrs
        )
        return f"{self.__class__.__name__}({attrs_str})"


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

    if pd.api.types.is_categorical(col_type):
        variable = DiscreteVariable(
            col_name,
            values=col_type.categories.tolist(),
            ordered=col_type.ordered,
        )
    elif pd.api.types.is_numeric_dtype(col_type):
        variable = ContinuousVariable(col_name)
    else:
        raise ValueError(
            f"Only categorical and numeric dtypes supported! Got "
            f"`{col_type.name}`."
        )

    return variable


def ingest(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a pandas DataFrame to a DataFrame the library can understand.

    This really just creates a copy of the data frame, but swaps out the columns
    for instances of our `Variable` objects, so we know which explanatory
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
        if isinstance(column, ExplanatoryVariable):
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


def _discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize all continuous variables in the data frame."""
    df_ingested = ingest(df)

    # We can only discretize continuous columns
    cont_cols = [c for c in df_ingested.columns if c.is_continuous]
    disc_cols = [c for c in df_ingested.columns if c.is_discrete]
    df_cont = df_ingested[cont_cols]
    df_disc = df_ingested[disc_cols]

    # Use k-means for discretization
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(
        n_bins=5, strategy="kmeans", encode="onehot-dense"
    )
    x_discretized = discretizer.fit_transform(df_cont.values)

    # Create derived features
    derived_features = []
    for variable, bin_edges in zip(cont_cols, discretizer.bin_edges_):
        for idx, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
            rule = IntervalRule(lower, upper, value_name=variable.name)
            v = ExplanatoryVariable(variable, rule, discretization_indices=[idx])
            derived_features.append(v)

    assert len(derived_features) == len(discretizer.get_feature_names_out()), \
        "The number of derived features do not match discretization output!"

    df_cont = pd.DataFrame(x_discretized, columns=derived_features, index=df.index)
    return pd.concat([df_disc, df_cont], axis=1)


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encodings for all discrete variables in the data frame."""
    df_ingested = ingest(df)

    # We can only discretize continuous columns
    disc_cols = [c for c in df_ingested.columns if type(c) is DiscreteVariable]
    if len(disc_cols) == 0:
        return df_ingested

    othr_cols = [c for c in df_ingested.columns if type(c) is not DiscreteVariable]
    df_disc = df_ingested[disc_cols]
    df_othr = df_ingested[othr_cols]

    x_onehot = pd.get_dummies(df_disc).values

    # Create derived features
    derived_features = []
    for variable in df_disc.columns:
        for category in variable.categories:
            rule = EqualityRule(category, value_name=variable.name)
            v = ExplanatoryVariable(variable, rule)
            derived_features.append(v)

    assert len(derived_features) == x_onehot.shape[1], \
        "The number of derived features do not match one-hot output!"

    df_disc = pd.DataFrame(x_onehot, columns=derived_features, index=df.index)
    return pd.concat([df_othr, df_disc], axis=1)


def generate_explanatory_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the data frame with mixed features into explanatory features."""
    return _one_hot(_discretize(ingest(df)))
