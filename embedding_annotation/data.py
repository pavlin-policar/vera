from collections.abc import Iterable

import pandas as pd
import numpy as np


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
        return str(self.name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __hash__(self):
        raise NotImplementedError()


class DiscreteVariable(Variable):
    def __init__(self, name, values: list, ordered: bool = False):
        super().__init__(name)
        self.categories = values
        self.ordered = ordered

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, values={repr(self.categories)}, ordered={self.ordered})"

    def __eq__(self, other):
        if not isinstance(other, DiscreteVariable):
            return False
        return (
            self.name == other.name
            and self.categories == other.categories
            and self.ordered == other.ordered
        )

    def __hash__(self):
        return hash(
            (self.__class__.__name__, self.name, tuple(self.categories), self.ordered)
        )


class ContinuousVariable(Variable):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __hash__(self):
        return hash((self.__class__.__name__, self.name))


class IncompatibleRuleError(ValueError):
    pass


class Rule:
    def can_merge_with(self, other: "Rule") -> bool:
        raise NotImplementedError()

    def merge_with(self, other: "Rule") -> "Rule":
        raise NotImplementedError()


class IntervalRule(Rule):
    def __init__(
        self,
        lower: float = -np.inf,
        upper: float = np.inf,
        value_name: str = "x",
    ):
        if lower is None and upper is None:
            raise ValueError("`lower` and `upper` can't both be `None`!")
        self.lower = lower
        self.upper = upper
        self.value_name = value_name

    def can_merge_with(self, other: Rule) -> bool:
        if not isinstance(other, IntervalRule):
            return False
        if np.isclose(self.lower, other.upper):  # edges match
            return True
        if np.isclose(self.upper, other.lower):  # edges match
            return True
        if self.lower <= other.upper <= self.upper:  # other.upper in interval
            return True
        if self.lower <= other.lower <= self.upper:  # other.lower in interval
            return True
        if other.lower <= self.lower <= other.upper:  # my.lower in interval
            return True
        if other.lower <= self.upper <= other.upper:  # my.upper in interval
            return True
        return False

    def merge_with(self, other: Rule) -> Rule:
        if not self.can_merge_with(other):
            raise IncompatibleRuleError(other)
        lower = min(self.lower, other.lower)
        upper = max(self.upper, other.upper)
        return self.__class__(lower=lower, upper=upper, value_name=self.value_name)

    def __str__(self):
        s = ""
        if np.isfinite(self.lower):
            s += f"{self.lower:.2f} < "
        s += str(self.value_name)
        if np.isfinite(self.upper):
            s += f" < {self.upper:.2f}"
        return s
        if self.lower is not None and self.upper is not None:
            return f"{self.lower:.2f} < {self.value_name} < {self.upper:.2f}"
        elif self.lower is not None and self.upper is None:
            return f"{self.lower:.2f} < {self.value_name}"
        elif self.lower is None and self.upper is not None:
            return f"x < {self.upper:.2f}"

    def __repr__(self):
        attrs = ["lower", "upper", "value_name"]
        attrs_str = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, IntervalRule):
            return False
        return self.lower == other.lower and self.upper == other.upper

    def __hash__(self):
        return hash((self.__class__.__name__, self.lower, self.upper))


class EqualityRule(Rule):
    def __init__(self, value, value_name: str = "x"):
        self.value = value
        self.value_name = value_name

    def can_merge_with(self, other: Rule) -> bool:
        return isinstance(other, (EqualityRule, OneOfRule))

    def merge_with(self, other: Rule) -> Rule:
        if not self.can_merge_with(other):
            raise IncompatibleRuleError(other)

        if isinstance(other, EqualityRule):
            new_values = {self.value, other.value}
            return OneOfRule(new_values, value_name=self.value_name)
        elif isinstance(other, OneOfRule):
            return other.merge_with(self)
        else:
            raise RuntimeError(f"Can't merge with type `{other.__class__.__name__}`")

    def __str__(self):
        return f"{self.value_name} == {repr(self.value)}"

    def __repr__(self):
        attrs = ["value", "value_name"]
        attrs_str = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, EqualityRule):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash((self.__class__.__name__, self.value))


class OneOfRule(Rule):
    def __init__(self, values: Iterable, value_name: str = "x"):
        self.values = set(values)
        self.value_name = value_name

    def can_merge_with(self, other: Rule) -> bool:
        return isinstance(other, (EqualityRule, OneOfRule))

    def merge_with(self, other: Rule) -> Rule:
        if not self.can_merge_with(other):
            raise IncompatibleRuleError(other)

        if isinstance(other, OneOfRule):
            new_values = self.values | other.values
        elif isinstance(other, EqualityRule):
            new_values = self.values | {other.value}
        else:
            raise RuntimeError(f"Can't merge with type `{other.__class__.__name__}`")

        return self.__class__(new_values, value_name=self.value_name)

    def __str__(self):
        return f"{self.value_name} is in {repr(self.values)}"

    def __repr__(self):
        attrs = ["values", "value_name"]
        attrs_str = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, OneOfRule):
            return False
        return frozenset(self.values) == frozenset(other.values)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(sorted(tuple(self.values)))))


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
        if self.discretization_indices is not None:
            self.discretization_indices = sorted(self.discretization_indices)

    @property
    def is_discretized(self):
        return self.discretization_indices is not None

    @property
    def name(self):
        return str(self.rule)

    def can_merge_with(self, other: "ExplanatoryVariable") -> bool:
        if not isinstance(other, ExplanatoryVariable):
            return False
        # The variables must match on their base variable
        if self.base_variable != other.base_variable:
            return False

        if self.is_discretized and other.is_discretized:
            # The merged discretization bins must form a contiguous sequence
            my_bins = set(self.discretization_indices)
            other_bins = set(other.discretization_indices)
            bins = sorted(list(my_bins | other_bins))
            if all(i == j for i, j in zip(range(bins[0], bins[-1] + 1), bins)):
                return True

        if all(
            isinstance(r, (EqualityRule, OneOfRule)) for r in [self.rule, other.rule]
        ):
            return True

        return False

    def merge_with(self, other: "ExplanatoryVariable") -> "ExplanatoryVariable":
        if not isinstance(other, ExplanatoryVariable):
            raise ValueError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise ValueError(
                f"Cannot merge explanatory variables `{self}` and {other}!"
            )

        assert self.rule.can_merge_with(other.rule)
        merged_rule = self.rule.merge_with(other.rule)

        if self.is_discretized:
            assert other.is_discretized
            discretization_indices = set(
                self.discretization_indices + other.discretization_indices
            )
            discretization_indices = sorted(list(discretization_indices))
        else:
            discretization_indices = None

        return self.__class__(
            base_variable=self.base_variable,
            rule=merged_rule,
            discretization_indices=discretization_indices,
        )

    def __repr__(self):
        attrs = ["base_variable", "rule", "discretization_indices"]
        attrs_str = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __str__(self):
        return f"{self.rule}"

    def __eq__(self, other):
        if not isinstance(other, ExplanatoryVariable):
            return False
        return (
            self.base_variable == other.base_variable
            and self.rule == other.rule
            and self.discretization_indices == other.discretization_indices
        )

    def __hash__(self):
        discretization_indices = self.discretization_indices
        if discretization_indices is not None:
            discretization_indices = tuple(discretization_indices)
        return hash(
            (
                self.__class__.__name__,
                self.base_variable,
                self.rule,
                discretization_indices,
            )
        )


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
            values=col_type.categories.tolist(),
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

    # If there are no continuous features to be discretized, return
    if len(cont_cols) == 0:
        return df_ingested

    # Use k-means for discretization
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(n_bins=5, strategy="kmeans", encode="onehot-dense")
    x_discretized = discretizer.fit_transform(df_cont.values)

    # Create derived features
    derived_features = []

    for variable, bin_edges in zip(cont_cols, discretizer.bin_edges_):
        # Ensure open intervals
        bin_edges = np.array(bin_edges)
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf

        for idx, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
            rule = IntervalRule(lower, upper, value_name=variable.name)
            v = ExplanatoryVariable(variable, rule, discretization_indices=[idx])
            derived_features.append(v)

    assert len(derived_features) == len(
        discretizer.get_feature_names_out()
    ), "The number of derived features do not match discretization output!"

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

    assert (
        len(derived_features) == x_onehot.shape[1]
    ), "The number of derived features do not match one-hot output!"

    df_disc = pd.DataFrame(x_onehot, columns=derived_features, index=df.index)
    return pd.concat([df_othr, df_disc], axis=1)


def generate_explanatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the data frame with mixed features into explanatory features."""
    return _one_hot(_discretize(ingest(df)))
