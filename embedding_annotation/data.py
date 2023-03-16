import operator
from collections.abc import Iterable
from functools import reduce

import pandas as pd
import numpy as np

from embedding_annotation.region import Region, CompositeRegion


class Variable:
    __slots__ = ["name"]

    def __init__(self, name: str):
        self.name = name

    @property
    def is_discrete(self):
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self):
        return isinstance(self, ContinuousVariable)

    @property
    def is_derived(self):
        return isinstance(self, DerivedVariable)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        attrs_str = ", ".join(
            f"{attr}={repr(getattr(self, attr))}" for attr in self.__slots__
        )
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.__class__.__name__, self.name))


class DiscreteVariable(Variable):
    __slots__ = Variable.__slots__ + ["categories", "ordered"]

    def __init__(self, name, categories: list, ordered: bool = False):
        super().__init__(name)
        self.categories = tuple(categories)
        self.ordered = ordered

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
            (self.__class__.__name__, self.name, self.categories, self.ordered)
        )


class ContinuousVariable(Variable):
    pass


class IncompatibleRuleError(ValueError):
    pass


class Rule:
    def can_merge_with(self, other: "Rule") -> bool:
        raise NotImplementedError()

    def merge_with(self, other: "Rule") -> "Rule":
        raise NotImplementedError()

    def contains(self, other: "Rule") -> "Rule":
        """Check if a rule is completely encompassed by this rule."""
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

    def contains(self, other: Rule) -> Rule:
        if not isinstance(other, IntervalRule):
            return False
        return other.lower >= self.lower and other.upper <= self.upper

    def __str__(self):
        # Special handling for `x > 5`. Easier to read
        if np.isfinite(self.lower) and not np.isfinite(self.upper):
            return f"{self.value_name} > {self.lower:.2f}"

        s = ""
        if np.isfinite(self.lower):
            s += f"{self.lower:.2f} < "
        s += str(self.value_name)
        if np.isfinite(self.upper):
            s += f" < {self.upper:.2f}"
        return s

    def __repr__(self):
        attrs = ["lower", "upper"]
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

    def contains(self, other: Rule) -> Rule:
        if not isinstance(other, EqualityRule):
            return False
        return self.value == other.value

    def __str__(self):
        return f"{self.value_name} = {repr(self.value)}"

    def __repr__(self):
        attrs = ["value"]
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

    def contains(self, other: Rule) -> Rule:
        if not isinstance(other, (OneOfRule, EqualityRule)):
            return False
        if isinstance(other, EqualityRule):
            values = {other.value}
        else:
            values = other.values
        return all(v in self.values for v in values)

    def __str__(self):
        return f"{self.value_name} is in {repr(self.values)}"

    def __repr__(self):
        attrs = ["values"]
        attrs_str = ", ".join(f"{attr}={repr(getattr(self, attr))}" for attr in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, OneOfRule):
            return False
        return frozenset(self.values) == frozenset(other.values)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(sorted(tuple(self.values)))))


class DerivedVariable(Variable):
    __slots__ = Variable.__slots__ + ["base_variable", "rule"]

    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
    ):
        self.name = str(rule)
        self.base_variable = base_variable
        self.rule = rule

    def can_merge_with(self, other: "DerivedVariable") -> bool:
        if not isinstance(other, DerivedVariable):
            return False
        # The variables must match on their base variable
        if self.base_variable != other.base_variable:
            return False

        return self.rule.can_merge_with(other.rule)

    def merge_with(self, other: "DerivedVariable") -> "DerivedVariable":
        if not isinstance(other, DerivedVariable):
            raise ValueError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise ValueError(
                f"Cannot merge derived variables `{self}` and {other}!"
            )

        merged_rule = self.rule.merge_with(other.rule)

        return self.__class__(base_variable=self.base_variable, rule=merged_rule)

    def __str__(self):
        return str(self.rule)

    def __eq__(self, other):
        if not isinstance(other, DerivedVariable):
            return False
        return (
            self.base_variable == other.base_variable and self.rule == other.rule
        )

    def __hash__(self):
        return hash((self.__class__.__name__, self.base_variable, self.rule))


class ExplanatoryVariable(DerivedVariable):
    __slots__ = DerivedVariable.__slots__ + ["values", "region", "embedding", "scale"]

    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
        values: np.ndarray,
        region: Region,
        embedding: np.ndarray,
        scale: float,
    ):
        super().__init__(base_variable, rule)

        self.values = values
        self.region = region
        self.embedding = embedding
        self.scale = scale

        if self.values.shape[0] != self.embedding.shape[0]:
            raise ValueError(
                f"The number of samples in the feature values "
                f"({self.values.shape[0]}) does not match the number "
                f"of samples in the embedding ({self.embedding.shape[0]})."
            )

    def can_merge_with(self, other: "ExplanatoryVariable") -> bool:
        if not isinstance(other, ExplanatoryVariable):
            return False
        # The variables must match on their base variable
        if self.base_variable != other.base_variable:
            return False

        if not self.rule.can_merge_with(other.rule):
            return False

        if not np.allclose(self.embedding, other.embedding):
            return False

        return True

    def merge_with(self, other: "ExplanatoryVariable") -> "ExplanatoryVariable":
        if not isinstance(other, ExplanatoryVariable):
            raise ValueError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise ValueError(
                f"Cannot merge explanatory variables `{self}` and {other}!"
            )

        merged_rule = self.rule.merge_with(other.rule)

        return CompositeExplanatoryVariable([self, other], merged_rule)

    def __repr__(self):
        attrs = [
            ("base_variable", repr(self.base_variable)),
            ("rule", repr(self.rule)),
            ("values", "[...]"),
            ("region", repr(self.region)),
            ("embedding", "[[...]]"),
            ("scale", repr(self.scale)),
        ]
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    @property
    def contained_variables(self):
        return [self]


class CompositeExplanatoryVariable(ExplanatoryVariable):
    def __init__(
        self,
        variables: list[ExplanatoryVariable],
        rule: Rule,
    ):
        self.base_variables = variables

        v0 = variables[0]
        base_variable = v0.base_variable
        if not all(v.base_variable == base_variable for v in variables[1:]):
            raise RuntimeError(
                "Cannot merge Explanatory variables which do not share the "
                "same base variable!"
            )

        feature_values = np.vstack([v.values for v in variables])
        merged_values = np.max(feature_values, axis=0)
        merged_region = CompositeRegion([v.region for v in variables])

        embedding = v0.embedding
        if not all(np.allclose(v.embedding, embedding) for v in variables[1:]):
            raise RuntimeError(
                "Cannot merge Explanatory variables which do not share the "
                "same embedding!"
            )

        scale = v0.scale

        super().__init__(
            base_variable, rule, merged_values, merged_region, embedding, scale
        )

    @property
    def contained_variables(self):
        return reduce(operator.add, [v.contained_variables for v in self.base_variables])


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
