import operator
from functools import cached_property, reduce

import numpy as np

from veca import metrics
from veca.region import Region, CompositeRegion
from veca.rules import Rule


class MergeError(Exception):
    pass


class Variable:
    repr_attrs = ["name"]

    def __init__(self, name: str, values: np.ndarray):
        self.name = name
        self.values = values
        self._explanatory_variables = []

    @property
    def is_discrete(self) -> bool:
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self) -> bool:
        return isinstance(self, ContinuousVariable)

    @property
    def is_derived(self) -> bool:
        return isinstance(self, DerivedVariable)

    @property
    def explanatory_variables(self) -> list["ExplanatoryVariable"]:
        return sorted(self._explanatory_variables)

    def register_explanatory_variable(self, variable: "ExplanatoryVariable") -> None:
        self._explanatory_variables.append(variable)

    def unregister_explanatory_variable(self, variable: "ExplanatoryVariable") -> None:
        self._explanatory_variables.remove(variable)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        attrs_str = ", ".join(
            f"{attr}={repr(getattr(self, attr))}" for attr in self.repr_attrs
        )
        return f"{self.__class__.__name__}({attrs_str})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.__class__.__name__, self.name))


class DiscreteVariable(Variable):
    repr_attrs = Variable.repr_attrs + ["categories", "ordered"]

    def __init__(self, name, values, categories: list, ordered: bool = False):
        super().__init__(name, values)
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
        return hash((self.__class__.__name__, self.name, self.categories, self.ordered))


class ContinuousVariable(Variable):
    pass


class DerivedVariable(Variable):
    repr_attrs = Variable.repr_attrs + ["base_variable", "rule"]

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
            raise MergeError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise MergeError(f"Cannot merge derived variables `{self}` and {other}!")

        merged_rule = self.rule.merge_with(other.rule)

        return self.__class__(base_variable=self.base_variable, rule=merged_rule)

    def __str__(self):
        return str(self.rule)

    def __eq__(self, other):
        if not isinstance(other, DerivedVariable):
            return False
        return self.base_variable == other.base_variable and self.rule == other.rule

    def __hash__(self):
        return hash((self.__class__.__name__, self.base_variable, self.rule))

    def __lt__(self, other):
        return (self.base_variable, self.rule) < (other.base_variable, other.rule)


class VariableGroup:
    def __init__(
        self,
        variables: list[Variable],
        explanatory_variables: list["ExplanatoryVariable"],
        name: str = None,
    ):
        self.variables = frozenset(variables)
        self.name = name
        self.explanatory_variables = explanatory_variables

    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self.variables))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.variables == other.variables

    def __repr__(self):
        return ",".join([str(v) for v in self.variables])


class EmbeddingRegionMixin:
    """This mixin contains all the functionality to do with anything computed on
    the embedding and the values in the embedding."""

    def __init__(
        self,
        values: np.ndarray,
        region: Region,
        embedding: "Embedding",
    ):
        self.values = values
        self.region = region
        self.embedding = embedding

        if self.values.shape[0] != self.embedding.shape[0]:
            raise ValueError(
                f"The number of samples in the feature values "
                f"({self.values.shape[0]}) does not match the number "
                f"of samples in the embedding ({self.embedding.shape[0]})."
            )

    @cached_property
    def num_all_samples(self):
        return int(self.values.sum())

    @cached_property
    def contained_samples(self):
        return self.region.get_contained_samples(self.embedding)

    @cached_property
    def num_contained_samples(self) -> int:
        return len(self.contained_samples)

    @cached_property
    def pct_contained_samples(self) -> float:
        return self.num_contained_samples / self.embedding.shape[0]

    @cached_property
    def purity(self):
        # We only have binary features, so we can just compute the mean
        return np.mean(self.values[list(self.contained_samples)])

    @cached_property
    def morans_i(self):
        return metrics.morans_i(self.values, self.embedding.adj)

    @cached_property
    def gearys_c(self):
        return metrics.gearys_c(self.values, self.embedding.adj)


class ExplanatoryVariable(DerivedVariable, EmbeddingRegionMixin):
    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
        values: np.ndarray,
        region: Region,
        embedding: np.ndarray,
    ):
        super().__init__(base_variable, rule)
        EmbeddingRegionMixin.__init__(self, values, region, embedding)

    def can_merge_with(self, other: "ExplanatoryVariable") -> bool:
        if not isinstance(other, ExplanatoryVariable):
            return False
        # The variables must match on their base variable
        if self.base_variable != other.base_variable:
            return False

        if not self.rule.can_merge_with(other.rule):
            return False

        if not np.allclose(self.embedding.X, other.embedding.X):
            return False

        return True

    def merge_with(self, other: "ExplanatoryVariable") -> "ExplanatoryVariable":
        if not isinstance(other, ExplanatoryVariable):
            raise MergeError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise MergeError(
                f"Cannot merge explanatory variables `{self}` and {other}!"
            )

        return CompositeExplanatoryVariable([self, other])

    def __repr__(self):
        attrs = [
            ("rule", repr(self.rule)),
            # ("values", "[...]"),
            # ("region", repr(self.region)),
            # ("embedding", "[[...]]"),
        ]
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def split_region(self):
        """If a variable comprises multiple regions, split each region into its
        own explanatory variable."""
        return [
            ExplanatoryVariable(
                self.base_variable, self.rule, self.values, r, self.embedding
            ) for r in self.region.split_into_parts()
        ]

    @property
    def contained_variables(self):
        return [self]

    @property
    def plot_detail(self) -> str:
        """Region details to be shown in a plot."""
        return None

    def __hash__(self):
        return hash(
            (self.__class__.__name__, self.base_variable, self.rule, self.region)
        )

    def __eq__(self, other):
        if not isinstance(other, ExplanatoryVariable):
            return False
        return (
            self.base_variable == other.base_variable and
            self.rule == other.rule and
            self.region == other.region
        )


class CompositeExplanatoryVariable(ExplanatoryVariable):
    def __init__(self, variables: list[ExplanatoryVariable]):
        self.base_variables = variables

        v0 = variables[0]
        base_variable = v0.base_variable
        if not all(v.base_variable == base_variable for v in variables[1:]):
            raise RuntimeError(
                "Cannot merge Explanatory variables which do not share the "
                "same base variable!"
            )

        try:
            new_rule = variables[0].rule
            for v in variables[1:]:
                new_rule = new_rule.merge_with(v.rule)
        except MergeError as e:
            raise MergeError(
                f"CompositeExplanatoryVariable cannot be created because the "
                f"variables given were incompatible:\n{e.message}\nPlease "
                f"ensure that the variables are given in a valid merge order."
            )

        feature_values = np.vstack([v.values for v in variables])
        merged_values = np.max(feature_values, axis=0)
        merged_region = CompositeRegion([v.region for v in variables])

        embedding = v0.embedding
        if not all(np.allclose(v.embedding.X, embedding.X) for v in variables[1:]):
            raise RuntimeError(
                "Cannot merge Explanatory variables which do not share the "
                "same embedding!"
            )

        super().__init__(
            base_variable, new_rule, merged_values, merged_region, embedding
        )

    @property
    def contained_variables(self):
        return sorted(
            reduce(operator.add, [v.contained_variables for v in self.base_variables])
        )

    @property
    def plot_detail(self) -> str:
        """Region details to be shown in a plot."""
        return None

    @cached_property
    def contained_samples(self):
        """Checking the region for contained samples is slow."""
        return reduce(operator.or_, (v.contained_samples for v in self.base_variables))


class ExplanatoryVariableGroup(EmbeddingRegionMixin):
    def __init__(self, variables: list[ExplanatoryVariable], name: str = None):
        self.variables = variables
        self.name = name

        v0 = variables[0]

        feature_values = np.vstack([v.values for v in variables])
        # Take min: if plotted together, we expect each point to fulfill all the
        # rules in the feature group
        merged_values = np.max(feature_values, axis=0)
        merged_region = CompositeRegion([v.region for v in variables])

        embedding = v0.embedding
        if not all(np.allclose(v.embedding.X, embedding.X) for v in variables[1:]):
            raise MergeError(
                "Cannot merge explanatory variables which do not share the "
                "same embedding!"
            )

        EmbeddingRegionMixin.__init__(self, merged_values, merged_region, embedding)

    @property
    def contained_variables(self):
        return self.variables

    def __repr__(self):
        attrs = [
            ("name", repr(self.name)),
            ("variables", repr(self.variables)),
            ("values", "[...]"),
            ("region", repr(self.region)),
            ("embedding", "[[...]]"),
        ]
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    @property
    def plot_detail(self) -> str:
        return str(self.name)

    @cached_property
    def contained_samples(self):
        """Checking the region for contained samples is slow."""
        return reduce(operator.or_, (v.contained_samples for v in self.variables))
