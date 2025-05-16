import abc
from collections import defaultdict

import numpy as np

from vera.rules import Rule


class MergeError(Exception):
    pass


class RegionDescriptor(metaclass=abc.ABCMeta):
    """Abstract interface that provides a textual description of regions."""
    def __init__(self, values: np.ndarray):
        self.values = values

    @abc.abstractmethod
    def merge_with(self, other: "RegionDescriptor") -> "RegionDescriptor":
        pass

    @property
    @abc.abstractmethod
    def contained_variables(self) -> tuple["Variable"]:
        pass

    @staticmethod
    def merge(descriptors: list["RegionDescriptor"]) -> "RegionDescriptor":
        if any(not isinstance(d, RegionDescriptor) for d in descriptors):
            descriptor_str = ", ".join(
                [f"{d} ({d.__class__.__name__})" for d in descriptors]
            )
            raise TypeError(
                f"Can only merge `RegionDescriptor` instances!\nGot [{descriptor_str}]"
            )

        all_indicators = []
        for descriptor in descriptors:
            if isinstance(descriptor, IndicatorVariable):
                all_indicators.append(descriptor)
            elif isinstance(descriptor, IndicatorVariableGroup):
                all_indicators.extend(descriptor.variables)
            else:
                raise RuntimeError("This should never be reached.")

        merged_indicators = merge_indicator_variables(all_indicators)

        # If the merging resulted in a single indicator, just return that
        if len(merged_indicators) == 1:
            return merged_indicators[0]

        return IndicatorVariableGroup(merged_indicators)

    def can_merge_with(self, other: "RegionDescriptor"):
        try:
            self.merge_with(other)
            return True
        except MergeError:
            return False


class Variable(metaclass=abc.ABCMeta):
    repr_attrs = ["name"]
    eq_attrs = ["name"]

    def __init__(self, name: str, values: np.ndarray, base_variable: "Variable" = None):
        self.name = name
        self.values = values
        self.base_variable = base_variable

    @property
    def is_discrete(self) -> bool:
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self) -> bool:
        return isinstance(self, ContinuousVariable)

    @property
    def is_indicator(self) -> bool:
        return isinstance(self, IndicatorVariable)

    @property
    def is_derived(self):
        return self.base_variable is not None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        eq_cond = all(getattr(self, f) == getattr(other, f) for f in self.eq_attrs)
        val_cond = np.allclose(self.values, other.values, equal_nan=True)
        return eq_cond and val_cond

    def __hash__(self):
        return hash(
            (self.__class__.__name__,) + tuple(getattr(self, f) for f in self.eq_attrs)
        )

    def __lt__(self, other: "Variable"):
        return self.name < other.name

    def __repr__(self):
        attrs_str = ", ".join(
            f"{attr}={repr(getattr(self, attr))}" for attr in self.repr_attrs
        )
        return f"{self.__class__.__name__}({attrs_str})"


class DiscreteVariable(Variable):
    repr_attrs = Variable.repr_attrs + ["categories", "ordered"]
    eq_attrs = Variable.eq_attrs + ["categories", "ordered"]

    def __init__(
        self,
        name: str,
        values: np.ndarray,
        categories: list[str],
        ordered: bool = False,
    ):
        super().__init__(name, values)
        self.categories = tuple(categories)
        self.ordered = ordered


class ContinuousVariable(Variable):
    pass


class IndicatorVariable(Variable, RegionDescriptor):
    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
        values: np.ndarray,
    ):
        super().__init__(name=None, values=values, base_variable=base_variable)
        self.rule = rule

    def merge_with(self, other: RegionDescriptor) -> "IndicatorVariable":
        if not isinstance(other, RegionDescriptor):
            raise TypeError(
                f"Cannot merge `{self}` with `{other}`. Only instances of "
                f"`RegionDescriptor` can be merged!"
            )

        if isinstance(other, IndicatorVariable):
            if other.base_variable == self.base_variable:
                if self.rule.can_merge_with(other.rule):
                    # Merge compatible indicators into a more general indicator
                    new_rule = self.rule.merge_with(other.rule)
                    # The values indicate if the sample belongs to ANY group
                    new_values = np.max(np.vstack([self.values, other.values]), axis=0)
                    return IndicatorVariable(self.base_variable, new_rule, new_values)
                else:
                    # If the rules are not compatible, the descriptors can't be merged
                    raise MergeError(f"Cannot merge `{self}` with `{other}`!")
            else:
                return IndicatorVariableGroup([self, other])

        # If we're merging with a group, let the group handle the merge
        elif isinstance(other, IndicatorVariableGroup):
            return other.merge_with(self)

        else:
            raise RuntimeError("This should never be reached.")

    @property
    def contained_variables(self) -> tuple[Variable]:
        return (self.base_variable,)

    def __hash__(self):
        return hash((self.__class__.__name__, self.base_variable, self.rule))

    def __eq__(self, other):
        if not isinstance(other, IndicatorVariable):
            return False
        return self.base_variable == other.base_variable and self.rule == other.rule

    def __lt__(self, other):
        return (self.base_variable, self.rule) < (other.base_variable, other.rule)

    def __str__(self):
        return str(self.rule)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.rule)})"


def merge_indicator_variables(variables: list[IndicatorVariable]) -> list[IndicatorVariable]:
    """Inspect the list of indicator variables, and merge whatever possible."""
    grouped = defaultdict(set)
    for v in variables:
        grouped[v.base_variable].add(v)

    merged_variables = []
    for base_var, indicator_vars in grouped.items():
        indicator_vars = list(indicator_vars)
        # If the base var has only a single indicator, just append that
        if len(indicator_vars) == 1:
            merged_variables.append(indicator_vars[0])

        # Otherwise, we will try to merge the indicators
        else:
            # Sort the variables so their rules should be compatible
            indicator_vars = sorted(indicator_vars)

            new_var = indicator_vars[0]
            for other_var in indicator_vars[1:]:
                try:
                    new_var = new_var.merge_with(other_var)
                except MergeError:
                    # If the merging failed, append the current merged
                    # variable, which contains all the variables up to this
                    # point, to the result set, and use the current variable
                    # as a new basis point
                    merged_variables.append(new_var)
                    new_var = other_var
            merged_variables.append(new_var)

    return merged_variables


class IndicatorVariableGroup(RegionDescriptor):
    """A group of potentially unrelated indicator variables.

    The values of an indicator variable group indicate which samples belong to
    ALL the contained indicator variables.

    """
    def __init__(self, variables: list[IndicatorVariable]):
        self.variables = sorted(merge_indicator_variables(variables))
        # The merged values indicate which samples belong to ALL contained variables
        merged_values = np.min(np.vstack([v.values for v in self.variables]), axis=0)

        super().__init__(values=merged_values)

    def merge_with(self, other: RegionDescriptor) -> "IndicatorVariableGroup":
        if not isinstance(other, RegionDescriptor):
            raise TypeError(
                f"Cannot merge `{self}` with `{other}`. Only instances of "
                f"`RegionDescriptor` can be merged!"
            )

        if isinstance(other, IndicatorVariable):
            other_variables = [other]
        elif isinstance(other, IndicatorVariableGroup):
            other_variables = other.variables
        else:
            raise RuntimeError("This should never be reached.")

        return IndicatorVariableGroup(self.variables + other_variables)

    @property
    def contained_variables(self) -> tuple[Variable]:
        return tuple(v.base_variable for v in self.variables)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, frozenset(self.variables)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return frozenset(self.variables) == frozenset(other.variables)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.variables)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join([str(v) for v in self.variables])}])"
