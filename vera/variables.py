import numpy as np

from vera.rules import Rule


class MergeError(Exception):
    pass


class Variable:
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
        eq_cond = all(getattr(self, f) == getattr(other, f) for f in self.eq_attrs)
        val_cond = np.allclose(self.values, other.values)
        return eq_cond and val_cond

    def __hash__(self):
        return hash(
            (self.__class__.__name__,) + tuple(getattr(self, f) for f in self.eq_attrs)
        )

    def __lt__(self, other: "Variable"):
        return self.name < other.name


class DiscreteVariable(Variable):
    repr_attrs = Variable.repr_attrs + ["categories", "ordered"]
    eq_attrs = Variable.eq_attrs + ["categories", "ordered"]

    def __init__(self, name, values, categories: list, ordered: bool = False):
        super().__init__(name, values)
        self.categories = tuple(categories)
        self.ordered = ordered


class ContinuousVariable(Variable):
    pass


class IndicatorVariable(Variable):
    repr_attrs = Variable.repr_attrs + ["base_variable", "rule"]
    eq_attrs = Variable.eq_attrs + ["base_variable", "rule"]

    def __init__(
        self,
        base_variable: Variable,
        rule: Rule,
        values: np.ndarray,
        name: str = None,
    ):
        self.base_variable = base_variable
        self.rule = rule
        self.values = values
        if name is None:
            self.name = str(rule)

    def can_merge_with(self, other: "IndicatorVariable") -> bool:
        if not isinstance(other, IndicatorVariable):
            return False
        # The variables must match on their base variable
        if self.base_variable != other.base_variable:
            return False

        return self.rule.can_merge_with(other.rule)

    def merge_with(self, other: "IndicatorVariable") -> "IndicatorVariable":
        if not isinstance(other, IndicatorVariable):
            raise MergeError(
                f"Cannot merge `{self.__class__.__name__}` with  `{other}`!"
            )

        if not self.can_merge_with(other):
            raise MergeError(f"Cannot merge derived variables `{self}` and {other}!")

        merged_rule = self.rule.merge_with(other.rule)

        return self.__class__(base_variable=self.base_variable, rule=merged_rule)

    def __str__(self):
        return self.name

    # def __eq__(self, other):
    #     if not isinstance(other, IndicatorVariable):
    #         return False
    #     return self.base_variable == other.base_variable and self.rule == other.rule

    # def __hash__(self):
    #     return hash((self.__class__.__name__, self.base_variable, self.rule))

    def __lt__(self, other):
        return (self.base_variable, self.rule) < (other.base_variable, other.rule)


class VariableGroup:
    def __init__(
        self,
        variables: list[Variable],
        values: np.ndarray,
        name: str = None,
    ):
        self.variables = variables
        self.values = values
        self.name = name

    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self.variables))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.variables == other.variables

    def __repr__(self):
        return ",".join([str(v) for v in self.variables])
