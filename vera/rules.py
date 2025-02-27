from typing import Iterable, Union

import numpy as np


class IncompatibleRuleError(ValueError):
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    @property
    def message(self):
        return f"Incompatible rules `{self.r1}` and `{self.r2}`!"


class Rule:
    def can_merge_with(self, other: "Rule") -> bool:
        raise NotImplementedError()

    def merge_with(self, other: "Rule") -> "Rule":
        raise NotImplementedError()

    def contains(self, other: "Rule") -> "Rule":
        """Check if a rule is completely encompassed by this rule."""
        raise NotImplementedError()

    def __lt__(self, other: "Rule") -> "Rule":
        raise NotImplementedError()


class IntervalRule(Rule):
    def __init__(
        self,
        lower: float = -np.inf,
        upper: float = np.inf,
        value_name: str = "x",
        precision: int = 2,
    ):
        if lower is None and upper is None:
            raise ValueError("`lower` and `upper` can't both be `None`!")
        self.lower = lower
        self.upper = upper
        self.value_name = value_name
        self.precision = precision

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
            raise IncompatibleRuleError(self, other)
        lower = min(self.lower, other.lower)
        upper = max(self.upper, other.upper)
        # Use the higher precision of the two
        new_prec = max(self.precision, other.precision)
        return IntervalRule(
            lower=lower, upper=upper, value_name=self.value_name, precision=new_prec,
        )

    def contains(self, other: Rule) -> Rule:
        if not isinstance(other, IntervalRule):
            return False
        return other.lower >= self.lower and other.upper <= self.upper

    def __str__(self):
        def _format_num(x):
            if self.precision < 0:
                return str(round(x, self.precision))
            else:
                return f"{x:.{self.precision}f}"

        # Special handling for `x > 5`. Easier to read
        if np.isfinite(self.lower) and not np.isfinite(self.upper):
            return f"{self.value_name} > {_format_num(self.lower)}"

        s = ""
        if np.isfinite(self.lower):
            s += f"{_format_num(self.lower)} < "
        s += str(self.value_name)
        if np.isfinite(self.upper):
            s += f" < {_format_num(self.upper)}"
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

    def __lt__(self, other: "IntervalRule") -> bool:
        return self.lower < other.lower


class EqualityRule(Rule):
    def __init__(self, value, value_name: str = "x"):
        self.value = value
        self.value_name = value_name

    def can_merge_with(self, other: Rule) -> bool:
        return isinstance(other, (EqualityRule, OneOfRule))

    def merge_with(self, other: Rule) -> Rule:
        if not self.can_merge_with(other):
            raise IncompatibleRuleError(self, other)

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
        return f"{self.value_name} = {str(self.value)}"

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

    def __lt__(self, other: Union["EqualityRule", "OneOfRule"]) -> bool:
        if isinstance(other, OneOfRule):
            other_vals = sorted(list(other.values))
            return self.value < other_vals[0]
        elif isinstance(other, EqualityRule):
            return self.value < other.value
        else:
            raise NotImplementedError()


class OneOfRule(Rule):
    def __init__(self, values: Iterable, value_name: str = "x"):
        self.values = set(values)
        self.value_name = value_name

    def can_merge_with(self, other: Rule) -> bool:
        return isinstance(other, (EqualityRule, OneOfRule))

    def merge_with(self, other: Rule) -> Rule:
        if not self.can_merge_with(other):
            raise IncompatibleRuleError(self, other)

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
        values = sorted(list(self.values))
        values_str = ", ".join(str(v) for v in values)
        return f"{self.value_name} in {{{values_str}}}"

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

    def __lt__(self, other: Union["EqualityRule", "OneOfRule"]) -> bool:
        my_vals = sorted(list(self.values))
        if isinstance(other, EqualityRule):
            return my_vals[0] < other.value
        elif isinstance(other, OneOfRule):
            other_vals = sorted(list(other.values))
            return my_vals[0] < other_vals[0]
        else:
            raise NotImplementedError()
