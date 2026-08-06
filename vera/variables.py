import abc
import copy
from collections import defaultdict
from typing import Callable

import numpy as np

from vera.rules import Rule


class MergeError(Exception):
    pass


def conjunction(values: list[np.ndarray]) -> np.ndarray:
    """Which samples every one of the indicators flags.

    A sample is excluded as soon as one indicator does not flag it, whether or
    not the others measured it. Where every indicator that did measure the
    sample flags it, but at least one did not measure it, membership is
    unknown: the sample could belong either way, and NaN says so.
    """
    stacked = np.vstack(values)
    excluded = np.any(stacked == 0, axis=0)
    unknown = np.any(np.isnan(stacked), axis=0)
    return np.where(excluded, 0.0, np.where(unknown, np.nan, 1.0))


def disjunction(values: list[np.ndarray]) -> np.ndarray:
    """Which samples at least one of the indicators flags.

    The mirror of :func:`conjunction`: one flag is enough to include a sample,
    and membership is unknown only where nothing flags it and something failed
    to measure it.
    """
    stacked = np.vstack(values)
    included = np.any(stacked == 1, axis=0)
    unknown = np.any(np.isnan(stacked), axis=0)
    return np.where(included, 1.0, np.where(unknown, np.nan, 0.0))


def _validate_format_label_args(max_descriptors, truncation_template, sep, truncation_sep):
    """Validation is shared by every descriptor type, so a call fails the same
    way whether or not the arguments end up being used."""
    if max_descriptors is not None and max_descriptors < 1:
        raise ValueError(
            f"`max_descriptors` must be a positive integer or None, got "
            f"{max_descriptors}."
        )
    if not isinstance(sep, str):
        raise ValueError(f"`sep` must be a string, got {sep!r}.")
    if truncation_sep is not None and not isinstance(truncation_sep, str):
        raise ValueError(
            f"`truncation_sep` must be a string or None, got {truncation_sep!r}."
        )
    try:
        marker = truncation_template.format(n=1)
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"`truncation_template` may only reference the {{n}} placeholder, "
            f"got {truncation_template!r}."
        ) from e
    if not marker:
        raise ValueError(
            "`truncation_template` must produce a non-empty marker; a "
            "truncated label has to indicate its hidden variables."
        )


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

    @property
    def partial_values(self) -> np.ndarray:
        """For each sample, the fraction of the descriptor it satisfies.

        A sample satisfying the descriptor fully scores 1 and one satisfying
        none of it scores 0. `values` is the thresholded form: 1 exactly where
        this is 1.
        """
        return np.asarray(self.values, dtype=float)

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

    def format_label(
        self,
        max_descriptors: int = None,
        truncation_template: str = "(+{n} more)",
        sep: str = "\n",
        truncation_sep: str = None,
    ) -> str:
        """The descriptor's display text.

        Equal to ``str(self)``; descriptors composed of multiple variables
        join them with `sep` and truncate to the first `max_descriptors` of
        them. A single-variable descriptor has nothing to join, but the
        arguments are validated all the same.
        """
        _validate_format_label_args(
            max_descriptors, truncation_template, sep, truncation_sep
        )
        return str(self)

    def ranked_by(
        self, compute_scores: Callable[[], dict]
    ) -> "RegionDescriptor":
        """This descriptor with its constituent variables ordered by
        descending score.

        `compute_scores` is a zero-argument callable returning a score per
        variable. A descriptor without constituent parts has nothing to
        reorder and returns itself without invoking it, so callers may defer
        expensive score computation.
        """
        return self


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

    def subset(self, idx) -> "Variable":
        """This variable restricted to the samples selected by ``idx``.

        A derived variable holds values on both itself and its base variable,
        so the selection is applied down the chain: every variable reachable
        from the copy describes the same samples, in the same order.

        Parameters
        ----------
        idx: np.ndarray
            Integer or boolean index into the sample axis.
        """
        new_variable = copy.copy(self)
        new_variable.values = np.asarray(self.values)[idx]
        if self.base_variable is not None:
            new_variable.base_variable = self.base_variable.subset(idx)
        return new_variable

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
        if not isinstance(other, Variable):
            return NotImplemented
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
                    new_values = disjunction([self.values, other.values])
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
        merged_values = conjunction([v.values for v in self.variables])

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
        # Canonical order: the tuple is used as an identity key (e.g. in
        # `utils.group_by_descriptor`), so it must not depend on the display
        # order of `self.variables`
        return tuple(sorted(v.base_variable for v in self.variables))

    @property
    def partial_values(self) -> np.ndarray:
        """For each sample, the fraction of the group's variables it satisfies."""
        return np.mean(np.vstack([v.values for v in self.variables]), axis=0)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, frozenset(self.variables)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return frozenset(self.variables) == frozenset(other.variables)

    def ranked_by(
        self, compute_scores: Callable[[], dict]
    ) -> "IndicatorVariableGroup":
        """A copy of the group with its variables ordered by descending score.

        Ties are broken by the variables' natural order, which makes the
        ranking deterministic, and a NaN score is treated as the lowest
        possible score — as a sort key it would corrupt the whole ordering.
        Descriptor objects can be shared between region annotations, so the
        group is copied rather than reordered in place.
        """
        scores = compute_scores()

        def sort_key(v):
            return -np.inf if np.isnan(scores[v]) else scores[v]

        ranked = sorted(sorted(self.variables), key=sort_key, reverse=True)

        ranked_group = copy.copy(self)
        ranked_group.variables = ranked
        return ranked_group

    def format_label(
        self,
        max_descriptors: int = None,
        truncation_template: str = "(+{n} more)",
        sep: str = "\n",
        truncation_sep: str = None,
    ) -> str:
        """The group's display text: the variables joined with `sep`,
        truncated to the first `max_descriptors` of them.

        Truncation is display-only: `values` and everything computed from the
        full variable set are unaffected. A truncated label always ends with a
        marker showing the number of hidden variables.

        Parameters
        ----------
        max_descriptors: int
            The maximum number of variables to display. When None or at least
            one less than the group size, the full label is returned unchanged
            — a marker standing in for a single variable would not shorten the
            label.
        truncation_template: str
            Template for the truncation marker; ``{n}`` is replaced with the
            number of hidden variables.
        sep: str
            Separator between variables. The default, one variable per line,
            suits rule-style descriptors; ``", "`` reads better for groups of
            bare names.
        truncation_sep: str
            Separator between the last displayed variable and the truncation
            marker. None means `sep`; with ``sep=", "``, passing ``" "`` gives
            ``... water (+4 more)`` rather than ``... water, (+4 more)``.
        """
        _validate_format_label_args(
            max_descriptors, truncation_template, sep, truncation_sep
        )
        if truncation_sep is None:
            truncation_sep = sep
        if max_descriptors is None or max_descriptors >= len(self.variables) - 1:
            return sep.join(str(v) for v in self.variables)

        label = sep.join(str(v) for v in self.variables[:max_descriptors])
        marker = truncation_template.format(n=len(self.variables) - max_descriptors)
        return label + truncation_sep + marker

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.variables)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join([str(v) for v in self.variables])}])"
