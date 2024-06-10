import unittest

import numpy as np
import pandas as pd

import vera.preprocessing as pp
from vera.variables import (
    Variable,
    ContinuousVariable,
    DiscreteVariable,
    IndicatorVariable,
)


def find_var(variables, name):
    result = [v for v in variables if v.name == name]
    if len(result) == 0:
        raise RuntimeError(f"No variables with `name='{name}'` were found!")
    if len(result) > 1:
        raise RuntimeError(f"Multiple variables with `name='{name}'` were found!")
    return result[0]


def find_derived_vars(variables, name):
    result = [
        v for v in variables
        if v.base_variable is not None and v.base_variable.name == name
    ]
    return result


class TestIngest(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        df["cont2"] = [2.1, 5.2, 5.1, 5.2, 2.0]
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        df["disc2"] = pd.Categorical(["r", "g", "b", "r", "b"])
        df["disc3"] = pd.Categorical(
            ["low", "low", "high", "med", "high"],
            categories=["low", "med", "high"],
            ordered=True,
        )
        self.df = df

    def test_ingest_with_raw_dataframe(self):
        result = pp.ingest(self.df)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(v, Variable) for v in result))
        self.assertTrue(all(isinstance(v.name, str) for v in result))

    def test_ingest_with_mixed_types_dataframe(self):
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        v = ContinuousVariable(
            "cont2", [2.1, 5.2, 5.1, 5.2, 2.0]
        )
        df[v] = v.values
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        v = DiscreteVariable(
            "disc2",
            pd.Categorical(["r", "g", "b", "r", "b"]),
            categories=["r", "g", "b"]
        )
        df[v] = v.values
        df["disc3"] = pd.Categorical(
            ["low", "low", "high", "med", "high"],
            categories=["low", "med", "high"],
            ordered=True,
        )

        result = pp.ingest(df)

        self.assertEqual(len(result), len(df.columns))
        self.assertTrue(all(isinstance(v, Variable) for v in result))
        self.assertTrue(all(isinstance(v.name, str) for v in result))

        cont1_var = find_var(result, "cont1")
        self.assertTrue(isinstance(cont1_var, ContinuousVariable))
        np.testing.assert_equal(cont1_var.values, df["cont1"].values)

        disc1_var = find_var(result, "disc1")
        self.assertTrue(isinstance(disc1_var, DiscreteVariable))
        np.testing.assert_equal(disc1_var.values, df["disc1"].values.codes)

        disc3_var = find_var(result, "disc3")
        self.assertTrue(isinstance(disc3_var, DiscreteVariable))
        np.testing.assert_equal(disc3_var.values, df["disc3"].values.codes)

    def test_ingest_with_nans(self):
        df = self.df
        df["nans"] = [1, 2, np.nan, np.nan, 5]
        result = pp.ingest(df)

        nans_var = find_var(result, "nans")
        self.assertEqual(len(result), len(df.columns))
        self.assertTrue(isinstance(nans_var, ContinuousVariable))
        np.testing.assert_equal(nans_var.values, df["nans"].values)


class TestIngestedToPandas(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        df["cont2"] = [2.1, 5.2, 5.1, 5.2, 2.0]
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        df["disc2"] = pd.Categorical(["r", "g", "b", "r", "b"])
        df["disc3"] = pd.Categorical(
            ["low", "low", "high", "med", "high"],
            categories=["low", "med", "high"],
            ordered=True,
        )
        self.df = df
        self.df_ingested = pp.ingest(df)

    def test_ingested_to_pandas(self):
        reverted = pp.ingested_to_pandas(self.df_ingested)
        self.assertTrue(self.df.equals(reverted))

    def test_with_nans(self):
        df = self.df
        df["cont_na"] = [3.0, np.nan, 1.2, np.nan, 5.1]
        df["disc_na"] = pd.Categorical([0, np.nan, 1, 0, 1])

        reverted = pp.ingested_to_pandas(pp.ingest(df))
        self.assertTrue(df.equals(reverted))


class TestDiscretize(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        df["cont2"] = [2.1, 5.2, 5.1, 5.2, 2.0]
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        df["disc2"] = pd.Categorical(["r", "g", "b", "r", "b"])
        df["disc3"] = pd.Categorical(
            ["low", "low", "high", "med", "high"],
            categories=["low", "med", "high"],
            ordered=True,
        )
        self.df = df

    def test_discretization_with_no_continuous_columns(self):
        df = self.df[["disc1", "disc2", "disc3"]]
        result = pp._discretize(df)
        self.assertEqual(df.shape[1], len(result))

    def test_discretization_retains_number_of_samples(self):
        result = pp._discretize(self.df)
        self.assertTrue(
            all(self.df.shape[0] == v.values.shape[0] for v in result),
            "Discretization changed number of instances!",
        )

    def test_discretize_mixed_variable_types(self):
        result = pp._discretize(self.df)
        self.assertFalse(
            any(isinstance(v, ContinuousVariable) for v in result),
            "Discretized dataframe contains continuous variables!",
        )

        self.assertTrue(
            any(isinstance(v, IndicatorVariable) for v in result),
            "Discretized dataframe contains no explanatory variables!",
        )

    def test_discretization_on_only_continuous_variables(self):
        result = pp._discretize(self.df[["cont1", "cont2"]])
        self.assertTrue(
            all(isinstance(v, IndicatorVariable) for v in result),
            "Discretized dataframe should contain only explanatory variables!",
        )

    def test_discretization_with_constant_feature(self):
        df_const = pd.DataFrame(index=self.df.index)
        df_const["const"] = 0
        result = pp._discretize(df_const)

        self.assertEqual(
            len(result),
            1,
            "Discretization of constant continuous variable produced more than "
            "one variable."
        )

        np.testing.assert_equal(result[0].values, 1)

    def test_discretize_with_nans(self):
        df = self.df
        # Add a continuous column containing NaNs
        df["nans"] = [1, 2, np.nan, np.nan, 5]
        result = pp._discretize(df)

        # Construct dataframe using only the NaN columns
        nan_vars = find_derived_vars(result, "nans")
        nan_cols_df = pp.ingested_to_pandas(nan_vars)

        # Ensure that the rows that had the NaNs haven't been assigned to any
        # particular bin
        nan_mask = df["nans"].isna()
        self.assertEqual(np.sum(nan_cols_df[nan_mask].values), 0, "NaNs mapped to bin!")

    def test_discretization_preserves_column_order(self):
        # TODO: Implement test this
        ...

    def test_discretization_correctly_sets_up_base_variable(self):
        df = self.df[["cont1"]]
        ingested = pp.ingest(df)

        assert len(ingested) == 1
        assert isinstance(ingested[0], ContinuousVariable)

        result = pp._discretize(ingested, n_bins=2)
        self.assertEqual(len(result), 2)

        self.assertTrue(all(isinstance(v, IndicatorVariable) for v in result))
        self.assertTrue(all(isinstance(v.base_variable, ContinuousVariable) for v in result))
        self.assertTrue(all(v.base_variable is ingested[0] for v in result))


class TestOneHotEncoding(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        df["cont2"] = [2.1, 5.2, 5.1, 5.2, 2.0]
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        df["disc2"] = pd.Categorical(["r", "g", "b", "r", "b"])
        df["disc3"] = pd.Categorical(
            ["low", "low", "high", "med", "high"],
            categories=["low", "med", "high"],
            ordered=True,
        )
        self.df = df

    def test_one_hot_encoding_with_no_discrete_columns(self):
        df = self.df[["cont1", "cont2"]]
        result = pp._one_hot(df)
        self.assertEqual(df.shape[1], len(result))

    def test_one_hot_encoding_retains_number_of_samples(self):
        result = pp._one_hot(self.df)

        self.assertTrue(
            all(self.df.shape[0] == v.values.shape[0] for v in result),
            "One-hot encoding changed number of instances!",
        )

    def test_one_hot_encoding_on_only_categorical_variables(self):
        result = pp._one_hot(self.df[["disc1"]])
        self.assertEqual(len(result), 2)

        result = pp._one_hot(self.df[["disc2"]])
        self.assertEqual(len(result), 3)

        result = pp._one_hot(self.df[["disc3"]])
        self.assertEqual(len(result), 3)

        result = pp._one_hot(self.df[["disc1", "disc2", "disc3"]])
        self.assertEqual(len(result), 8)

        self.assertTrue(
            all(isinstance(c, IndicatorVariable) for c in result),
            "Encoded dataframe should contain only explanatory variables!",
        )

    def test_one_hot_encoding_on_mixed_type(self):
        result = pp._one_hot(self.df)
        self.assertEqual(len(result), 10)

        self.assertFalse(
            any(type(v) is DiscreteVariable for v in result),
            "Encoded dataframe contains discrete variables!",
        )

        self.assertTrue(
            any(isinstance(v, IndicatorVariable) for v in result),
            "Encoded dataframe contains no explanatory variables!",
        )

    def test_one_hot_encoding_with_no_discrete_variables(self):
        ingested = pp.ingest(self.df[["cont1", "cont2"]])
        result = pp._one_hot(ingested)
        self.assertTrue(tuple(ingested) == tuple(result))

        ingested = pp._discretize(pp.ingest(self.df[["cont1", "cont2"]]))
        result = pp._one_hot(ingested)
        self.assertTrue(tuple(ingested) == tuple(result))

    def test_one_hot_encoding_with_nans(self):
        df = self.df
        # Add a categorical column containing NaNs
        df["nans"] = pd.Categorical([0, np.nan, 1, 0, 1])
        
        result = pp._one_hot(df)

        # Construct dataframe using only the NaN columns
        nan_vars = find_derived_vars(result, "nans")
        nan_cols_df = pp.ingested_to_pandas(nan_vars)

        # Ensure that the rows that had the NaNs haven't been assigned to any
        # particular bin
        nan_mask = df["nans"].isna()
        self.assertEqual(np.sum(nan_cols_df[nan_mask].values), 0, "NaNs mapped to bin!")

    def test_one_hot_encoding_preserves_column_order(self):
        # TODO: Implement test this
        ...

    def test_one_hot_encoding_correctly_sets_up_base_variable(self):
        df = self.df[["disc2"]]
        ingested = pp.ingest(df)

        assert len(ingested) == 1
        assert isinstance(ingested[0], DiscreteVariable)
        assert len(ingested[0].categories) == 3

        result = pp._one_hot(ingested)
        self.assertEqual(len(result), 3)

        self.assertTrue(all(isinstance(v, IndicatorVariable) for v in result))
        self.assertTrue(all(isinstance(v.base_variable, DiscreteVariable) for v in result))
        self.assertTrue(all(v.base_variable is ingested[0] for v in result))
