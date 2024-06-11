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

    def test_ingest_series(self):
        series = self.df["cont1"]
        assert isinstance(series, pd.Series)
        result = pp.ingest(series)

        self.assertIsInstance(result, ContinuousVariable)
        self.assertEqual(result.name, "cont1")
        np.testing.assert_equal(result.values, series.values)


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
        self.cont1 = pp.ingest(pd.Series([5, 2, 3, 1, 5], name="cont1"))
        self.cont2 = pp.ingest(pd.Series([2.1, 5.2, 5.1, 5.2, 2.0, 1.2], name="cont2"))

    def test_discretization_retains_number_of_samples(self):
        result = pp.discretize(self.cont1)
        self.assertTrue(
            all(self.cont1.values.shape[0] == v.values.shape[0] for v in result),
            "Discretization changed number of instances!",
        )

    def test_standard_discretization(self):
        for n_bins in [2, 3, 4, 5]:
            result = pp.discretize(self.cont2, n_bins=n_bins)

            self.assertEqual(
                len(result),
                n_bins,
                f"Discretization with `n_bins={n_bins}` produced incorrect "
                f"number of indicator variables"
            )
            self.assertTrue(
                all(isinstance(v, IndicatorVariable) for v in result),
                "Discretized dataframe should contain only explanatory variables!",
            )
            # Ensure that each sample was mapped to bin
            merged_df = pp.ingested_to_pandas(result)
            np.testing.assert_equal(merged_df.sum(axis=1).values, 1)

    def test_discretization_with_constant_feature(self):
        df_const = pp.ingest(pd.Series([0] * 10))
        result = pp.discretize(df_const)

        self.assertEqual(
            len(result),
            1,
            "Discretization of constant continuous variable produced more than "
            "one variable."
        )
        self.assertTrue(
            all(isinstance(v, IndicatorVariable) for v in result),
            "Discretized dataframe should contain only explanatory variables!",
        )

        np.testing.assert_equal(result[0].values, 1)

    def test_discretize_with_nans(self):
        data = pd.Series([1, 2, np.nan, np.nan, 5], name="nans")
        variable = pp.ingest(data)
        result = pp.discretize(variable)

        # Construct dataframe using only the NaN columns
        nan_cols_df = pp.ingested_to_pandas(result)

        # Ensure that the rows that had the NaNs haven't been assigned to any
        # particular bin
        nan_mask = data.isna()
        self.assertEqual(np.sum(nan_cols_df[nan_mask].values), 0, "NaNs mapped to bin!")

    def test_discretization_correctly_sets_up_base_variable(self):
        result = pp.discretize(self.cont1, n_bins=2)
        self.assertEqual(len(result), 2)

        self.assertTrue(all(isinstance(v, IndicatorVariable) for v in result))
        self.assertTrue(all(isinstance(v.base_variable, ContinuousVariable) for v in result))
        self.assertTrue(all(v.base_variable is self.cont1 for v in result))


class TestOneHotEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.disc1 = pp.ingest(pd.Series(pd.Categorical([0, 0, 1, 0, 1]), name="disc1"))
        self.disc2 = pp.ingest(pd.Series(pd.Categorical(["r", "g", "b", "r", "b"]), name="disc2"))
        self.disc3 = pp.ingest(
            pd.Series(
                pd.Categorical(
                    ["low", "low", "high", "med", "high"],
                    categories=["low", "med", "high"],
                    ordered=True,
                ),
                name="disc3",
            )
        )

    def test_one_hot_encoding_retains_number_of_samples(self):
        result = pp.one_hot(self.disc1)

        self.assertTrue(
            all(self.disc1.values.shape[0] == v.values.shape[0] for v in result),
            "One-hot encoding changed number of instances!",
        )

    def test_standard_one_hot_encoding(self):
        for v, num_cats in [
            (self.disc1, 2),
            (self.disc2, 3),
            (self.disc3, 3),
        ]:
            result = pp.one_hot(v)
            self.assertEqual(len(result), num_cats)

            self.assertTrue(
                all(isinstance(c, IndicatorVariable) for c in result),
                "Encoded dataframe should contain only explanatory variables!",
            )
            # Ensure that each sample was mapped to bin
            merged_df = pp.ingested_to_pandas(result)
            np.testing.assert_equal(merged_df.sum(axis=1).values, 1)

    def test_one_hot_encoding_with_nans(self):
        series = pd.Series(pd.Categorical([0, np.nan, 1, 0, 1]), name="nans")
        result = pp.one_hot(pp.ingest(series))

        # Construct dataframe using only the NaN columns
        nan_cols_df = pp.ingested_to_pandas(result)

        # Ensure that the rows that had the NaNs haven't been assigned to any
        # particular bin
        nan_mask = series.isna()
        self.assertEqual(np.sum(nan_cols_df[nan_mask].values), 0, "NaNs mapped to bin!")

    def test_one_hot_encoding_correctly_sets_up_base_variable(self):
        result = pp.one_hot(self.disc2)
        self.assertEqual(len(result), 3)

        self.assertTrue(all(isinstance(v, IndicatorVariable) for v in result))
        self.assertTrue(all(isinstance(v.base_variable, DiscreteVariable) for v in result))
        self.assertTrue(all(v.base_variable is self.disc2 for v in result))
