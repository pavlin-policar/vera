import unittest

import numpy as np
import pandas as pd

import veca.rules
import veca.variables
import veca.preprocessing as pp


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
        self.assertTrue(all(isinstance(c, veca.variables.Variable) for c in result.columns))
        self.assertTrue(all(isinstance(c.name, str) for c in result.columns))

    def test_ingest_with_mixed_types_dataframe(self):
        df = pd.DataFrame()
        df["cont1"] = [5, 2, 3, 1, 5]
        v = veca.variables.ContinuousVariable(
            "cont2", [2.1, 5.2, 5.1, 5.2, 2.0]
        )
        df[v] = v.values
        df["disc1"] = pd.Categorical([0, 0, 1, 0, 1])
        v = veca.variables.DiscreteVariable(
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
        self.assertTrue(all(isinstance(c, veca.variables.Variable) for c in result.columns))
        self.assertTrue(all(isinstance(c.name, str) for c in result.columns))

    def test_already_ingested_dataframe_does_nothing(self):
        ingested = pp.ingest(self.df)
        result = pp.ingest(ingested)
        self.assertTrue(all(isinstance(c, veca.variables.Variable) for c in result.columns))
        self.assertTrue(all(isinstance(c.name, str) for c in result.columns))


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
        df_discretized = pp._discretize(df)
        self.assertEqual(df.shape, df_discretized.shape)

    def test_discretization_retains_number_of_samples(self):
        df_discretized = pp._discretize(self.df)
        self.assertEqual(
            self.df.shape[0],
            df_discretized.shape[0],
            "Discretization changed number of instances!",
        )

    def test_discretize_mixed_variable_types(self):
        df_discretized = pp._discretize(self.df)
        self.assertFalse(
            any(isinstance(c, veca.variables.ContinuousVariable) for c in df_discretized.columns),
            "Discretized dataframe contains continuous variables!",
        )

        self.assertTrue(
            any(
                isinstance(c, veca.variables.DerivedVariable) for c in df_discretized.columns
            ),
            "Discretized dataframe contains no explanatory variables!",
        )

    def test_discretization_on_only_continuous_variables(self):
        df_discretized = pp._discretize(self.df[["cont1", "cont2"]])
        self.assertTrue(
            all(
                isinstance(c, veca.variables.DerivedVariable) for c in df_discretized.columns
            ),
            "Discretized dataframe should contain only explanatory variables!",
        )

    def test_discretization_with_constant_feature(self):
        df_const = pd.DataFrame(index=self.df.index)
        df_const["const"] = 0
        df_discretized = pp._discretize(df_const)

        self.assertEqual(
            len(df_discretized.columns),
            1,
            "Discretization of constant continuous variable produced more than "
            "one variable."
        )

        np.testing.assert_equal(df_discretized.values, 1)


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
        df_continuized = pp._one_hot(df)
        self.assertEqual(df.shape, df_continuized.shape)

    def test_one_hot_encoding_retains_number_of_samples(self):
        df_one_hot = pp._one_hot(self.df)
        self.assertEqual(
            self.df.shape[0],
            df_one_hot.shape[0],
            "One-hot encoding changed number of instances!",
        )

    def test_one_hot_encoding_on_only_categorical_variables(self):
        df_encoded = pp._one_hot(self.df[["disc1"]])
        self.assertEqual(df_encoded.shape[1], 2)

        df_encoded = pp._one_hot(self.df[["disc2"]])
        self.assertEqual(df_encoded.shape[1], 3)

        df_encoded = pp._one_hot(self.df[["disc3"]])
        self.assertEqual(df_encoded.shape[1], 3)

        df_encoded = pp._one_hot(self.df[["disc1", "disc2", "disc3"]])
        self.assertEqual(df_encoded.shape[1], 8)

        self.assertTrue(
            all(isinstance(c, veca.variables.DerivedVariable) for c in df_encoded.columns),
            "Encoded dataframe should contain only explanatory variables!",
        )

    def test_one_hot_encoding_on_mixed_type(self):
        df_encoded = pp._one_hot(self.df)
        self.assertEqual(df_encoded.shape[1], 10)

        self.assertFalse(
            any(type(c) is veca.variables.DiscreteVariable for c in df_encoded.columns),
            "Encoded dataframe contains discrete variables!",
        )

        self.assertTrue(
            any(isinstance(c, veca.variables.DerivedVariable) for c in df_encoded.columns),
            "Encoded dataframe contains no explanatory variables!",
        )

    def test_one_hot_encoding_with_no_discrete_variables(self):
        df = pp.ingest(self.df[["cont1", "cont2"]])
        df_encoded = pp._one_hot(df)
        self.assertTrue(df.equals(df_encoded))

        df = pp._discretize(pp.ingest(self.df[["cont1", "cont2"]]))
        df_encoded = pp._one_hot(df)
        self.assertTrue(df.equals(df_encoded))
