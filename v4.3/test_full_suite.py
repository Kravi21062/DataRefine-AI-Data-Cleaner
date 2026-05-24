import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import data_utils as du
import chart_utils as cu
import pdf_utils as pu

# test full suite
class TestFullSuite(unittest.TestCase):
    # setup test dataframe
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "num1": [1, 2, np.nan, 4, -5, 1000],
                "num2": [10, 20, 30, np.nan, 50, 60],
                "cat": ["A", "B", "B", " ", "NA", None],
                "date_col": ["2024-01-01", "2024-01-02", "2024-01-03", None, "2024-01-05", "2024-01-06"],
            }
        )


    # data_utils
    # -------------------------
    def test_basic_metrics_and_quality(self):
        m = du.compute_basic_metrics(self.df)
        self.assertEqual(m["rows"], 6)
        self.assertEqual(m["cols"], 4)
        q = du.compute_quality_score(self.df)
        self.assertTrue(0.0 <= q <= 100.0)

    # missing value testing
    def test_missing_value_strategies(self):
        for strategy in ["Drop", "Mean", "Median", "Mode", "KNN Imputation"]:
            out = du.handle_missing_values(self.df, strategy)
            self.assertIsInstance(out, pd.DataFrame)
            self.assertEqual(out.shape[1], self.df.shape[1])

    # outlier detection testing
    def test_outlier_detection(self):
        iqr_flags = du.detect_outliers_iqr(self.df)
        z_flags = du.detect_outliers_zscore(self.df, threshold=2.0)
        self.assertEqual(len(iqr_flags), len(self.df))
        self.assertEqual(len(z_flags), len(self.df))
        self.assertTrue(bool(iqr_flags.iloc[-1]))

    # column operations testing
    def test_column_ops_cleaning_encoding_scaling(self):
        renamed = du.apply_column_operation(self.df, "Rename column", "num1", new_name="num_one")
        self.assertIn("num_one", renamed.columns)

        typed = du.apply_column_operation(self.df, "Change datatype", "date_col", target_type="datetime")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(typed["date_col"]))

        dropped = du.apply_column_operation(self.df, "Drop column", "num2")
        self.assertNotIn("num2", dropped.columns)

        enc = du.encode_categorical(self.df, "cat", method="Label")
        self.assertIn("cat_encoded", enc.columns)
        onehot = du.encode_categorical(self.df, "cat", method="One-Hot")
        self.assertNotIn("cat", onehot.columns)

        cleaned = du.clean_column_names(pd.DataFrame({"A B/C-D": [1]}))
        self.assertIn("a_b_c_d", cleaned.columns)

        stdtxt = du.standardize_text_columns(self.df, columns=["cat"], lowercase=True, strip=True)
        self.assertEqual(stdtxt.loc[0, "cat"], "a")

        replaced = du.replace_invalid_values(self.df, treat_negative_as_invalid=True)
        self.assertTrue(pd.isna(replaced.loc[4, "num1"]))

        mm = du.scale_numeric(self.df, method="Min-Max")
        std = du.scale_numeric(self.df, method="Standard")
        self.assertIn("num1_scaled", mm.columns)
        self.assertIn("num1_std", std.columns)

    # rule-based suggestions testing
    @patch("data_utils._call_ollama_phi3", return_value=None)
    def test_rule_based_suggestions(self, _):
        s = du.generate_cleaning_suggestions(self.df)
        self.assertIn("Source", s.columns)
        self.assertTrue((s["Source"] == "Rule-based (Ollama unavailable)").all())

    # ai suggestions testing
    @patch(
        "data_utils._call_ollama_phi3",
        return_value=[{"Column": "num1", "Issue Detected": "Outliers", "Suggestion": "Cap", "Severity": "High"}],
    )
    def test_ai_suggestions(self, _):
        s = du.generate_cleaning_suggestions(self.df)
        self.assertEqual(s["Source"].iloc[0], "AI (phi3)")

    # comparison and insights testing
    def test_comparison_and_insights(self):
        clean = self.df.fillna(0).drop_duplicates()
        comp, mb, ma = du.build_before_after_comparison(self.df, clean)
        self.assertEqual(comp.shape[0], 6)
        self.assertIsInstance(mb, dict)
        self.assertIsInstance(ma, dict)

        trend_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12, freq="D"),
                "x": np.arange(12),
                "y": np.arange(12) * 3 + 1,
                "cat": ["A"] * 8 + ["B"] * 4,
            }
        )
        insights = du.ai_insight_analysis(trend_df)
        self.assertIsInstance(insights, list)
        self.assertLessEqual(len(insights), 4)

    # chart_utils
    # -------------------------
    @patch("chart_utils.st.plotly_chart")
    @patch("chart_utils.st.info")
    def test_chart_missing_bar(self, m_info, m_plot):
        cu.plot_missing_values_bar(self.df)
        self.assertTrue(m_plot.called or m_info.called)

    # missing heatmap testing
    @patch("chart_utils.st.plotly_chart")
    @patch("chart_utils.st.info")
    def test_chart_missing_heatmap(self, m_info, m_plot):
        cu.plot_missing_values_heatmap(self.df)
        self.assertTrue(m_plot.called or m_info.called)

    # correlation and distributions testing
    @patch("chart_utils.st.plotly_chart")
    @patch("chart_utils.st.info")
    def test_chart_corr_and_distributions(self, m_info, m_plot):
        cu.plot_correlation_heatmap(self.df)
        cu.plot_distributions(self.df)
        self.assertTrue(m_plot.called or m_info.called)

    # before and after boxplot, gauge, and bar testing
    @patch("chart_utils.st.plotly_chart")
    def test_chart_box_and_gauge_and_bar(self, m_plot):
        cu.plot_before_after_boxplot(self.df["num1"], self.df["num1"].fillna(0), "num1")
        cu.render_quality_gauge(42.5, 84.0)
        comp, _, _ = du.build_before_after_comparison(self.df, self.df.fillna(0))
        figs = cu.plot_before_after_bar(comp)
        self.assertIsInstance(figs, list)
        self.assertIsInstance(figs[0], go.Figure)
        self.assertTrue(m_plot.called)

    # insight chart testing
    @patch("chart_utils.st.plotly_chart")
    @patch("chart_utils.st.info")
    @patch("chart_utils.st.caption")
    def test_plot_insight_chart_all_types(self, _cap, m_info, m_plot):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "x": np.arange(20),
                "y": np.arange(20) * 2,
                "cat": ["A"] * 12 + ["B"] * 8,
            }
        )
        cu.plot_insight_chart(df, {"type": "distribution", "columns": ["x"], "title": "dist"})
        cu.plot_insight_chart(df, {"type": "correlation", "columns": ["x", "y"], "title": "corr"})
        cu.plot_insight_chart(df, {"type": "category", "columns": ["cat"], "title": "cat"})
        cu.plot_insight_chart(df, {"type": "trend", "columns": ["date", "x"], "title": "trend"})
        cu.plot_insight_chart(df, {"type": "unknown", "columns": [], "title": "unknown"})
        self.assertTrue(m_plot.called or m_info.called)

    # pdf_utils coverage
    # -------------------------
    def test_pdf_helpers(self):
        styles = pu._build_styles()
        self.assertIn("title", styles)
        self.assertEqual(pu.QualityBar(50).wrap()[1], 16)

        mb = du.compute_basic_metrics(self.df)
        ma = du.compute_basic_metrics(self.df.fillna(0))
        self.assertIsNotNone(pu._chart_before_after(mb, ma))
        self.assertIsNotNone(pu._chart_missing_bar(self.df))
        self.assertIsNotNone(pu._chart_dist_shift(self.df, self.df.fillna(0)))
        self.assertIsNotNone(pu._chart_distributions(self.df.fillna(0)))
        self.assertIsNotNone(pu._chart_corr(self.df.fillna(0)))

        self.assertIsNotNone(pu._comparison_table([("Rows", 10, 8)], styles))
        self.assertIsNotNone(pu._schema_table(self.df, styles))
        self.assertIsNotNone(pu._preview_table(self.df, styles))
        self.assertIsNotNone(pu._cleaning_summary_table([("Drop dup", self.df, ["Dropped 2 dupes"], "Manual")], styles))

    # pdf report building testing
    def test_build_pdf_report(self):
        raw = self.df.copy()
        clean = self.df.fillna(0).drop_duplicates()
        pdf_bytes = pu.build_pdf_report(
            raw_df=raw,
            clean_df=clean,
            file_name="sample.csv",
            cleaning_history=[("Auto Clean", clean, ["Imputed missing", "Removed outliers"], "Rule-based")],
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    # invalid input testing
    def test_build_pdf_report_invalid_input(self):
        with self.assertRaises(ValueError):
            pu.build_pdf_report(raw_df=None, clean_df=self.df)

if __name__ == "__main__":
    unittest.main(verbosity=2)
