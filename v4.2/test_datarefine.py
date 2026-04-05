"""
test_datarefine.py
==================
Pytest test suite for DataRefine — AI Data Cleaning Tool
Delhi Skill Entrepreneurship University | BCA Sem 6 | Roll No: 3323

Run:
    pip install pytest
    pytest test_datarefine.py -v
"""

import sys
import unittest.mock as mock

# ── Mock UI-only packages so tests run without a browser ──────────────────────
for _mod in ["streamlit", "plotly", "plotly.graph_objects", "plotly.express"]:
    sys.modules[_mod] = mock.MagicMock()

import numpy as np
import pandas as pd
import pytest

import data_utils as du


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures  (reusable sample DataFrames)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small DataFrame with missing values and duplicates."""
    return pd.DataFrame({
        "age":    [25, 30, None, 45, 22, 30, 25],
        "salary": [50000, 75000, 60000, None, 45000, 75000, 50000],
        "dept":   ["HR", "IT", "IT", None, "Sales", "IT", "HR"],
        "score":  [88, 92, 76, 85, None, 92, 88],
    })


@pytest.fixture
def clean_df():
    """DataFrame with no issues — perfect quality."""
    return pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "dept":   ["HR", "IT", "Sales", "Finance", "Ops"],
        "score":  [80, 85, 90, 75, 88],
    })


@pytest.fixture
def numeric_df():
    """Pure numeric DataFrame."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def outlier_df():
    """DataFrame with one extreme outlier."""
    return pd.DataFrame({
        "value": [10, 12, 11, 13, 10, 12, 500],   # 500 is outlier
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_basic_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBasicMetrics:

    def test_row_count(self, sample_df):
        m = du.compute_basic_metrics(sample_df)
        assert m["rows"] == 7

    def test_col_count(self, sample_df):
        m = du.compute_basic_metrics(sample_df)
        assert m["cols"] == 4

    def test_missing_count(self, sample_df):
        # age=1, salary=1, dept=1, score=1 → total 4
        m = du.compute_basic_metrics(sample_df)
        assert m["missing"] == 4

    def test_duplicate_count(self, sample_df):
        # rows 0&6 (age=25,salary=50000,dept=HR,score=88) and rows 1&5 (IT)
        m = du.compute_basic_metrics(sample_df)
        assert m["duplicates"] == 2

    def test_no_issues(self, clean_df):
        m = du.compute_basic_metrics(clean_df)
        assert m["missing"] == 0
        assert m["duplicates"] == 0

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1, 1]})
        m = du.compute_basic_metrics(df)
        assert m["duplicates"] == 3   # first row is original, 3 are dupes

    def test_returns_dict_with_correct_keys(self, sample_df):
        m = du.compute_basic_metrics(sample_df)
        assert set(m.keys()) == {"rows", "cols", "missing", "duplicates"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_quality_score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeQualityScore:

    def test_perfect_data_scores_100(self, clean_df):
        score = du.compute_quality_score(clean_df)
        assert score == 100.0

    def test_empty_df_scores_zero(self):
        score = du.compute_quality_score(pd.DataFrame())
        assert score == 0.0

    def test_score_in_valid_range(self, sample_df):
        score = du.compute_quality_score(sample_df)
        assert 0.0 <= score <= 100.0

    def test_missing_reduces_score(self, sample_df, clean_df):
        score_with_missing = du.compute_quality_score(sample_df)
        score_clean        = du.compute_quality_score(clean_df)
        assert score_with_missing < score_clean

    def test_all_missing_score_below_60(self):
        df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
        score = du.compute_quality_score(df)
        assert score < 60.0

    def test_score_is_float(self, sample_df):
        assert isinstance(du.compute_quality_score(sample_df), float)

    def test_duplicates_reduce_score(self):
        df_dupes = pd.DataFrame({"a": [1, 1, 1, 1, 1]})
        df_clean = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        assert du.compute_quality_score(df_dupes) < du.compute_quality_score(df_clean)


# ─────────────────────────────────────────────────────────────────────────────
# 3. handle_missing_values
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleMissingValues:

    def test_drop_removes_rows_with_nan(self, sample_df):
        result = du.handle_missing_values(sample_df, "Drop")
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(sample_df)

    def test_mean_fills_all_missing(self, sample_df):
        result = du.handle_missing_values(sample_df, "Mean")
        assert result.isnull().sum().sum() == 0
        assert result.shape == sample_df.shape

    def test_median_fills_all_missing(self, sample_df):
        result = du.handle_missing_values(sample_df, "Median")
        assert result.isnull().sum().sum() == 0

    def test_mode_fills_all_missing(self, sample_df):
        result = du.handle_missing_values(sample_df, "Mode")
        assert result.isnull().sum().sum() == 0

    def test_knn_fills_all_missing(self, sample_df):
        result = du.handle_missing_values(sample_df, "KNN Imputation")
        assert result.isnull().sum().sum() == 0

    def test_does_not_modify_original(self, sample_df):
        original_missing = sample_df.isnull().sum().sum()
        du.handle_missing_values(sample_df, "Mean")
        assert sample_df.isnull().sum().sum() == original_missing

    def test_mean_numeric_value_is_reasonable(self):
        df = pd.DataFrame({"x": [10.0, 20.0, None, 40.0]})
        result = du.handle_missing_values(df, "Mean")
        # Mean of 10,20,40 = 23.33
        assert 20 <= result["x"].iloc[2] <= 30


# ─────────────────────────────────────────────────────────────────────────────
# 4. Outlier Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestOutlierDetection:

    def test_iqr_detects_extreme_outlier(self, outlier_df):
        flags = du.detect_outliers_iqr(outlier_df)
        assert flags.iloc[-1] == True      # 500 should be flagged
        assert flags.iloc[0]  == False     # 10 is normal

    def test_iqr_no_outliers_in_clean_data(self, numeric_df):
        flags = du.detect_outliers_iqr(numeric_df)
        # Uniform range — no outliers expected
        assert flags.sum() == 0

    def test_iqr_returns_series_same_length(self, sample_df):
        flags = du.detect_outliers_iqr(sample_df)
        assert len(flags) == len(sample_df)

    def test_iqr_returns_boolean_series(self, outlier_df):
        flags = du.detect_outliers_iqr(outlier_df)
        assert flags.dtype == bool

    def test_zscore_detects_outlier_at_threshold(self, outlier_df):
        flags = du.detect_outliers_zscore(outlier_df, threshold=2.0)
        assert flags.iloc[-1] == True    # 500 is far from mean

    def test_zscore_strict_threshold_catches_more(self, outlier_df):
        flags_3 = du.detect_outliers_zscore(outlier_df, threshold=3.0)
        flags_2 = du.detect_outliers_zscore(outlier_df, threshold=2.0)
        assert flags_2.sum() >= flags_3.sum()

    def test_isolation_forest_returns_tuple(self, sample_df):
        clean = sample_df.dropna()
        result = du.detect_outliers_isolation_forest(clean, contamination=0.1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_isolation_forest_summary_keys(self, sample_df):
        clean = sample_df.dropna()
        _, summary = du.detect_outliers_isolation_forest(clean, contamination=0.1)
        assert "n_anomalies" in summary
        assert "anomaly_ratio" in summary

    def test_isolation_forest_no_numeric_returns_empty(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        flags, summary = du.detect_outliers_isolation_forest(df)
        assert flags.sum() == 0
        assert summary["n_anomalies"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. clean_column_names
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanColumnNames:

    def test_spaces_replaced_with_underscore(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        result = du.clean_column_names(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_hyphens_replaced(self):
        df = pd.DataFrame({"col-one": [1], "col-two": [2]})
        result = du.clean_column_names(df)
        assert "col_one" in result.columns
        assert "col_two" in result.columns

    def test_slashes_replaced(self):
        df = pd.DataFrame({"AGE/YEARS": [1]})
        result = du.clean_column_names(df)
        assert "age_years" in result.columns

    def test_uppercase_converted_to_lowercase(self):
        df = pd.DataFrame({"SALARY": [1], "Department": [2]})
        result = du.clean_column_names(df)
        assert "salary" in result.columns
        assert "department" in result.columns

    def test_original_not_modified(self):
        df = pd.DataFrame({"First Name": [1]})
        du.clean_column_names(df)
        assert "First Name" in df.columns   # original unchanged

    def test_data_preserved(self):
        df = pd.DataFrame({"My Col": [10, 20, 30]})
        result = du.clean_column_names(df)
        assert result["my_col"].tolist() == [10, 20, 30]


# ─────────────────────────────────────────────────────────────────────────────
# 6. standardize_text_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardizeTextColumns:

    def test_lowercase_applied(self):
        df = pd.DataFrame({"name": ["HELLO", "WORLD"]})
        result = du.standardize_text_columns(df, lowercase=True, strip=False)
        assert result["name"].tolist() == ["hello", "world"]

    def test_strip_whitespace(self):
        df = pd.DataFrame({"city": ["  Delhi  ", " Mumbai "]})
        result = du.standardize_text_columns(df, lowercase=False, strip=True)
        assert result["city"].tolist() == ["Delhi", "Mumbai"]

    def test_both_lowercase_and_strip(self):
        df = pd.DataFrame({"tag": ["  HR ", " IT "]})
        result = du.standardize_text_columns(df)
        assert result["tag"].tolist() == ["hr", "it"]

    def test_nan_values_preserved(self):
        df = pd.DataFrame({"col": ["hello", None, "world"]})
        result = du.standardize_text_columns(df)
        assert pd.isna(result["col"].iloc[1])

    def test_non_text_columns_unchanged(self):
        df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
        result = du.standardize_text_columns(df, columns=["name"])
        assert result["age"].tolist() == [25, 30]


# ─────────────────────────────────────────────────────────────────────────────
# 7. replace_invalid_values
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaceInvalidValues:

    def test_common_tokens_replaced_with_nan(self):
        df = pd.DataFrame({"a": pd.Series(
            ["NA", "none", "-", "--", "null", "hello", "nan"], dtype="object"
        )})
        result = du.replace_invalid_values(df)
        assert result["a"].isnull().sum() == 6
        assert result["a"].dropna().tolist() == ["hello"]

    def test_valid_values_unchanged(self):
        df = pd.DataFrame({"col": pd.Series(["Alice", "Bob", "Charlie"], dtype="object")})
        result = du.replace_invalid_values(df)
        assert result["col"].isnull().sum() == 0

    def test_negative_numbers_treated_as_invalid(self):
        df = pd.DataFrame({"score": [-1.0, 5.0, 10.0, -99.0]})
        result = du.replace_invalid_values(df, treat_negative_as_invalid=True)
        assert result["score"].isnull().sum() == 2

    def test_original_not_modified(self):
        df = pd.DataFrame({"x": pd.Series(["NA", "hello"], dtype="object")})
        du.replace_invalid_values(df)
        assert df["x"].isnull().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. encode_categorical
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeCategorical:

    def test_label_encoding_creates_new_column(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        result = du.encode_categorical(df, "color", method="Label")
        assert "color_encoded" in result.columns

    def test_label_encoding_numeric_values(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        result = du.encode_categorical(df, "color", method="Label")
        assert result["color_encoded"].dtype in [np.int32, np.int64, object]
        # All values should be integers 0, 1, or 2
        vals = set(result["color_encoded"].astype(int).tolist())
        assert vals == {0, 1, 2}

    def test_onehot_creates_dummy_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        result = du.encode_categorical(df, "color", method="One-Hot")
        assert "color_blue"  in result.columns
        assert "color_green" in result.columns
        assert "color_red"   in result.columns

    def test_onehot_removes_original_column(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        result = du.encode_categorical(df, "color", method="One-Hot")
        assert "color" not in result.columns

    def test_nonexistent_column_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = du.encode_categorical(df, "nonexistent", method="Label")
        assert result.equals(df)


# ─────────────────────────────────────────────────────────────────────────────
# 9. validate_range
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateRange:

    def test_cap_mode_clips_min(self):
        df = pd.DataFrame({"age": [-5.0, 25.0, 80.0]})
        result = du.validate_range(df, "age", min_value=0, mode="cap")
        assert result["age"].min() >= 0

    def test_cap_mode_clips_max(self):
        df = pd.DataFrame({"age": [20.0, 25.0, 150.0]})
        result = du.validate_range(df, "age", max_value=100, mode="cap")
        assert result["age"].max() <= 100

    def test_nan_mode_sets_out_of_range_to_nan(self):
        df = pd.DataFrame({"age": [-5.0, 25.0, 150.0]})
        result = du.validate_range(df, "age", min_value=0, max_value=100, mode="nan")
        assert result["age"].isnull().sum() == 2
        assert result["age"].dropna().tolist() == [25.0]

    def test_valid_values_unchanged_in_cap_mode(self):
        df = pd.DataFrame({"score": [50.0, 75.0, 90.0]})
        result = du.validate_range(df, "score", min_value=0, max_value=100, mode="cap")
        assert result["score"].tolist() == [50.0, 75.0, 90.0]

    def test_nonexistent_column_returns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = du.validate_range(df, "nonexistent", min_value=0, max_value=10)
        assert result.equals(df)


# ─────────────────────────────────────────────────────────────────────────────
# 10. scale_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleNumeric:

    def test_minmax_creates_scaled_column(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Min-Max")
        assert "x_scaled" in result.columns
        assert "y_scaled" in result.columns

    def test_minmax_range_is_0_to_1(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Min-Max")
        assert result["x_scaled"].min() == pytest.approx(0.0)
        assert result["x_scaled"].max() == pytest.approx(1.0)

    def test_standard_creates_std_column(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Standard")
        assert "x_std" in result.columns

    def test_standard_mean_is_zero(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Standard")
        assert result["x_std"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_standard_std_is_one(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Standard")
        assert result["x_std"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_already_scaled_columns_skipped(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "x_scaled": [0.0, 0.5, 1.0]})
        result = du.scale_numeric(df, method="Min-Max")
        assert "x_scaled_scaled" not in result.columns

    def test_original_columns_preserved(self, numeric_df):
        result = du.scale_numeric(numeric_df, method="Min-Max")
        assert "x" in result.columns
        assert "y" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 11. build_dtype_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDtypeSummary:

    def test_returns_dataframe(self, sample_df):
        result = du.build_dtype_summary(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        result = du.build_dtype_summary(sample_df)
        for col in ["Column", "Data Type", "Missing Count", "Missing %", "Unique Values"]:
            assert col in result.columns

    def test_row_count_equals_column_count(self, sample_df):
        result = du.build_dtype_summary(sample_df)
        assert len(result) == sample_df.shape[1]

    def test_missing_count_correct(self, sample_df):
        result = du.build_dtype_summary(sample_df)
        age_row = result[result["Column"] == "age"]
        assert int(age_row["Missing Count"].values[0]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 12. detect_feature_types
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectFeatureTypes:

    def test_numeric_detected(self):
        df = pd.DataFrame({"age": [25, 30, 35]})
        result = du.detect_feature_types(df)
        assert result.loc[result["Column"] == "age", "Inferred Type"].values[0] == "Numeric"

    def test_categorical_detected(self):
        df = pd.DataFrame({"dept": ["HR", "IT", "Sales", "Finance"]})
        result = du.detect_feature_types(df)
        assert result.loc[result["Column"] == "dept", "Inferred Type"].values[0] == "Categorical"

    def test_returns_required_columns(self, sample_df):
        result = du.detect_feature_types(sample_df)
        for col in ["Column", "Pandas Dtype", "Inferred Type", "Unique Values"]:
            assert col in result.columns

    def test_row_count_equals_column_count(self, sample_df):
        result = du.detect_feature_types(sample_df)
        assert len(result) == sample_df.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 13. generate_cleaning_suggestions
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateCleaningSuggestions:

    def test_returns_dataframe(self, sample_df):
        result = du.generate_cleaning_suggestions(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        result = du.generate_cleaning_suggestions(sample_df)
        for col in ["Column", "Issue Detected", "Suggestion", "Severity"]:
            assert col in result.columns

    def test_detects_missing_value_issues(self, sample_df):
        result = du.generate_cleaning_suggestions(sample_df)
        issues = result["Issue Detected"].str.lower()
        assert issues.str.contains("missing").any()

    def test_clean_data_returns_info_row(self):
        # Needs high-cardinality numeric cols so no 'low cardinality' warning triggers
        df = pd.DataFrame({
            "id":     range(100),
            "value":  [float(i) * 1.1 for i in range(100)],
            "label":  [f"cat_{i % 20}" for i in range(100)],
        })
        result = du.generate_cleaning_suggestions(df)
        assert result["Severity"].iloc[0] == "Info"

    def test_empty_df_returns_empty_suggestions(self):
        result = du.generate_cleaning_suggestions(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_severity_values_are_valid(self, sample_df):
        result = du.generate_cleaning_suggestions(sample_df)
        valid = {"High", "Medium", "Low", "Info"}
        assert set(result["Severity"].unique()).issubset(valid)


# ─────────────────────────────────────────────────────────────────────────────
# 14. build_before_after_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBeforeAfterComparison:

    def test_returns_tuple_of_three(self, sample_df, clean_df):
        result = du.build_before_after_comparison(sample_df, clean_df)
        assert len(result) == 3

    def test_comparison_has_metric_column(self, sample_df, clean_df):
        comp, _, _ = du.build_before_after_comparison(sample_df, clean_df)
        assert "Metric" in comp.columns
        assert "Before" in comp.columns
        assert "After"  in comp.columns

    def test_comparison_has_six_rows(self, sample_df, clean_df):
        comp, _, _ = du.build_before_after_comparison(sample_df, clean_df)
        assert len(comp) == 6

    def test_metrics_before_matches_raw(self, sample_df, clean_df):
        _, mb, _ = du.build_before_after_comparison(sample_df, clean_df)
        assert mb["rows"]    == len(sample_df)
        assert mb["missing"] == int(sample_df.isnull().sum().sum())

    def test_metrics_after_matches_clean(self, sample_df, clean_df):
        _, _, ma = du.build_before_after_comparison(sample_df, clean_df)
        assert ma["rows"]    == len(clean_df)
        assert ma["missing"] == int(clean_df.isnull().sum().sum())


# ─────────────────────────────────────────────────────────────────────────────
# 15. standardize_categories
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardizeCategories:

    def test_mapping_applied(self):
        df = pd.DataFrame({"dept": ["HR", "IT", "Sales"]})
        mapping = {"HR": "Human Resources", "IT": "Information Technology"}
        result = du.standardize_categories(df, "dept", mapping)
        assert "Human Resources"       in result["dept"].values
        assert "Information Technology" in result["dept"].values

    def test_unmapped_values_unchanged(self):
        df = pd.DataFrame({"dept": ["HR", "Finance"]})
        mapping = {"HR": "Human Resources"}
        result = du.standardize_categories(df, "dept", mapping)
        assert "Finance" in result["dept"].values

    def test_nan_preserved(self):
        df = pd.DataFrame({"dept": ["HR", None, "IT"]})
        mapping = {"HR": "Human Resources"}
        result = du.standardize_categories(df, "dept", mapping)
        assert pd.isna(result["dept"].iloc[1])

    def test_case_insensitive_matching(self):
        df = pd.DataFrame({"dept": ["hr", "HR", "Hr"]})
        mapping = {"hr": "Human Resources"}
        result = du.standardize_categories(df, "dept", mapping)
        assert (result["dept"] == "Human Resources").all()


# ─────────────────────────────────────────────────────────────────────────────
# 16. ai_insight_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestAiInsightAnalysis:

    def test_returns_list(self, clean_df):
        result = du.ai_insight_analysis(clean_df)
        assert isinstance(result, list)

    def test_max_4_insights_returned(self, clean_df):
        result = du.ai_insight_analysis(clean_df)
        assert len(result) <= 4

    def test_each_insight_has_required_keys(self, clean_df):
        result = du.ai_insight_analysis(clean_df)
        for insight in result:
            assert "type"        in insight
            assert "title"       in insight
            assert "description" in insight
            assert "columns"     in insight
            assert "score"       in insight

    def test_insight_types_are_valid(self, clean_df):
        valid_types = {"distribution", "correlation", "category", "trend", "proportion"}
        result = du.ai_insight_analysis(clean_df)
        for insight in result:
            assert insight["type"] in valid_types

    def test_empty_df_returns_empty_list(self):
        result = du.ai_insight_analysis(pd.DataFrame())
        assert result == []

    def test_no_duplicate_type_exceeds_2(self, clean_df):
        result = du.ai_insight_analysis(clean_df)
        from collections import Counter
        counts = Counter(i["type"] for i in result)
        for count in counts.values():
            assert count <= 2


# ─────────────────────────────────────────────────────────────────────────────
# 17. load_dataset (file size validation — no real file needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDataset:

    def test_oversized_file_raises_value_error(self):
        class FakeLargeFile:
            name = "huge.csv"
            size = 300 * 1024 * 1024   # 300 MB > 200 MB limit

        with pytest.raises(ValueError, match="200 MB"):
            du.load_dataset(FakeLargeFile())

    def test_none_input_returns_none(self):
        result = du.load_dataset(None)
        assert result == (None, None)

    def test_unsupported_extension_raises_error(self):
        class FakeFile:
            name = "data.json"
            size = 1024

        with pytest.raises(ValueError, match="Unsupported"):
            du.load_dataset(FakeFile())