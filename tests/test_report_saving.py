import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalDtype
from pandas import DatetimeTZDtype
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
from pandas.testing import assert_frame_equal

import laser_polio as lp
from laser_polio.run_sim import run_sim
from laser_polio.utils import save_sim_results


@pytest.mark.parametrize("ext", [".csv", ".h5"])
def test_save_sim_results(ext):
    # Run a tiny real simulation
    lp.root = Path()
    sim = run_sim(
        regions=["ZAMFARA"],
        start_year=2018,
        n_days=10,  # quick run
        r0=12,
        init_prev=0.01,
        background_seeding=False,
        use_pim_scalars=False,
        verbose=0,
        run=True,
        save_data=False,
        save_plots=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = Path(tmpdir) / f"test_output{ext}"

        # Save results
        df = save_sim_results(sim, outfile)

        # Basic checks
        assert outfile.exists(), f"File {outfile} was not created"
        assert not df.empty, "Saved DataFrame is empty"

        # Required columns — customize as needed
        expected_columns = {
            "timestep",
            "date",
            "node",
            "dot_name",
            "S",
            "E",
            "I",
            "R",
            "P",
            "births",
            "deaths",
            "new_exposed",
            "potentially_paralyzed",
            "new_potentially_paralyzed",
            "new_paralyzed",
        }
        assert expected_columns.issubset(df.columns), f"Missing columns: {expected_columns - set(df.columns)}"

        # Round-trip read
        if ext == ".csv":
            df2 = pd.read_csv(outfile, parse_dates=["date"])
        else:
            df2 = pd.read_hdf(outfile)

        # Equality checks
        assert len(df2) == len(df), f"Row count mismatch: {len(df2)} vs {len(df)}"
        assert set(df2.columns) == set(df.columns), "Column mismatch"

        # Spot check values — add more if needed
        assert df2["I"].sum() >= 0, "Infected count should be non-negative"
        assert df2["timestep"].max() < sim.nt, "Timestep values exceed simulation bounds"

        # Group-wise checks (e.g. ensure every timestep/node pair exists)
        expected_rows = sim.nt * len(sim.nodes)
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

        # Dot name uniqueness
        assert df["dot_name"].nunique() == len(sim.nodes), "Unexpected number of dot_name values"


def test_save_sim_results_equal():
    # Test that the CSV and H5 files are equal aside from dtypes
    lp.root = Path()
    sim = run_sim(
        regions=["BENIN"],
        start_year=2018,
        n_days=100,
        r0=12,
        init_region="ABOMEY_CALAVI",
        init_prev=200,
        verbose=0,
        run=True,
        save_data=True,
        save_plots=False,
        summary_config={
            "region_groupings": ["BENIN"],
            "grouping_level": "adm01",
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = Path(tmpdir) / "test_output"

        # Save results
        save_sim_results(sim, outfile.with_suffix(".csv"))
        save_sim_results(sim, outfile.with_suffix(".h5"))
        df_csv = pd.read_csv(outfile.with_suffix(".csv"))
        df_h5 = pd.read_hdf(outfile.with_suffix(".h5"))

        # The dtypes are different (e.g. int32 vs int64), so we need to align them first
        df_csv_fixed = align_dtypes(df_h5, df_csv)

        # verify
        assert_frame_equal(df_h5, df_csv_fixed, check_like=True, check_dtype=True, rtol=1e-6, atol=1e-12)

        # # For manual inspection
        # mismatch_counts = (df_h5.value_counts(sort=False) - df_csv_fixed.value_counts(sort=False)).loc[lambda s: s != 0]
        # print(f"mismatch_counts: {mismatch_counts}")  # non-empty => there are row-level differences
        # # Exact compare, aligning by a key column (recommended)
        # report = compare_dfs(df_h5, df_csv_fixed)
        # print("Equal?", report["equal"])
        # print(f"dtype_differences: {report['dtype_differences']}")  # where dtypes differ
        # print(f"cell_differences: {report['cell_differences'].head()}")  # shows mismatching cells (left vs right)
        # print(f"extra_rows_in_left: {report['extra_rows_in_left']}")  # rows present only on the left (after key align)
        # print(f"extra_rows_in_right: {report['extra_rows_in_right']}")  # rows present only on the right


# ----- Helper functions -----


def align_dtypes(reference: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast df's columns to match reference.dtypes, handling
    categorical, datetimes (tz-aware or naive), ints, floats, and others.
    """
    out = df.copy()
    for col, dt in reference.dtypes.items():
        if col not in out:
            continue

        # --- categorical ---
        if isinstance(dt, CategoricalDtype):
            ref_cat = reference[col].astype("category")
            cats = ref_cat.cat.categories
            out[col] = pd.Categorical(
                out[col],
                categories=cats,
                ordered=ref_cat.cat.ordered,
            )

        # --- datetimes (naive or tz-aware) ---
        elif isinstance(dt, DatetimeTZDtype) or is_datetime64_any_dtype(dt):
            # parse to datetime first
            out[col] = pd.to_datetime(out[col], errors="raise", utc=False)
            if isinstance(dt, DatetimeTZDtype):
                # ensure timezone matches reference
                if out[col].dt.tz is None:
                    out[col] = out[col].dt.tz_localize(dt.tz)
                else:
                    out[col] = out[col].dt.tz_convert(dt.tz)
            else:
                # ensure ns resolution, naive
                out[col] = out[col].astype("datetime64[ns]")

        # --- integers (avoid bool) ---
        elif is_integer_dtype(dt) and not is_bool_dtype(dt):
            out[col] = pd.to_numeric(out[col], errors="raise").astype(dt)

        # --- floats ---
        elif is_float_dtype(dt):
            out[col] = pd.to_numeric(out[col], errors="raise").astype(dt)

        # --- everything else (e.g., object/string, bool) ---
        else:
            out[col] = out[col].astype(dt)

    return out


def compare_dfs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key=None,  # column name or list of columns to align rows (recommended)
    ignore_column_order=False,  # if True, compare after sorting columns
    float_rtol=1e-6,
    float_atol=1e-12,
):
    out = {}

    L, R = left.copy(), right.copy()

    # --- normalize columns order (optional) ---
    if ignore_column_order:
        L = L.reindex(sorted(L.columns), axis=1)
        R = R.reindex(sorted(R.columns), axis=1)

    # --- align rows if a key is provided ---
    if key is not None:
        if isinstance(key, (list, tuple)):
            L = L.sort_values(list(key)).set_index(list(key))
            R = R.sort_values(list(key)).set_index(list(key))
        else:
            L = L.sort_values([key]).set_index([key])
            R = R.sort_values([key]).set_index([key])

    # --- union-align both axes ---
    L, R = L.align(R, join="outer", axis=1)
    L, R = L.align(R, join="outer", axis=0)

    # --- structure differences ---
    out["extra_columns_in_left"] = sorted(set(L.columns) - set(R.columns))
    out["extra_columns_in_right"] = sorted(set(R.columns) - set(L.columns))
    out["extra_rows_in_left"] = L.index.difference(R.index)
    out["extra_rows_in_right"] = R.index.difference(L.index)

    # --- dtype differences (after alignment) ---
    dtypes_left = pd.Series(L.dtypes, name="left")
    dtypes_right = pd.Series(R.dtypes, name="right")
    dtypes = pd.concat([dtypes_left, dtypes_right], axis=1)
    out["dtype_differences"] = dtypes[dtypes["left"] != dtypes["right"]]

    # --- cell-by-cell differences with float tolerance ---
    # Build a boolean mask of "not equal", allowing for NaN==NaN and float tolerance.
    mask = pd.DataFrame(False, index=L.index, columns=L.columns)

    num_cols = L.select_dtypes(include="number").columns.intersection(R.select_dtypes(include="number").columns)
    if len(num_cols):
        a = L[num_cols].astype("float64")
        b = R[num_cols].astype("float64")
        ne_num = ~np.isclose(a, b, rtol=float_rtol, atol=float_atol, equal_nan=True)
        mask.loc[:, num_cols] = ne_num

    other_cols = L.columns.difference(num_cols)
    if len(other_cols):
        a = L[other_cols]
        b = R[other_cols]
        ne_other = (a != b) & ~(a.isna() & b.isna())
        mask.loc[:, other_cols] = ne_other

    # Compose a compact diff table: only cells that differ
    diff_left = L.where(mask)
    diff_right = R.where(mask)
    out["cell_differences"] = pd.concat({"left": diff_left, "right": diff_right}, axis=1).dropna(how="all")

    # Overall verdict
    out["equal"] = (
        not mask.any().any()
        and not out["extra_columns_in_left"]
        and not out["extra_columns_in_right"]
        and len(out["extra_rows_in_left"]) == 0
        and len(out["extra_rows_in_right"]) == 0
    )

    return out


if __name__ == "__main__":
    test_save_sim_results_equal()
    print("All save_sim_results tests passed!")
