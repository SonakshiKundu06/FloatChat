# src/ingest/argo_loader.py
import xarray as xr
import pandas as pd
from pathlib import Path

def load_netcdf_to_df(nc_path: str) -> pd.DataFrame:
    """
    Load a single ARGO NetCDF file into a pandas DataFrame.
    Always creates a 'time' and 'year' column, falling back to metadata or folder.
    """
    ds = xr.open_dataset(nc_path, decode_times=False)
    df = ds.to_dataframe().reset_index()

    # -------- Try to find a usable time column --------
    time_col = None
    if "time" in df.columns:
        time_col = "time"
    elif "TIME" in df.columns:
        time_col = "TIME"
    elif "JULD" in df.columns:
        time_col = "JULD"

    if time_col:
        if time_col.lower() == "juld":
            # Convert JULD (days since 1950-01-01)
            df["time"] = pd.to_datetime("1950-01-01") + pd.to_timedelta(df[time_col], unit="D")
        else:
            df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        # No time variable in dataframe → check global attributes
        start_time = None
        if "time_coverage_start" in ds.attrs:
            start_time = pd.to_datetime(ds.attrs["time_coverage_start"], errors="coerce")
        elif "date_creation" in ds.attrs:
            start_time = pd.to_datetime(ds.attrs["date_creation"], errors="coerce", format="%Y%m%d%H%M%S")

        if start_time is not None:
            df["time"] = start_time
        else:
            # Fallback: no time info → NaT
            df["time"] = pd.NaT

    # -------- Derive year --------
    if df["time"].notna().any():
        df["year"] = df["time"].dt.year.fillna(-1).astype(int)
    else:
        # Final fallback → use folder name as year
        try:
            year_from_folder = int(Path(nc_path).parts[-2])
        except Exception:
            year_from_folder = -1
        df["year"] = year_from_folder

    # -------- Keep relevant columns --------
    keep = [
        "platform_number", "cycle_number", "latitude", "longitude",
        "pres", "temp", "psal",
        "pres_adjusted", "temp_adjusted", "psal_adjusted",
        "time", "year"
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df


def load_all_years(raw_root="./data/raw") -> pd.DataFrame:
    dfs = []
    raw_root = Path(raw_root)

    for year_dir in raw_root.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for nc_file in year_dir.glob("*.nc"):
                print(f"📂 Reading {nc_file}")
                try:
                    df = load_netcdf_to_df(nc_file)
                    df["source_file"] = str(nc_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Failed to read {nc_file}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
