# save as extract_markers.py  (Python 3.9+)
import sys, pandas as pd, pathlib as pl

def extract_markers(raw_csv: pl.Path):
    df = pd.read_csv(raw_csv, sep=r"\s+|\t|,", engine="python", header=None)
    ts_col = df.columns[-2]   # Unix timestamp column
    mr_col = df.columns[-1]   # Marker column
    events = df[df[mr_col] != 0][[ts_col, mr_col]].dropna()
    events.columns = ["t_unix", "marker"]
    t0 = events["t_unix"].iloc[0] if len(events) else 0.0
    events["t_rel_s"] = events["t_unix"] - t0
    out = raw_csv.with_suffix(".markers.csv")
    events.to_csv(out, index=False)
    print(f"wrote {out}  |  {len(events)} events, markers={sorted(events.marker.unique())}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        extract_markers(pl.Path(arg))
