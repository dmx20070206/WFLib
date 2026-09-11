#!/usr/bin/env python3
"""Read every stored trace, audit signed timestamps, and extract trace-level features.

Zero is padding (confirmed against the repository data processor); signs are kept
as positive/negative, since physical direction is not documented in the NPZ.
Time units are treated as seconds following the upstream processor conventions.
"""
import argparse
import hashlib
import json
import platform
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = [14, 30, 90, 150, 270]
NAMES = [
    "n_packets", "duration", "positive_fraction", "direction_mean", "direction_variance",
    "switch_rate", "burst_count", "burst_mean", "burst_std", "burst_p90", "burst_max",
    "positive_burst_mean", "negative_burst_mean", "iat_mean", "iat_variance", "iat_median",
    "iat_p90", "iat_p99", "iat_zero_fraction", "iat_cv", "first_time", "packet_rate",
    "positive_iat_mean", "negative_iat_mean", "early_positive_fraction", "early_switch_rate",
    "time_q25", "time_q50", "time_q75",
]
# Do not give deterministic/near-duplicate coordinates extra weight in distances.
CORE = ["n_packets", "duration", "positive_fraction", "switch_rate", "burst_std",
        "burst_p90", "positive_burst_mean", "negative_burst_mean", "iat_mean",
        "iat_variance", "iat_median", "iat_p90", "iat_p99", "iat_zero_fraction",
        "early_positive_fraction", "early_switch_rate"]
RATIOS = {"positive_fraction", "direction_mean", "direction_variance", "switch_rate",
          "iat_zero_fraction", "early_positive_fraction", "early_switch_rate"}
TIME = {"duration", "iat_mean", "iat_median", "iat_p90", "iat_p99", "first_time",
        "positive_iat_mean", "negative_iat_mean", "time_q25", "time_q50", "time_q75"}


def transform(frame, names=CORE):
    """Fixed unit conversion + log; a scaler must subsequently fit SOURCE TRAIN only."""
    values = frame[names].to_numpy(dtype=np.float64, copy=True)
    for j, name in enumerate(names):
        if name not in RATIOS:
            factor = 1e6 if name == "iat_variance" else 1000.0 if name in TIME else 1.0
            values[:, j] = np.log1p(np.maximum(values[:, j], 0) * factor)
    return values


def features(row):
    mask = row != 0
    v = row[mask]
    if len(v) < 2 or not np.isfinite(v).all():
        return None, {"invalid_trace": 1}
    t, s = np.abs(v), np.sign(v)
    dt = np.diff(t)
    negative = dt < 0
    corrected_t = np.maximum.accumulate(t)
    audit = {
        "negative_iat_count": int(negative.sum()),
        "negative_iat_trace": int(negative.any()),
        "internal_zero_trace": int(np.any(~mask[:np.flatnonzero(mask)[-1] + 1])),
        "full_length_trace": int(mask.all()),
        "zero_iat_count": int((dt == 0).sum()),
        "iat_count": len(dt),
        "n_packets": len(v),
        "first_positive_trace": int(s[0] > 0),
        "corrected_timestamp_count": int(np.sum(corrected_t != t)),
        "negative_iat_abs_sum": float(-dt[negative].sum()),
    }
    # Monotone envelope preserves stored packet order/direction and uses the
    # maximum observed timestamp as the endpoint (not necessarily the last raw t).
    # Clipping differences independently would inflate their total duration.
    t = corrected_t
    dt = np.diff(t)
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    bursts = np.diff(np.r_[starts, len(v)])
    bsign = s[starts]
    pos_b, neg_b = bursts[bsign > 0], bursts[bsign < 0]
    early = s[:min(500, len(s))]
    pos_t, neg_t = t[s > 0], t[s < 0]
    dur = t[-1] - t[0]
    values = [len(v), dur, (s > 0).mean(), s.mean(), s.var(),
              (len(bursts) - 1) / (len(v) - 1), len(bursts), bursts.mean(), bursts.std(),
              np.quantile(bursts, .9), bursts.max(),
              pos_b.mean() if len(pos_b) else 0, neg_b.mean() if len(neg_b) else 0,
              dt.mean(), dt.var(), np.median(dt), np.quantile(dt, .9), np.quantile(dt, .99),
              (dt == 0).mean(), dt.std() / max(dt.mean(), 1e-12), t[0],
              len(v) / max(dur, 1e-12),
              np.diff(pos_t).mean() if len(pos_t) > 1 else 0,
              np.diff(neg_t).mean() if len(neg_t) > 1 else 0,
              (early > 0).mean(), np.mean(np.diff(early) != 0),
              *np.quantile(t - t[0], [.25, .5, .75])]
    return values, audit


def extract_day(args):
    day, data_dir, out_dir = args
    path = Path(data_dir) / f"day{day}.npz"
    out = Path(out_dir)
    rows, prefix_rows, digests, audits, negative_values = [], [], [], {}, []
    source_hash = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        with z.open("y.npy") as f:
            y = np.lib.format.read_array(f, allow_pickle=False)
        assert np.isfinite(y).all() and np.equal(y, y.astype(int)).all()
        with z.open("X.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            assert len(shape) == 2 and not fortran and shape[0] == len(y)
            row_bytes = shape[1] * dtype.itemsize
            for start in range(0, shape[0], 256):
                buf = f.read(min(256, shape[0] - start) * row_bytes)
                source_hash.update(buf)
                block = np.frombuffer(buf, dtype=dtype).reshape(-1, shape[1])
                for offset, row in enumerate(block):
                    rid = start + offset
                    vals, audit = features(row)
                    original_dt = np.diff(np.abs(row[row != 0]))
                    negative_values.extend((-original_dt[original_dt < 0]).tolist())
                    for key, val in audit.items():
                        audits[key] = audits.get(key, 0) + val
                    digest = hashlib.sha256(row.tobytes()).hexdigest()
                    digests.append((day, rid, int(y[rid]), digest))
                    if vals is not None:
                        rows.append([day, rid, int(y[rid]), audit["full_length_trace"], digest] + vals)
                    pvals, _ = features(row[:1000])
                    if pvals is not None:
                        prefix_rows.append([day, rid, int(y[rid]), int(np.all(row[:1000] != 0)), digest] + pvals)
    columns = ["day", "row_id", "label", "at_cap", "trace_hash"] + NAMES
    pd.DataFrame(rows, columns=columns).to_csv(out / f"features_day{day}.csv", index=False)
    pd.DataFrame(prefix_rows, columns=columns).to_csv(out / f"prefix1000_day{day}.csv", index=False)
    pd.DataFrame(digests, columns=["day", "row_id", "label", "trace_hash"]).to_csv(out / f"hashes_day{day}.csv", index=False)
    labels, counts = np.unique(y, return_counts=True)
    audits.update(day=day, file=str(path), shape=list(shape), dtype=str(dtype),
                  bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                  raw_X_payload_sha256=source_hash.hexdigest(), classes=len(labels),
                  class_counts={str(int(k)): int(v) for k, v in zip(labels, counts)},
                  valid_feature_rows=len(rows), valid_prefix_rows=len(prefix_rows))
    audits["negative_iat_abs_quantiles_seconds"] = dict(zip(
        ["min", "p50", "p90", "p99", "max"],
        np.quantile(negative_values, [0, .5, .9, .99, 1]).tolist())) if negative_values else {}
    (out / f"audit_day{day}.json").write_text(json.dumps(audits, indent=2))
    print(f"Day {day}: {len(rows)} traces extracted; audit={audits}", flush=True)
    return audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/diaomingxuan/Traffic-Drift/datasets/TemporalDrift")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(extract_day, [(d, args.data, args.out) for d in DAYS]))
    hashes = pd.concat([pd.read_csv(out / f"hashes_day{d}.csv") for d in DAYS], ignore_index=True)
    dup = hashes[hashes.trace_hash.duplicated(keep=False)]
    dup.to_csv(out / "duplicate_traces.csv", index=False)
    manifest = dict(days=DAYS, features=NAMES, core_features=CORE, audits=audits,
                    duplicate_rows=int(len(dup)), duplicate_groups=int(dup.trace_hash.nunique()),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
