"""
check_kin02_kin05_trials.py — old vs new pooled movements, trial by trial (kinematics-core migration,
TODO step 6). One-off check; run on Code Ocean from code/: ``python check_kin02_kin05_trials.py``.

Old: ``all_tongue_movements_04022026`` (library < 0.5.0). New: ``all_tongue_movements_v053``.
Movements match one to one on start/end time (verify_kinematics_v053 §4), so what can differ per trial
is which trial a movement is filed under, the cue-response movement number k, and lick_latency.

1. kin_02: trials with a cue-response movement (kin_02's selection) in both, old only, new only; for
   shared trials, whether k and lick_latency agree; kin_02's numbers on shared trials only.
2. Why trials are in one table only.
3. kin_05: movements per trial; the old trials above 124 movements; r(lick, non-lick movements per trial)
   with and without them.
"""
import numpy as np
import pandas as pd
from scipy import stats

from kin_utils import POOLED_MOVEMENTS, POOLED_MOVEMENTS_OLD, pooled_path

KEY = ["session", "trial"]
MAX_K = 4

old = pd.read_parquet(pooled_path(POOLED_MOVEMENTS_OLD)).rename(columns={
    "movement_latency_from_go": "latency_from_go", "movement_number_in_trial": "number_in_trial"})
new = pd.read_parquet(pooled_path(POOLED_MOVEMENTS))
old["has_lick"] = old["has_lick"].isin([True, 1, "True"])
new["has_lick"] = new["n_licks"] > 0
shared_sessions = sorted(set(old["session"]) & set(new["session"]))
print(f"sessions: old {old['session'].nunique()}, new {new['session'].nunique()}, both {len(shared_sessions)}")
print("old only:", sorted(set(old["session"]) - set(new["session"])))
print("new only:", sorted(set(new["session"]) - set(old["session"])))


def cue_response_trials(df):
    """kin_02 §3: one row per trial, the k-th movement with 0.02 < latency_from_go < 2 s."""
    v = df.dropna(subset=["cue_response_movement_number", "latency_from_go"])
    v = v[(v["latency_from_go"] > 0.02) & (v["latency_from_go"] < 2.0)]
    v = v[v["number_in_trial"] == v["cue_response_movement_number"]]
    v = v.assign(k=v["cue_response_movement_number"].astype(int))
    return v.drop_duplicates(KEY)[KEY + ["k", "latency_from_go", "lick_latency", "start_time"]]


def kin02_numbers(tr, label):
    """kin_02's counts by k, lick-latency SD at k=1 and session-mean lick latency by k."""
    t = tr.dropna(subset=["lick_latency"])
    sess = t[t["k"] <= MAX_K].groupby(["session", "k"])["lick_latency"].mean().groupby("k").mean()
    print(f"  {label:<28} n by k {t['k'].value_counts().sort_index().head(MAX_K).to_dict()}  "
          f"SD(k=1) {t.loc[t['k'] == 1, 'lick_latency'].std(ddof=0):.4f}  "
          f"session mean by k {sess.round(4).to_dict()}")


# ---------------------------------------------------------------- 1. kin_02
print("\n=== 1. kin_02: trials with a cue-response movement")
tr_old, tr_new = cue_response_trials(old), cue_response_trials(new)
j = tr_old.merge(tr_new, on=KEY, how="outer", suffixes=("_old", "_new"), indicator=True)
j = j[j["session"].isin(shared_sessions)]
print(j["_merge"].value_counts().to_string())
b = j[j["_merge"] == "both"]


def same(a, c, tol=1e-3):
    return ((a - c).abs() < tol) | (a.isna() & c.isna())


print(f"\nshared trials {len(b):,}:")
print(f"  same k               {(b['k_old'] == b['k_new']).mean():.4f}")
print(f"  same start_time      {same(b['start_time_old'], b['start_time_new']).mean():.4f}")
print(f"  same latency_from_go {same(b['latency_from_go_old'], b['latency_from_go_new']).mean():.4f}")
print(f"  same lick_latency    {same(b['lick_latency_old'], b['lick_latency_new']).mean():.4f}")
diff = b[~same(b["lick_latency_old"], b["lick_latency_new"])]
if len(diff):
    d = diff["lick_latency_new"] - diff["lick_latency_old"]
    print(f"  lick_latency differs in {len(diff):,}: new - old median {d.median():+.4f} s, "
          f"NaN old {diff['lick_latency_old'].isna().sum()}, NaN new {diff['lick_latency_new'].isna().sum()}")
    print(diff[KEY + ["k_old", "k_new", "lick_latency_old", "lick_latency_new"]].head(8).round(4).to_string(index=False))
kdiff = b[b["k_old"] != b["k_new"]]
if len(kdiff):
    print(f"  k differs in {len(kdiff):,}:")
    print(pd.crosstab(kdiff["k_old"], kdiff["k_new"]).to_string())

print("\nkin_02 numbers:")
kin02_numbers(tr_old, "old, all sessions")
kin02_numbers(tr_new, "new, all sessions")
kin02_numbers(tr_old[tr_old["session"].isin(shared_sessions)], "old, shared sessions")
kin02_numbers(tr_new[tr_new["session"].isin(shared_sessions)], "new, shared sessions")
shared_keys = b[KEY]
kin02_numbers(tr_old.merge(shared_keys, on=KEY), "old, shared trials")
kin02_numbers(tr_new.merge(shared_keys, on=KEY), "new, shared trials")
for side, sel in [("old only", "left_only"), ("new only", "right_only")]:
    s = j[j["_merge"] == sel]
    col = "lick_latency_old" if side == "old only" else "lick_latency_new"
    kcol = "k_old" if side == "old only" else "k_new"
    print(f"  {side:<28} n {len(s):,}  k {s[kcol].value_counts().sort_index().head(MAX_K).to_dict()}  "
          f"lick_latency median {s[col].median():.4f}")


# ---------------------------------------------------------------- 2. why one side only
def explain(keys, other):
    """For (session, trial) pairs missing from ``other``'s kin_02 selection, what ``other`` has."""
    rows = []
    g = other.groupby(KEY)
    for s, t in keys[KEY].itertuples(index=False):
        if (s, t) not in g.groups:
            rows.append("no movements filed under this trial")
            continue
        m = g.get_group((s, t))
        crmn = m["cue_response_movement_number"].dropna()
        if crmn.empty:
            rows.append("trial has no cue-response movement number")
        elif not (m["number_in_trial"] == crmn.iloc[0]).any():
            rows.append("no movement with number_in_trial == k")
        else:
            lat = m.loc[m["number_in_trial"] == crmn.iloc[0], "latency_from_go"].iloc[0]
            rows.append("latency outside 0.02-2 s" if np.isfinite(lat) else "latency missing")
    return pd.Series(rows, dtype=object).value_counts()


print("\n=== 2. Why a trial is in one table only (shared sessions)")
print("new-only trials, what the old table has:")
print(explain(j[j["_merge"] == "right_only"], old).to_string())
print("old-only trials, what the new table has:")
print(explain(j[j["_merge"] == "left_only"], new).to_string())
print("\nnew-only trials by session (top 10):")
print(j[j["_merge"] == "right_only"].groupby("session").size().sort_values(ascending=False).head(10).to_string())


# ---------------------------------------------------------------- 3. kin_05
def per_trial(df):
    """kin_05 §7: movements per (session, trial), split by lick."""
    m = df.dropna(subset=["trial"])
    out = m.groupby(KEY).agg(total=("has_lick", "size"), lick=("has_lick", "sum"))
    out["nonlick"] = out["total"] - out["lick"]
    return out


print("\n=== 3. kin_05: movements per trial")
pt_old, pt_new = per_trial(old), per_trial(new)
last_trial = old.groupby("session")["trial"].max()
big = pt_old[pt_old["total"] > pt_new["total"].max()].reset_index()
big["last_trial_of_session"] = big["trial"].eq(big["session"].map(last_trial))
first_last = old.dropna(subset=["trial"]).groupby(KEY)["start_time"].agg(["min", "max"])
big = big.join(first_last, on=KEY).assign(span_s=lambda x: x["max"] - x["min"])
print(f"old trials with more movements than the new maximum ({pt_new['total'].max()}): {len(big)}, "
      f"{big['total'].sum():,} movements; last trial of its session: {big['last_trial_of_session'].sum()}")
print(big.sort_values("total", ascending=False).head(10)[KEY + ["total", "lick", "nonlick", "span_s",
      "last_trial_of_session"]].round(1).to_string(index=False))
if len(big):
    m = new.merge(big[KEY], on=KEY)
    print(f"new table, same trials: {len(m):,} movements; the old ones, where they are now:")
    oldm = old.merge(big[KEY], on=KEY)[["session", "start_time", "end_time"]].round(4)
    newk = new.assign(start_time=new["start_time"].round(4), end_time=new["end_time"].round(4))
    where = oldm.merge(newk[["session", "start_time", "end_time", "trial", "spans_go_cue"]],
                       on=["session", "start_time", "end_time"], how="left")
    print(f"  no trial {where['trial'].isna().sum():,} (spanning a go cue {where['spans_go_cue'].eq(True).sum():,}), "
          f"with a trial {where['trial'].notna().sum():,}")


def r(pt, label):
    rr = stats.pearsonr(pt["lick"], pt["nonlick"])[0]
    rs = stats.spearmanr(pt["lick"], pt["nonlick"])[0]
    print(f"  {label:<40} n {len(pt):>6,}  Pearson {rr:+.3f}  Spearman {rs:+.3f}")


print("\nr(lick, non-lick movements per trial):")
r(pt_old, "old")
r(pt_new, "new")
r(pt_old[pt_old["total"] <= pt_new["total"].max()], f"old, trials <= {pt_new['total'].max()} movements")
r(pt_old[pt_old["total"] <= 40], "old, trials <= 40 movements")
r(pt_new[pt_new["total"] <= 40], "new, trials <= 40 movements")
both = pt_old.join(pt_new, lsuffix="_old", rsuffix="_new", how="inner")
print(f"\nshared trials {len(both):,}: same total {(both['total_old'] == both['total_new']).mean():.4f}, "
      f"same lick {(both['lick_old'] == both['lick_new']).mean():.4f}")
d = both["total_new"] - both["total_old"]
print("total new - old:", d.describe().round(2).to_dict())
