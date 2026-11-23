"""Per-mouse accounting of FIP sessions with usable motion energy, and why cameras are excluded.

Reads only the session list and the video screen in ``inputs/`` (no data assets), so it runs
anywhere:

* ``inputs/me_sessions_fip_curated.csv``: the 301 FIP sessions, whether each has a curated
  lateral-NAc DA and PL LC-axon fiber on the same hemisphere (``has_same_side_pair``), and whether
  it is in the 97-session analysis set of ``fip_10``–``fip_17`` (``used_in_fip05_07``: one
  hemisphere per mouse, at least 3 sessions; ``fu.choose_side``).
* ``inputs/video_screen_fip.csv``: per session and camera, the video timing and image-quality
  verdicts (``use``, ``reason``). A camera is usable when it passes both; every session has an
  ME run (``inputs/me_assets_fip.csv``), so usable video means usable ME.

Writes ``docs/me_session_accounting.md`` (both tables, with the category definitions) and
prints the tables. Run from ``code/``::

    python me_session_accounting.py
"""
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_CSV = os.path.join(REPO_ROOT, "inputs", "me_sessions_fip_curated.csv")
SCREEN_CSV = os.path.join(REPO_ROOT, "inputs", "video_screen_fip.csv")
OUT_MD = os.path.join(REPO_ROOT, "docs", "me_session_accounting.md")

CAMERAS = {"BottomCamera": "Bottom", "SideCameraRight": "Side"}

#: Screen ``reason`` → QC category shown in the table.
CATEGORY = {
    "timing: exclude: harp_evenly_spaced": "clock step",
    "timing: exclude: trigger_log_count": "trigger count",
    "timing: exclude: clock_rates_agree": "clock rate",
    "timing: exclude: unreadable": "unreadable",
    "quality: exclude: pct_clipped_high median <= 3.75": "overexposed",
    "quality: exclude: sharpness_dev <= 0.45": "blurred",
    "quality: exclude: similarity p5 < 0.998": "image changed",
}

DEFINITIONS = """\
Timing QC (from the video-analysis library's timing check):

- **clock step**: the Harp frame times jump (`harp_evenly_spaced`). The cameras share the Harp
  clock, so a clock step fails both cameras of a session.
- **trigger count**: the camera's trigger log and its frame numbers disagree (`trigger_log_count`).
- **clock rate**: the camera and Harp clocks run at different rates (`clock_rates_agree`).
- **unreadable**: the video could not be read.

Image-quality QC (median over reference frames):

- **overexposed**: too many pixels clipped at the top of the range (`pct_clipped_high` > 3.75 %).
- **blurred**: sharpness far below the camera's usual (`sharpness_dev` ≤ 0.45).
- **image changed**: the view changes during the session (`similarity_p5` < 0.998).
"""


def session_table() -> pd.DataFrame:
    """One row per session: photometry flags and whether each camera's ME is usable.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``raw_session``; columns ``subject``, ``ses_idx``, ``asset``,
        ``has_same_side_pair``, ``used_in_fip05_07``, ``bottom`` / ``side`` (usable ME) and
        ``bottom_reason`` / ``side_reason`` (QC category of an excluded camera, else NaN).
    """
    sessions = pd.read_csv(SESSIONS_CSV, dtype={"subject": str}).set_index("raw_session")
    screen = pd.read_csv(SCREEN_CSV)
    screen["category"] = screen["reason"].map(CATEGORY)
    unknown = set(screen["reason"].dropna()) - set(CATEGORY)
    assert not unknown, f"screen reasons without a category: {unknown}"
    for camera, name in CAMERAS.items():
        cam = screen[screen["camera"] == camera].set_index("session")
        sessions[name.lower()] = cam["use"].reindex(sessions.index).fillna(False).astype(bool)
        sessions[f"{name.lower()}_reason"] = cam["category"].reindex(sessions.index)
    return sessions


def usable_by_subject(sessions: pd.DataFrame) -> pd.DataFrame:
    """Sessions per mouse with usable ME, alone and with the photometry selections."""
    g = sessions.assign(
        both=sessions["bottom"] & sessions["side"],
        pair_bottom=sessions["has_same_side_pair"] & sessions["bottom"],
        analysis_bottom=sessions["used_in_fip05_07"] & sessions["bottom"],
    ).groupby("subject")
    table = pd.DataFrame({
        "Sessions": g.size(),
        "Bottom ME usable": g["bottom"].sum(),
        "Side ME usable": g["side"].sum(),
        "Both cameras usable": g["both"].sum(),
        "DA + LC (same side)": g["has_same_side_pair"].sum(),
        "DA + LC, bottom ME": g["pair_bottom"].sum(),
        "Analysis set": g["used_in_fip05_07"].sum(),
        "Analysis set, bottom ME": g["analysis_bottom"].sum(),
    }).astype(int)
    table.loc["ALL"] = table.sum()
    table.index.name = "Subject"
    return table


def exclusions_by_subject(sessions: pd.DataFrame) -> pd.DataFrame:
    """Excluded cameras per mouse, by camera and QC category (categories with none are left out)."""
    columns = {}
    for name in CAMERAS.values():
        reason = sessions[f"{name.lower()}_reason"]
        for category in dict.fromkeys(CATEGORY.values()):
            counts = (reason == category).groupby(sessions["subject"]).sum()
            if counts.any():
                columns[f"{name}: {category}"] = counts
    table = pd.DataFrame(columns).astype(int)
    table.loc["ALL"] = table.sum()
    table.index.name = "Subject"
    return table


def _markdown(table: pd.DataFrame) -> str:
    """Pipe table, the ``ALL`` row in bold, zeros left blank in the exclusion table."""
    header = "| " + " | ".join([table.index.name] + list(table.columns)) + " |"
    rule = "|---|" + "---:|" * len(table.columns)
    lines = [header, rule]
    for subject, row in table.iterrows():
        cells = [str(v) for v in row]
        if subject == "ALL":
            lines.append("| **ALL** | " + " | ".join(f"**{c}**" for c in cells) + " |")
        else:
            lines.append(f"| {subject} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    sessions = session_table()
    usable = usable_by_subject(sessions)
    excluded = exclusions_by_subject(sessions)
    n_bottom = int(usable.loc["ALL", "Bottom ME usable"])
    n_side = int(usable.loc["ALL", "Side ME usable"])
    text = f"""\
# FIP sessions with usable motion energy

Made by `code/me_session_accounting.py` from `inputs/me_sessions_fip_curated.csv` and
`inputs/video_screen_fip.csv`; rerun it after either changes.

{len(sessions)} FIP sessions from {sessions['subject'].nunique()} mice (the `DANE_3channels_curated`
and `DANE_4channels_curated` assets). Every session has an ME run, so a camera's ME is usable when
its video passes the timing and image-quality screen: {n_bottom} bottom cameras and {n_side} side
cameras.

## Usable ME, sessions per mouse

{_markdown(usable)}

- **DA + LC (same side)**: a curated lateral-NAc DA fiber (`latNAcc(X)-DA`) and a curated PL
  LC-axon fiber (`PL(X)-LCAxonCa`) on the same hemisphere (`fu.inventory_pairs`).
- **Analysis set**: the sessions `fip_10`–`fip_17` use: one hemisphere per mouse (the side with
  more sessions) and at least 3 sessions on it (`fu.choose_side`). With bottom-camera ME, these
  are the movement sessions of the `fip_*` notebooks (`fu.add_motion_energy`).

## Excluded cameras, by QC category

{_markdown(excluded).replace("| 0 ", "|  ")}

{DEFINITIONS}"""
    with open(OUT_MD, "w") as fh:
        fh.write(text)
    pd.set_option("display.width", 250)
    print(usable.to_string(), "\n")
    print(excluded.to_string())
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
