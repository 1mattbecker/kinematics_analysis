# TODO — kinematics_analysis

Deferred work items. Newest first. Dates are YYYY-MM-DD.

---

## Upgrade the Code Ocean environment off Python 3.9

_Logged 2026-09-07. Not scheduled — do this deliberately, on a branch, not mid-analysis._

### Why

`environment/Dockerfile` builds from
`codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04`. Python 3.9 reached
end-of-life in October 2025. That single pin is the root cause of several unrelated-looking
workarounds scattered through this repo:

1. **`rachel-analysis-utils` needs `--ignore-requires-python`.** The package declares
   `requires-python>=3.10`. The Dockerfile installs it in an isolated `RUN` layer with
   `--ignore-requires-python` specifically so that flag does not leak into the main pip block
   (where it would let pip resolve a too-new numpy and break the build). See the comment above
   that layer.
2. **`rachel_analysis_utils.analysis_utils` cannot be imported at all** on 3.9 — it has a
   nested-quote f-string at `analysis_utils.py:294` (`f'{x.split('_dff')...}'`), which is 3.12+
   syntax. Because of this, `fip_00_explore.ipynb` reimplements `enrich_df_trials`'s
   `num_reward_past` locally as `enrich_streaks` (unit-tested to match the package). That
   reimplementation exists *only* to dodge the import and can be deleted after the upgrade.
3. **The repo-wide "Python 3.9 compatible syntax only" rule** in `CLAUDE.md` — no `|` type
   unions, no structural pattern matching, no walrus in complex contexts — exists solely to
   match this image.

Upgrading the base image removes all three.

### Risk (why this is not a quick change)

The main pip block pins a mutually-consistent stack that was resolved against 3.9:

```
spikeinterface[full]==0.100.0   scipy==1.13.0        pynwb==3.0.0
hdmf-zarr==0.11.0               zarr==2.18.2         statsmodels==0.14.2
pyarrow==21.0.0                 seaborn==0.13.2      opencv_python==4.11.0.86
moviepy==1.0.3                  scikit-image==0.24.0 aind-ephys-utils==0.0.15
```

Several of these will need version bumps on a newer interpreter, and `spikeinterface[full]`
in particular pulls a large transitive tree. ~41 notebooks depend on this environment, with
the `eph_*` and `kin_*` families being the most exposed (they use spikeinterface, pynwb and
the ephys stack directly). A bad upgrade silently breaks working analyses.

### Suggested approach

- Do it on a dedicated branch, and use Code Ocean's environment versioning so the current
  working environment stays recoverable.
- Bump the base image to the Code Ocean `python3.11` JupyterLab image (matches
  `aind-motion-energy`, which already targets 3.11).
- Relax the pins in the main block to `>=` where possible, rebuild, then re-pin to whatever
  actually resolves — do not hand-guess versions.
- Drop `--ignore-requires-python` from the `rachel-analysis-utils` layer and fold it back
  into the main pip block.
- Verification pass, in rough order of blast radius: `eph_00`, `kin_00`, then `fip_00`.
  Confirm `import rachel_analysis_utils.analysis_utils` succeeds, then delete the local
  `enrich_streaks` reimplementation in `fip_00_explore.ipynb` and switch back to
  `enrich_df_trials`.
- Remove the Python 3.9 syntax rule from `CLAUDE.md` once the image is live.

### Related cleanups unblocked by this

- `fip_00_explore.ipynb` cell 3 pip-installs `fastparquet` at runtime because the image does
  not ship it (`rachel_analysis_utils.load_nwb_list` hardcodes `engine="fastparquet"`).
  That belongs in the Dockerfile pip block regardless of the Python version — it does not
  need to wait for this upgrade.
