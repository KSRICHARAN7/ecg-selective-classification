# Release verification — October 15, 2026

Performed with the Linux/Python 3.12 frozen scientific environment. These are internal automated checks, not independent human audit or fresh external validation.

- Original scientific suite: **64 tests passed** using the released code and the provider condition dictionary.
- Saved aggregate evidence: **1,862 metric rows** checked for count/error arithmetic; all **16 primary method-by-condition cells** agree with the frozen primary-result record.
- Original primary plotting script: all **five PNG/SVG figure pairs** regenerated without source ECG data. SVGs are included; PNGs can be generated locally.
- Public fixed-model replay: all **140 matched-coverage rows** and **20 fixed-policy rows** for clean + four withheld-artifact conditions agree with saved evidence within the adapter’s stated tolerance (1e-10).
- Paired participant bootstrap: **2,000 resamples**, zero undefined resamples; the primary 95% interval reproduced as **[−0.02703376153738879, 0.011410111964674544]** in error-rate units, or **[−2.703376153738879, 1.1410111964674545]** percentage points.
- Replay outputs are aggregate-only. [Machine-readable replay comparison](verification/fixed_test_comparison.json).

The data-dependent replay used existing locally verified copies of the original provider files; a second network download was not performed. Dependency installation from an empty online environment was not retested. The final isolated source-checkout check and GitHub Actions evidence check are recorded in the repository history/workflow, not represented as a fresh independent scientific replication.

The adapter does not rerun training, the original cross-role waveform-identity audit, the external followup, or exploratory ablations. Its successful result preserves the original negative primary conclusion.
