# Source data and attribution

No source ECG waveforms, participant-level cohort manifests, individual prediction arrays or private research ZIPs are distributed here. Obtain data directly from the providers and follow their terms.

## Primary ECG source

PTB-XL, version **1.0.3**, PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/

Version DOI: https://doi.org/10.13026/kfzx-aw45

License: Creative Commons Attribution 4.0 International. The provider page supplies download instructions. The replay uses `ptbxl_database.csv`, `scp_statements.csv`, `SHA256SUMS.txt`, and selected 100 Hz records under `records100/`. Retain the provider license and attribution when obtaining or reusing the data.

Dataset paper: Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6

The fixed model weights and analysis outputs are new derivatives, not unmodified provider data. The metadata file hashes are in `research/config/task_definition.json`; the official inventory digest is pinned in `SOURCE_DATA.json`. No demographic fields are inputs to the classifier or error rankers.

## Recorded noise

MIT-BIH Noise Stress Test Database, version **1.0.0**, PhysioNet: https://physionet.org/content/nstdb/1.0.0/

DOI: https://doi.org/10.13026/C2HS3T

License: Open Data Commons Attribution License v1.0. The repository includes the small publisher checksum inventory, not the noise waveforms. The fixed-model replay uses `em.hea` and `em.dat`; original development code uses `bw` and `ma`.

Citation: Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. Computers in Cardiology 1984;11:381–384. Moody GB and Mark RG are the PhysioNet noise database creators.

The `em` source is recorded artifact. It is not Gaussian white noise, and lower dB means more noise relative to signal. The benchmark adds rescaled recorded noise to the extracted ECG lead.

## External followup source

Zheng J, Guo H, Chu H. A large scale 12-lead electrocardiogram database for arrhythmia study, version **1.0.0**: https://physionet.org/content/ecg-arrhythmia/1.0.0/ · DOI https://doi.org/10.13026/wgex-er52 · CC BY 4.0.

Also cite Zheng et al. (2020), Optimal Multi-Stage Arrhythmia Classification Approach, Scientific Reports 10, 2898. https://doi.org/10.1038/s41598-020-59821-7

The provider’s non-participant condition-name dictionary and license are included to support the synthetic label-rule tests. Only aggregate external results and original analysis source are otherwise included; the public primary replay does not download or rerun this followup. See the supplement for its fixed-sample and participant-linkage limitations.
