# Dataset audit — 2026-10-25

The benchmark reads the official archives at `DGSEG_DATA_ROOT/downloads/`. Original images and labels remain outside this code project. The manifests were generated from archive member names and verified that each listed image has its corresponding mask. The ZIP directory and 20 images/masks per available role were decoded; **this is not a full CRC/decode scan of every member**.

| Dataset archive | Size (bytes) | Integrity record | Usable benchmark split |
|---|---:|---|---|
| Cityscapes `leftImg8bit_trainvaltest.zip` | 11,592,327,197 | official MD5 `0a6e97e94b616a514066c9e2adb0c97f` checked previously | train 2,975; val 500 |
| Cityscapes `gtFine_trainvaltest.zip` | 252,567,705 | official MD5 `4237c19de34c8a376e9ba46b495d6f66` checked previously | 2,975/500 `labelIds` paired |
| LoveDA `Train.zip` | 4,021,669,263 | Zenodo MD5 `de2b196043ed9b4af1690b3f9a7d558f` checked previously | Urban 1,156; Rural 1,366 |
| LoveDA `Val.zip` | 2,425,958,254 | Zenodo MD5 `84cae2577468ff0b5386758bb386d31d` checked previously | Urban 677; Rural 992 |
| Mapillary Research v1.2 archive | 25,624,067,065 | local SHA-256 `61518267961941a440e3d561da2ed84ef1aa4980bb1feeeafbc5c546d120f24f`; no official checksum comparison | validation 2,000 images + 2,000 semantic masks |
| SYNTHIA-RAND-CITYSCAPES RAR | 20,970,055,295 preallocated | **incomplete: 10/200 chunks**; do not use | blocked |

BDD100K semantic segmentation and ACDC are absent because official access is pending. Their metrics are blank in `benchmark_summary.csv`.

## Label and leakage checks

- Cityscapes uses raw `labelIds` and the explicit 19-class LUT in `dgseg/labels.py`. The archive does not contain `labelTrainIds`; the code does not assume otherwise.
- Mapillary v1.2 uses indexed semantic PNGs and a [66-row mapping](../configs/mappings/mapillary_v1_2_to_cityscapes19.json) with reasons for every kept or ignored source label. The archive README gives v1.2; the internal config field `version=1.1` is not treated as the archive version.
- LoveDA maps raw `0` to `255` and raw `1..7` to `0..6`, preserving seven valid classes.
- Cityscapes `val` is the only selection set for `r19`. LoveDA source Train is partitioned 90/10 using SHA-256 of sample IDs and seed 42. Target labels are loaded only by the separate evaluator.
- `DATASET_AUDIT_SAMPLE.md` and `audit_overlays/` record 20 deterministic real samples per available role, their observed label IDs, sampled pixel counts, and ignore ratio. The overlays are local artifacts and should not be included in a public code repository.

## Actual manifest counts

| Role | Images |
|---|---:|
| `r19/source_train` | 2,975 |
| `r19/source_val` | 500 |
| `r19/target_mapillary_v1.2` | 2,000 |
| `rs7_u2r/source_train` | 1,040 |
| `rs7_u2r/source_val` | 116 |
| `rs7_u2r/target_rural` | 992 |
| `rs7_r2u/source_train` | 1,229 |
| `rs7_r2u/source_val` | 137 |
| `rs7_r2u/target_urban` | 677 |

Manifest SHA-256 values are in `reports/manifests/*_manifest_hashes.json`. The source code ZIP contains the manifests, mapping audit, protocol, and scripts; it contains no raw images, labels, or model weights.
