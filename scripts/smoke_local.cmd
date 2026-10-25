@echo off
setlocal
cd /d "%~dp0.."
if not defined DGSEG_DATA_ROOT set "DGSEG_DATA_ROOT=E:\ADA2\DGSEG_DATA"
py -m scripts.build_manifests --data-root "%DGSEG_DATA_ROOT%" --track all || exit /b 1
py -m unittest discover -s tests -v || exit /b 1
for %%M in (erm mixstyle randconv) do (
  py -m scripts.train --data-root "%DGSEG_DATA_ROOT%" --track r19 --method %%M --run-dir "runs\r19_%%M_smoke" --smoke --batch-size 2 || exit /b 1
)
py -m scripts.evaluate --data-root "%DGSEG_DATA_ROOT%" --track r19 --target mapillary_v1.2 --checkpoint "runs\r19_erm_smoke\best_source_val.pt" --output "reports\evaluations\r19_erm_mapillary_native_smoke.json" --native-smoke || exit /b 1
for %%T in (rs7_u2r rs7_r2u) do (
  py -m scripts.train --data-root "%DGSEG_DATA_ROOT%" --track %%T --method erm --run-dir "runs\%%T_erm_smoke" --smoke --batch-size 2 || exit /b 1
)
py -m scripts.evaluate --data-root "%DGSEG_DATA_ROOT%" --track rs7_u2r --target rural --checkpoint "runs\rs7_u2r_erm_smoke\best_source_val.pt" --output "reports\evaluations\rs7_u2r_erm_smoke.json" --smoke || exit /b 1
py -m scripts.evaluate --data-root "%DGSEG_DATA_ROOT%" --track rs7_r2u --target urban --checkpoint "runs\rs7_r2u_erm_smoke\best_source_val.pt" --output "reports\evaluations\rs7_r2u_erm_smoke.json" --smoke || exit /b 1
py -m scripts.summarize || exit /b 1
py -m scripts.report_smoke || exit /b 1
echo Local smoke completed. All smoke results are non-publishable.
