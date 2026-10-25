@echo off
setlocal
cd /d "%~dp0.."
if not defined DGSEG_DATA_ROOT set "DGSEG_DATA_ROOT=E:\ADA2\DGSEG_DATA"
set "TORCH_HOME=E:\ADA2\DGSEG_BENCHMARK\torch_cache"
py -m scripts.train --data-root "%DGSEG_DATA_ROOT%" --track r19 --method erm --run-dir "runs\r19_erm_pretrained_smoke" --smoke --batch-size 2 --pretrained-backbone
