# Paddle + PyTorch in one process (fixed 2026-11-18)

Symptom: whichever of paddle / torch is imported second fails with `OSError: [WinError 127] The specified procedure
could not be found` (cublas64_12.dll or torch\lib\shm.dll).

Cause: same-named DLLs in different versions.
- torch 2.11 cu128 bundles CUDA **12.8** cublas/cudart/cufft/curand/cusolver/cusparse in `torch\lib`; paddlepaddle-gpu
  3.2.1 cu129 uses the **12.9** builds from the `nvidia-*` wheels. Windows loads one DLL per name per process.
- Both also ship `libiomp5md.dll` (paddle\libs vs torch\lib).

Fix (`cuda_preload.py`, called at the top of `run.py` before cv2/paddle/torch imports):
1. load all 12.9 DLLs from `.venv\Lib\site-packages\nvidia\*\bin` first - torch built for 12.8 runs on newer 12.x;
2. import torch before paddle, so torch's libiomp5md.dll is the one loaded.

Verified: both orders of GPU conv + matmul work; `run.py --variant vietocr --pages vanbangoc_p2` on GPU ->
2.7 s/page, torch peak 333 MiB, WER 2.8%, stamp region found, 31/31 line boxes. No need to split into two processes.
Keep the versions as they are (torch cu128 + paddle cu129): upgrading either can change which DLLs clash.
