# WorldScore on one AMD Radeon Pro W7900

Upstream is [haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore) (MIT, ICCV 2025). The original README is kept as [UPSTREAM.md](UPSTREAM.md). This tree is that benchmark, with the changes needed to generate and score it on ROCm.

The run used one GPU. The container was given a single render node, and inside it `HIP_VISIBLE_DEVICES=0` and `ROCR_VISIBLE_DEVICES=0`. The device that answered was an AMD Radeon Pro W7900D, gfx1100, under PyTorch 2.11.0+rocm7.14.0.

## What was verified

Not the leaderboard. The leaderboard is 2000 static instances and 1000 dynamic ones. Verification is the upstream entry points on the first static instance and the first dynamic instance:

```bash
python world_generators/generate_videos.py --model_name cogvideox_2b_t2v --prompt_set smoke.json
python worldscore/run_evaluate.py --model_name cogvideox_2b_t2v
```

The generator is the upstream CogVideoX adapter, pointed at `THUDM/CogVideoX-2b`. The static instance is a photorealistic indoor dining scene with a push-in. The dynamic instance is an articulated elephant with a fixed camera. Each clip is 49 frames at 720×480, 50 denoising steps. Generation took 283 seconds and 281 seconds.

`worldscore/run_evaluate.py` then finished and wrote `worldscore.json`:

| axis | score |
| --- | --- |
| camera_control | 89.48 |
| object_control | 25.0 |
| content_alignment | 0 |
| 3d_consistency | 82.54 |
| photometric_consistency | 61.42 |
| style_consistency | 61.3 |
| subjective_quality | 37.96 |
| motion_accuracy | 0.0 |
| motion_magnitude | 24.41 |
| motion_smoothness | 56.93 |
| WorldScore-Static | 51.1 |
| WorldScore-Dynamic | 43.9 |

`content_alignment` is 0 because a push-in is an intra-scene move, and the evaluator skips content alignment for those (`evaluator.py` drops it when `camera_movement_type == "intra"`). With only that one static instance in the smoke set, the aggregate has nothing else to average. `motion_accuracy` ran across the dynamic frames; the raw score was negative and the benchmark's normalizer clamped it to 0. Per-instance files are in `results/`.

## How to run it

Image: `rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`. Start the container with one render node:

```bash
bash scripts/radeon_start_container.sh /path/to/workdir /dev/dri/renderD130
```

The Python that matters is `/opt/venv/bin/python`. Do not let pip replace that torch. Install the ROCm devel headers before any HIP extension; the runtime wheel does not ship rocThrust, and `torch` headers include `thrust/complex.h` while hipcc is compiling:

```bash
pip install --no-deps /path/to/rocm_sdk_devel-7.14.0-py3-none-linux_x86_64.whl
python -m rocm_sdk init
export ROCM_DEVEL=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export ROCM_SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
export CPLUS_INCLUDE_PATH=$ROCM_DEVEL/include:$ROCM_SDK/include
export DEVICE_LIB_PATH=$ROCM_SDK/lib/llvm/amdgcn/bitcode
export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=$ROCM_SDK --rocm-device-lib-path=$DEVICE_LIB_PATH"
export PYTORCH_ROCM_ARCH=gfx1100
```

`libamdhip64.so` is only shipped as `libamdhip64.so.7`. Symlink the unversioned soname onto `LIBRARY_PATH` before linking.

Then:

```bash
bash scripts/link_eigen.sh
export PYTHONPATH=$PWD/shim:$PWD:$PWD/thirdparty/DROID-SLAM:$PWD/worldscore/benchmark/metrics/third_party:$PWD/worldscore/benchmark/metrics/third_party/SEA-RAFT
pip install --no-build-isolation thirdparty/DROID-SLAM/thirdparty/lietorch
pip install --no-build-isolation thirdparty/DROID-SLAM
pip install --no-build-isolation --no-deps -e .
```

`shim/torch_scatter` provides `scatter_sum` and `scatter_mean` with `scatter_add_`. DROID calls only those two, and the torch-scatter HIP extension needs rocThrust sources this image does not unpack.

Weights and the dataset stay outside the repo. CogVideoX-2b, the WorldScore dataset, and the metric checkpoints (DROID-SLAM, GroundingDINO Swin-T, SAM 2.1 Hiera-B+, SEA-RAFT Tartan-C, VFIMamba) go under the paths in `config/`. `scripts/make_smoke.py` writes the one-instance prompt sets the loader expects.

## What changed relative to upstream

- `world_generators/configs/cogvideox_2b_t2v.yaml` asked for CogVideoX-5B weights while the model name selects the 2B scheduler. It now loads `THUDM/CogVideoX-2b`.
- DROID's `setup.py` called `setup()` twice and passed nvcc `-gencode` flags. Lietorch is built from its own tree. `droid_backends` is the only extension left, and its dispatch uses `scalar_type()`.
- Lietorch's backward kernel calls `Matrix4x4().template block<3,3>(0,0)`. Clang was parsing the angle brackets as a comparison.
- GroundingDINO's multi-scale deformable attention uses the PyTorch fallback unless `_C` actually imported. On HIP, `torch.cuda.is_available()` is true and the CUDA op is absent.
- The BERT wrapper no longer borrows `get_extended_attention_mask` from transformers 5. It builds the same additive mask the encoder still adds to attention scores, including the 3D phrase mask.
- VFIMamba uses the published PyTorch `selective_scan_ref` when `mamba_ssm` is not installed.

The diff is in `results/radeon.patch`.
