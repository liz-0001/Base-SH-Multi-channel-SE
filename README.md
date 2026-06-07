# TFG

TFG is a multi-channel speech enhancement project based on TFGridNetV2 with an
optional ADFS module for adaptive frequency and spherical-harmonic channel
selection.

The current training pipeline targets the `Mic8_2s_gpurir` dataset. Mixture
signals are converted to spherical harmonic coefficients using microphone array
geometry, then enhanced by TFGridNetV2. Evaluation reports PESQ, STOI, SDR, and
SI-SDR.

## Project Structure

```text
.
+-- train.py                  # Training entry point
+-- inference.py              # Inference and PESQ/STOI quick evaluation
+-- evaluation_fixed.py       # Full metric evaluation for enhanced wavs
+-- config/
|   +-- train_config.py       # Legacy PonderEnhancer config
+-- loader/
|   +-- IGCRN_dataloader.py   # Full dataset loader
|   +-- small_test.py         # Deterministic 10% subset loader
+-- networks/
|   +-- tfgridnetv2.py        # Main TFGridNetV2 model
|   +-- adfs_module.py        # ADFS module
|   +-- IGCRN.py              # Legacy/reference model
|   +-- enhancer.py           # Legacy PonderEnhancer model
+-- utils/                    # Loss, plotting, and ponder helpers
+-- model_test/               # Local checkpoints and curves
+-- logs/                     # Training logs
+-- runs/                     # TensorBoard events
+-- record/                   # Historical experiment records
```

## Environment

Python 3.8 or newer is recommended. The project uses PyTorch, ESPnet, and audio
metric packages that are easier to install in a dedicated environment.

```bash
conda create -n tfg python=3.9 -y
conda activate tfg
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default wheel is
not suitable for your server.

## Dataset Layout

By default, scripts expect the dataset at:

```text
/data/lizhe/SH_data/Mic8_2s_gpurir
```

Expected subdirectories include:

```text
loader_txt/wav_scp/
generated_data/train/mix/
generated_data/train/noreverb_ref/
generated_data/val/mix/
generated_data/val/noreverb_ref/
generated_data/test_mic_8/mix/
generated_data/test_mic_8/noreverb_ref/
RIR/cir_uniform_8/train_val_rir/MIC/
RIR/cir_uniform_8/test_rir/mic_8/MIC/
```

You can override these paths with command-line arguments in `train.py`,
`inference.py`, and `evaluation_fixed.py`.

## Training

The current `train.py` imports the 10% subset loader:

```python
from loader.small_test import make_fix_loader
```

This is useful for debugging and quick experiments. For full training, switch it
back to:

```python
from loader.IGCRN_dataloader import make_fix_loader
```

Run training with ADFS enabled:

```bash
python train.py --gpuid 0 --num_epoch 100 --batch_size 2
```

Run the no-ADFS baseline:

```bash
python train.py --no_adfs --gpuid 0 --num_epoch 100 --batch_size 2
```

Checkpoints are written to:

```text
model_test/          # ADFS enabled
model_test_noadfs/   # ADFS disabled
```

Logs and TensorBoard events are written to `logs/` and `runs/`.

## Inference

Run inference using the best ADFS checkpoint:

```bash
python inference.py \
  --modelpath model_test/ \
  --model_name model_best.pth \
  --file_path /data/lizhe/SH_data/Mic8_2s_gpurir \
  --mic_path_root /data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/test_rir
```

Run inference for a no-ADFS checkpoint:

```bash
python inference.py --no_adfs --modelpath model_test_noadfs/
```

Enhanced wavs are saved under the dataset root, for example:

```text
/data/lizhe/SH_data/Mic8_2s_gpurir/predictions_tfg_serial_test_mic_8/
```

## Evaluation

Use `evaluation_fixed.py` for detailed metrics:

```bash
python evaluation_fixed.py \
  --dataset_root /data/lizhe/SH_data/Mic8_2s_gpurir \
  --prediction_path /data/lizhe/SH_data/Mic8_2s_gpurir/predictions_tfg_serial_test_mic_8 \
  --test_name mic_8
```

It saves per-utterance CSV, summary CSV, and `.mat` metric files.

## GitCode Synchronization

The repository remote is expected to be:

```bash
git@gitcode.com:maple_leaff/TFG.git
```

Recommended workflow:

```bash
git status
git pull --rebase origin main
git add README.md requirements.txt .gitignore scripts/clean_artifacts.sh
git commit -m "Add project documentation and cleanup helpers"
git push origin main
```

## Cleanup

Runtime artifacts such as checkpoints, TensorBoard events, logs, caches, and
notebook checkpoints are ignored by `.gitignore`.

To clean local generated artifacts manually:

```bash
bash scripts/clean_artifacts.sh
```

The cleanup script removes common local outputs only. It does not touch source
files or Git history.

If artifacts were already tracked by Git, remove them from the Git index while
keeping local files:

```bash
bash scripts/untrack_artifacts.sh
```

Commit that change before pushing if you want GitCode to stop storing those
generated files.
