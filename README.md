# TFG 8Mic Baseline

This branch is for reproducing the 8-microphone serial TFGridNet baseline from the
SH-injection project. The model input is 4th-order real spherical harmonic
coefficients generated from 8-channel microphone signals, so the network input
dimension is 25.

## Project Structure

```text
.
├── train.py
├── inference.py
├── evaluation_fixed.py
├── requirements.txt
├── loader/
│   └── IGCRN_dataloader.py
├── networks/
│   └── tfgridnetv2.py
├── third_party/
│   └── SH_injection/SH_Generalization_num/
│       └── Data_prepare/
└── scripts/
    ├── clean_artifacts.sh
    └── untrack_artifacts.sh
```

The current source tree keeps only code related to training, inference,
evaluation, data loading, and 8Mic data preparation. Runtime
outputs such as logs, TensorBoard events, checkpoints, caches, and old experiment
records are intentionally not kept in the source tree.

## Baseline Dataset

The TFG-serial implementation and data preparation scripts used here come from
the original repository's `SH_Generalization_num` directory because that is where
the author provides the TFG serial code. This branch does not run the full
microphone-number generalization experiment; it only reproduces the 8Mic
TFG-serial PESQ/STOI setting.

The paper/code baseline uses:

- Clean/noise corpus: MS-SNSD split into train, validation, and test lists.
- Room simulation: gpuRIR.
- Microphone array: circular 8-microphone array.
- Test condition used in this branch: `mic_8` only.
- Sample rate: 16 kHz.
- Segment length: 2 seconds in this branch.
- SNR range in the data generation scripts: `[-10, 10]` dB.
- RT60 range in the RIR scripts: roughly `[0.1, 1.0]` seconds from the current
  author script implementation.
- Room size in the author scripts: `[6, 5, 4]` meters.

The generated dataset expected by this branch is named:

```text
Mic8_2s_gpurir
```

Default dataset root used by training/inference/evaluation:

```text
/data/lizhe/SH_data/Mic8_2s_gpurir
```

Change this path if your server stores the dataset elsewhere.

## Expected Dataset Layout

```text
/data/lizhe/SH_data/Mic8_2s_gpurir/
├── loader_txt/
│   ├── rir/
│   │   ├── rir_train_val.txt
│   │   ├── mic_train_val.txt
│   │   ├── rir_test_mic_8.txt
│   │   └── mic_test_mic_8.txt
│   ├── wav_list/
│   │   ├── train_clean_list.txt
│   │   ├── train_noise_list.txt
│   │   ├── val_clean_list.txt
│   │   ├── val_noise_list.txt
│   │   ├── test_clean_list.txt
│   │   └── test_noise_list.txt
│   └── wav_scp/
│       ├── wav_scp_train.txt
│       ├── wav_scp_val.txt
│       └── wav_scp_test_mic_8.txt
├── RIR/
│   └── cir_uniform_8/
│       ├── train_val_rir/
│       │   ├── RIR/
│       │   └── MIC/
│       └── test_rir/
│           └── mic_8/
│               ├── RIR/
│               └── MIC/
└── generated_data/
    ├── train/
    │   ├── mix/
    │   ├── reverb_ref/
    │   └── noreverb_ref/
    ├── val/
    │   ├── mix/
    │   ├── reverb_ref/
    │   └── noreverb_ref/
    └── test_mic_8/
        ├── mix/
        ├── reverb_ref/
        └── noreverb_ref/
```

`wav_scp` entries must match generated wav names. The dataloader and inference
code use the `rir` id embedded in each utterance name to find the corresponding
microphone geometry file:

```text
RIR/cir_uniform_8/train_val_rir/MIC/mic_array_pos*.npy
RIR/cir_uniform_8/test_rir/mic_8/MIC/mic_array_pos*.npy
```

## Data Preparation Scripts

The data preparation scripts are kept as third-party reference code:

```text
third_party/SH_injection/SH_Generalization_num/Data_prepare/
```

For 8Mic reproduction, use these scripts:

```text
genRIR_train_val.py      # Generate train/validation 8Mic RIR and MIC geometry
genRIR_test.py           # Generate test RIR/MIC geometry; keep only mic_8 if reproducing 8Mic
gen_mix_train.py         # Generate train mix/reverb_ref/noreverb_ref and wav_scp_train.txt
gen_mix_val.py           # Generate validation mix/reverb_ref/noreverb_ref and wav_scp_val.txt
gen_mix_test_num8.py     # Generate 8Mic test mix/reverb_ref/noreverb_ref and wav_scp_test_mic_8.txt
tools.py                 # Helper functions used by data generation scripts
```

### Paths To Modify Before Running

The author scripts still contain hardcoded `/ddnstor/...` paths. Replace them
with your actual dataset root before running.

Use this root consistently:

```text
TODO_PATH_DATA_ROOT=/data/lizhe/SH_data/Mic8_2s_gpurir
```

In `genRIR_train_val.py`, modify:

```python
out_path = "TODO_PATH_DATA_ROOT/RIR/cir_uniform_8/train_val_rir/"
txt_path = "TODO_PATH_DATA_ROOT/loader_txt/rir/"
```

In `genRIR_test.py`, modify the loop or hardcoded path so the 8Mic output is:

```python
out_path = "TODO_PATH_DATA_ROOT/RIR/cir_uniform_8/test_rir/mic_8/"
txt_path = "TODO_PATH_DATA_ROOT/loader_txt/rir/"
```

If you only reproduce 8Mic, you can change the test microphone loop to:

```python
for num_mics in [8]:
    ...
```

In `gen_mix_train.py`, modify:

```python
--clean_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/train_clean_list.txt
--noise_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/train_noise_list.txt
--rir_wav_list    TODO_PATH_DATA_ROOT/loader_txt/rir/rir_train_val.txt
--config_path     TODO_PATH_DATA_ROOT/generated_data/train/mix_train.config
--save_path       TODO_PATH_DATA_ROOT/generated_data/train
```

Also replace the hardcoded output path for:

```text
TODO_PATH_DATA_ROOT/loader_txt/wav_scp/wav_scp_train.txt
```

In `gen_mix_val.py`, modify:

```python
--clean_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/val_clean_list.txt
--noise_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/val_noise_list.txt
--rir_wav_list    TODO_PATH_DATA_ROOT/loader_txt/rir/rir_train_val.txt
--config_path     TODO_PATH_DATA_ROOT/generated_data/val/mix_val.config
--save_path       TODO_PATH_DATA_ROOT/generated_data/val
```

Also replace the hardcoded output path for:

```text
TODO_PATH_DATA_ROOT/loader_txt/wav_scp/wav_scp_val.txt
```

In `gen_mix_test_num8.py`, modify:

```python
--clean_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/test_clean_list.txt
--noise_wav_list  TODO_PATH_DATA_ROOT/loader_txt/wav_list/test_noise_list.txt
--rir_wav_list    TODO_PATH_DATA_ROOT/loader_txt/rir/rir_test_mic_8.txt
--config_path     TODO_PATH_DATA_ROOT/generated_data/test_mic_8/mix_test_mic_8.config
--save_path       TODO_PATH_DATA_ROOT/generated_data/test_mic_8
```

Also replace the hardcoded output path for:

```text
TODO_PATH_DATA_ROOT/loader_txt/wav_scp/wav_scp_test_mic_8.txt
```

### Data Preparation Order

1. Prepare MS-SNSD clean/noise wav lists.

```bash
mkdir -p /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list

find TODO_PATH_MS_SNSD/clean_train -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/train_clean_list.txt
find TODO_PATH_MS_SNSD/noise_train -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/train_noise_list.txt
find TODO_PATH_MS_SNSD/clean_val -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/val_clean_list.txt
find TODO_PATH_MS_SNSD/noise_val -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/val_noise_list.txt
find TODO_PATH_MS_SNSD/clean_test -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/test_clean_list.txt
find TODO_PATH_MS_SNSD/noise_test -name "*.wav" | sort > /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_list/test_noise_list.txt
```

2. Generate train/validation RIR and microphone geometry.

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/genRIR_train_val.py
```

3. Generate 8Mic test RIR and microphone geometry.

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/genRIR_test.py
```

4. Generate train mixtures.

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_train.py \
  --chunk_len 2 \
  --num_process 8
```

5. Generate validation mixtures.

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_val.py \
  --chunk_len 2 \
  --num_process 8
```

6. Generate 8Mic test mixtures.

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_test_num8.py \
  --chunk_len 2 \
  --num_process 8
```

The mix scripts write three parallel targets:

- `mix/`: noisy multi-channel mixture.
- `reverb_ref/`: reverberant clean speech.
- `noreverb_ref/`: early/direct clean reference used by the current training and evaluation scripts.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run 8Mic baseline training:

```bash
python train.py \
  --num_epoch 100 \
  --batch_size 2 \
  --num_worker 0 \
  --model_dir model_tfg_serial_8mic
```

Important default paths in `train.py`:

```text
--train_wav_scp  /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_train.txt
--train_mix_dir  /data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/train/mix
--train_ref_dir  /data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/train/noreverb_ref
--train_mic_dir  /data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC
--val_wav_scp    /data/lizhe/SH_data/Mic8_2s_gpurir/loader_txt/wav_scp/wav_scp_val.txt
--val_mix_dir    /data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/val/mix
--val_ref_dir    /data/lizhe/SH_data/Mic8_2s_gpurir/generated_data/val/noreverb_ref
--val_mic_dir    /data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/train_val_rir/MIC
```

Training saves checkpoints and loss curves to:

```text
model_tfg_serial_8mic/
```

## Inference

Run 8Mic inference with the best checkpoint:

```bash
python inference.py \
  --modelpath model_tfg_serial_8mic \
  --model_name model_best.pth \
  --test_name mic_8 \
  --file_path /data/lizhe/SH_data/Mic8_2s_gpurir \
  --mic_path_root /data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/test_rir
```

Enhanced wavs are saved to:

```text
/data/lizhe/SH_data/Mic8_2s_gpurir/predictions_tfg_serial_test_mic_8/
```

`inference.py` also prints average PESQ/STOI and saves a `.mat` summary under:

```text
model_tfg_serial_8mic/result_model_best/
```

## Evaluation

Run detailed evaluation after inference:

```bash
python evaluation_fixed.py \
  --dataset_root /data/lizhe/SH_data/Mic8_2s_gpurir \
  --prediction_path /data/lizhe/SH_data/Mic8_2s_gpurir/predictions_tfg_serial_test_mic_8 \
  --test_name mic_8
```

The evaluation script reports and saves:

- PESQ
- STOI
- SDR
- SI-SDR
- per-utterance CSV
- average summary CSV
- MATLAB `.mat` file

## Model Input Processing

During training and inference:

1. Read 8-channel mixture wav.
2. Read microphone geometry from `mic_array_pos*.npy`.
3. Convert Cartesian microphone positions to spherical coordinates.
4. Convert the 8-channel mixture to 4th-order real spherical harmonic
   coefficients with `spaudiopy.sph.src_to_sh`.
5. Feed 25 SH coefficients into TFGridNetV2.

The relevant code is:

```text
loader/IGCRN_dataloader.py
inference.py
networks/tfgridnetv2.py
```

## Cleanup

Generated artifacts are ignored by `.gitignore`. To remove local runtime outputs:

```bash
bash scripts/clean_artifacts.sh
```

If generated files were accidentally tracked, remove them from Git tracking while
keeping local copies:

```bash
bash scripts/untrack_artifacts.sh
```
