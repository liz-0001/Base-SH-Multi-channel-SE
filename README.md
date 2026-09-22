# Order-Aware Spherical Harmonic Speech Enhancement (8Mic)

This branch implements the proposed **order-aware spherical harmonic (SH) front end** for eight-microphone speech enhancement. It groups SH features by order, exchanges information between low and high orders, and models interactions between adjacent orders before passing the fused features to TF-GridNetV2.

The experiments use a simulated eight-microphone circular array and the MS-SNSD speech and noise data.

## Method

The model processes each mixture as follows:

1. Read the eight microphone signals and their saved array coordinates.
2. Project the signals onto a fourth-order real SH basis, producing 25 SH feature channels.
3. Apply STFT and concatenate real and imaginary components.
4. Split the features into five SH-order groups containing 1, 3, 5, 7, and 9 SH channels, respectively.
5. Encode each order group separately into 32-dimensional features.
6. Exchange information between the low-order group `{0, 1}` and high-order group `{2, 3, 4}` using gated mutual guidance.
7. Apply gated interactions between adjacent orders.
8. Fuse the order features and pass them to TF-GridNetV2 to predict the enhanced complex spectrum.
9. Apply inverse STFT to obtain the enhanced waveform.

The SH projection is used as a structured feature encoding. It is not intended to recover a unique fourth-order three-dimensional sound field from eight microphones.

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

`train.py` and `inference.py` must instantiate the proposed order-aware model. The data loader supplies the eight-channel mixture and the corresponding microphone coordinates. TF-GridNetV2 serves as the enhancement backbone.

Check this tree against the proposed branch before publishing, especially if the order-aware front end is stored in a separate file under `networks/`.

## Dataset

The data preparation workflow is adapted from the `SH_Generalization_num` directory of the SH-injection project. This branch uses its eight-microphone setting; it does not run the full microphone-number generalization experiment.

| Item | Setting |
| --- | --- |
| Clean speech and noise | MS-SNSD |
| Room impulse responses | gpuRIR |
| Array | Eight-microphone uniform circular array |
| Array radius | 0.35 m |
| Sample rate | 16 kHz |
| Training segment length | 2 s |
| Room size | 6 × 5 × 4 m |
| Test array | `mic_8` |

The expected dataset name is `Mic8_2s_gpurir`. The default dataset root in the existing scripts is:

```text
/data/lizhe/SH_data/Mic8_2s_gpurir
```

Update the script paths if the dataset is stored elsewhere.

### Expected Dataset Layout

```text
Mic8_2s_gpurir/
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

The `wav_scp` entries must match the generated WAV filenames. The data loader and inference code use the RIR ID in each filename to locate its microphone-coordinate file under the corresponding `MIC/` directory.

## Data Preparation

The reference scripts are located in:

```text
third_party/SH_injection/SH_Generalization_num/Data_prepare/
```

Before running them, replace their hardcoded `/ddnstor/...` paths with paths under your dataset root. Set the test RIR generator to produce the `mic_8` configuration.

Run the scripts in this order:

```bash
python third_party/SH_injection/SH_Generalization_num/Data_prepare/genRIR_train_val.py
python third_party/SH_injection/SH_Generalization_num/Data_prepare/genRIR_test.py

python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_train.py \
  --chunk_len 2 --num_process 8

python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_val.py \
  --chunk_len 2 --num_process 8

python third_party/SH_injection/SH_Generalization_num/Data_prepare/gen_mix_test_num8.py \
  --chunk_len 2 --num_process 8
```

Prepare the six clean/noise file lists under `loader_txt/wav_list/` before generating mixtures. The scripts should also write the RIR lists and `wav_scp` files shown in the dataset layout.

The generated targets are:

- `mix/`: noisy eight-channel mixture;
- `reverb_ref/`: reverberant clean speech;
- `noreverb_ref/`: clean reference used by the current training and evaluation workflow.

## Installation

```bash
pip install -r requirements.txt
```

The data generation scripts additionally require their simulation dependencies, including gpuRIR.

## Training

The paper trains the order-aware variants and the SH-only baseline with the same objective:

```text
MixLoss = MSE loss + 0.01 × SI-SDR loss + 0.5 × STFT loss
```

The reported setup uses Adam, an initial learning rate of `1e-3`, a batch size of `8`, and up to `100` epochs. The checkpoint with the lowest validation loss is selected for evaluation.

After confirming that `train.py` instantiates the order-aware model and uses MixLoss, run:

```bash
python train.py \
  --num_epoch 100 \
  --batch_size 8 \
  --num_worker 0 \
  --model_dir model_order_aware_sh_8mic
```

The existing path defaults in `train.py` point to the `Mic8_2s_gpurir` training and validation lists, mixtures, `noreverb_ref` targets, and microphone-coordinate directories. Change those defaults or pass the corresponding path arguments if your dataset is stored elsewhere.

Checkpoints and training records are written to the directory passed through `--model_dir`.

## Inference

Run inference using the best validation checkpoint:

```bash
python inference.py \
  --modelpath model_order_aware_sh_8mic \
  --model_name model_best.pth \
  --test_name mic_8 \
  --file_path /data/lizhe/SH_data/Mic8_2s_gpurir \
  --mic_path_root /data/lizhe/SH_data/Mic8_2s_gpurir/RIR/cir_uniform_8/test_rir
```

Before running this command, confirm that `inference.py` loads the same order-aware model definition used during training. Note the directory in which it writes enhanced WAV files; use that directory for the evaluation command below.

## Evaluation

```bash
python evaluation_fixed.py \
  --dataset_root /data/lizhe/SH_data/Mic8_2s_gpurir \
  --prediction_path PATH_TO_PROPOSED_PREDICTIONS \
  --test_name mic_8
```

The evaluation script reports PESQ, STOI, SDR, and SI-SDR, and saves per-utterance and average results.

### Results Reported in the Paper

All variants in the following comparison use MixLoss.

| Model | PESQ | STOI (%) | SDR (dB) | SI-SDR (dB) |
| --- | ---: | ---: | ---: | ---: |
| SH-only baseline | 2.47 | 84.70 | 8.28 | 6.69 |
| Order-wise grouping only | 2.52 | 85.75 | 8.41 | 6.94 |
| Grouping + adjacent-order interaction | 2.61 | 86.80 | 8.86 | 7.42 |
| Grouping + low/high-order guidance | 2.61 | 86.87 | 8.96 | 7.51 |
| Full order-aware SH model | **2.63** | **87.25** | **9.13** | **7.72** |

These are the paper's reported results, not values automatically produced by a new run. The comparison isolates the proposed front end from the SH-only baseline under the same training objective.

## Cleanup

Runtime artifacts such as checkpoints, logs, TensorBoard events, caches, and predictions should remain outside Git tracking.

```bash
bash scripts/clean_artifacts.sh
```

If generated files were accidentally tracked, remove them from Git tracking while retaining local copies:

```bash
bash scripts/untrack_artifacts.sh
```
