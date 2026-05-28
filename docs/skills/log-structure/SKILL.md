---
name: logs-structure
description: Training logs directory structure for post-processing and analysis
---

# Logs Directory Structure Documentation

## Overview
The `checkpoint/` directory contains training run outputs organized by `{exp_name}/{exp_id}` folder names. Usually, the training and inference run for the same experiment share `exp_id` but different `exp_name`.

## Directory Structure

### Run Directory Contents

#### 1. Configuration and Metadata
- **`configs.yaml`**: Complete configuration tree for the run
- **`wandb/latest-run/files/wandb-metadata.json`**: System metadata, git info, and run parameters

#### 2. Checkpoints
- **Files**: `checkpoint.pth`, `best-_l2_error.pth`, `periodic-X.pth`
- **Purpose**: Model state at different training steps

#### 3. Logs
- **`train.log`**: Training process logs (rank 0)
- **`train.log-X`**: Training process logs (rank X)
- **`wandb/`**: Weights & Biases experiment tracking files

#### 4. Visualizations: `eval/`
- **Structure**: `eval/epoch_X_Y/` - evaluation outputs at epoch X for rank Y
- **Files**: `D_B.png` - visualization outputs for dataset D and batch B
- **Purpose**: Visual outputs for debugging and analysis

### Finding Run Details
1. Check `configs.yaml` for training configuration
2. Check `wandb/latest-run/files/wandb-metadata.json` for system info and git commit