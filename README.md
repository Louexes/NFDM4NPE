# ** Neural Flow Diffusion Models for Amortized Neural Posterior Estimation

## Overview

This repository contains a comprehensive framework for comparing three approaches to neural posterior estimation (NPE) based on Chen et al 2024 (https://github.com/TianyuCodings/cDiff) (Paper: https://arxiv.org/abs/2410.19105)

## Project Structure

### Core Files
- **`main.py`** - Main script with command-line interface
- **`trainer.py`** - Training loop implementation with evaluation and logging
- **`utils.py`** - Utility functions for model saving, data handling, and visualization

### 📁 `datasets/` - Benchmark Datasets
Contains 15+ synthetic and real-world datasets for testing NPE methods:
- **IID datasets**: `cos`, `g_and_k`, `normal_gamma`, `normal_wishart`, `witch_hat`, `socks`, `species_sampling`, `dirichlet_multinomial`, `dirichlet_laplace`, `possion_gamma`
- **Time-series datasets**: `lotka_volterra`, `markov_switch`, `stochastic_vol`, `fBM`, `minnesota`
- Each dataset implements: `sample_theta()`, `sample_X()`, `my_gen_sample_size()`, `return_dl_ds()`

### 📁 `models/` - Three NPE Approaches
- **`neural_sampler.py`** - Main model classes for all three approaches:
  - `NormalizingFlowPosteriorSampler` - Traditional flow-based NPE
  - `DiffusionPosteriorSampler` - Conditional diffusion (EDM) approach
  - `NeuralDiffusionPosteriorSamplerSigma` - Neural diffusion sigma variant
  - `NeuralDiffusionPosteriorSampler` - Neural diffusion t variant
  - 
- **`normalizing_flow.py`** - Normalizing flow implementation (cNF baseline)
- **`diffusion.py`** - Conditional diffusion implementation (EDM)
- **`nfdm_sigma.py`** - NFDM core files (sigma)
- **`nfdm_t.py`** - NFDM core files (t)
- **`summary.py`** - Summary network architectures for data encoding
- **`utils.py`** - Model utility functions

### 📁 `evaluation/` - Assessment Methods
- **`SBC.py`** - Simulation-Based Calibration implementation
- **`TARP.py`** - Test of Amortized Posterior implementation

### 📁 `scripts/` - Training Scripts
- **`run_iid.sh`** - Batch training for IID datasets
- **`run_time.sh`** - Batch training for time-series datasets  
- **`run_raw.sh`** - Raw data processing scripts

### 📁 `Results(Reproduction)/` - Comparative Results
Contains experimental results comparing all three approaches:
- **`cos/`** - Cosine dataset: Diffusion vs NormalizingFlow comparison
- **`dirichlet_multinomial/`** - Dirichlet-multinomial: Both approaches with loss trajectories
- **`witches_hat(d=2)/`** - 2D witch's hat: Visual comparison of posterior quality
- **`witches_hat(d=5)/`** - 5D witch's hat: High-dimensional performance comparison

### 📁 `resultsNFDMVolatilityZero/` - NFDM Advanced Results
Contains results from Neural Diffusion experiments showing:
- **`cos/NeuralDiffusionSigma/`** - NFDM with learnable sigma parameter

### Configuration Files
- **`requirements.txt`** - Python dependencies (PyTorch, NumPy, Pandas, etc.)
- **`*.job`** - SLURM job configuration files for cluster execution

## Setup

### Environment Creation

Create a new conda environment:
```bash
conda create -n NFDM4NPE python=3.9
conda activate NFDM4NPE
```

### Installation

Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Experiment Example:

**Neural Diffusion (NFDM) for Sum of Cosines (Non Encoder):**
```bash
python main.py --save_path result --dataset cos --device 2 --data_type=iid --epochs=100000 --model=NeuralDiffusion --save_model --eval_interval=2500 --lr_decay --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200
python main.py --save_path result --dataset cos --device 2 --data_type=iid --epochs=100000 --model=NeuralDiffusionSigma --save_model --eval_interval=2500 --lr_decay --n_run=10 --ecp_n_sim=100 --ecp_n_samples=200
```

```
