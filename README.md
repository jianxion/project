# Study on All-to-all Communication for Mixture of Experts

**Authors:** Tianhua Xia (tx856), Jianxiong Shen (js12685)  
**Course:** High Speed Networks, Fall 2025

## Overview

This project studies expert load imbalance in Mixture of Experts (MoE) models and its impact on all-to-all communication performance.

## Objectives

1. Profile expert assignment distribution in MoE models
2. Study how changing expert assignment affects model accuracy
3. Fine-tune MoE models with expert imbalance loss
4. Simulate network performance under different expert assignments

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── profiling/          # Expert assignment profiling
│   ├── assignment/         # Expert assignment modification
│   ├── training/           # Fine-tuning with imbalance loss
│   ├── evaluation/         # Perplexity evaluation
│   └── simulation/         # Network simulation
├── experiments/            # Experiment scripts
├── results/               # Output results and plots
└── data/                  # Dataset cache
```

## Usage

### 1. Profile Expert Assignments
```bash
python experiments/profile_expert_assignment.py
```

### 2. Test Heuristic Assignment Changes
```bash
python experiments/test_assignment_changes.py
```

### 3. Fine-tune with Imbalance Loss
```bash
python experiments/finetune_with_imbalance_loss.py
```

### 4. Simulate Network Performance
```bash
python experiments/simulate_network.py
```
