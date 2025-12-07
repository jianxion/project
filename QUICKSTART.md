# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test

```bash
# Run all experiments in quick mode (uses fewer samples)
python run_all_experiments.py --quick
```

This will:
- Profile expert assignments (50 samples)
- Test assignment modifications
- Demonstrate fine-tuning setup
- Run network simulations

**Time**: ~10-15 minutes (depending on hardware)

---

## Step-by-Step Walkthrough

### Step 1: Profile Expert Assignments

```bash
python experiments/profile_expert_assignment.py --max_samples 50
```

**What it does**: Analyzes how the MoE model distributes tokens to experts

**Output**: 
- `results/profiling/profiling_report.txt` - Read this first!
- `results/profiling/plots/` - Visual analysis

**Look for**: Gini coefficient values (higher = more imbalance)

---

### Step 2: Simulate Network Impact

```bash
python experiments/simulate_network.py
```

**What it does**: Models how expert imbalance affects communication time

**Output**:
- `results/simulation/simulation_results.json`
- `results/simulation/communication_comparison.png`

**Key insight**: See how imbalance increases latency

---

### Step 3: Test Assignment Changes (Optional)

```bash
python experiments/test_assignment_changes.py --max_samples 50
```

**What it does**: Tests how modifying expert assignments affects accuracy

**Note**: This demonstrates the framework. Full implementation requires model integration.

---

### Step 4: Fine-tune (Advanced, Optional)

```bash
python experiments/finetune_with_imbalance_loss.py \
    --max_samples 100 \
    --num_epochs 1
```

**What it does**: Trains model to produce balanced expert assignments

**Requirements**: GPU with >16GB VRAM recommended

---

## Understanding Your Results

### 1. Check Profiling Report

```bash
cat results/profiling/profiling_report.txt
```

Look for:
- Average Gini coefficient (target: < 0.2)
- Most/least used experts
- Imbalance across layers

### 2. View Visualizations

Open in your browser or image viewer:
- `results/profiling/plots/imbalance_metrics.png`
- `results/profiling/plots/expert_heatmap.png`
- `results/simulation/communication_comparison.png`

### 3. Check Simulation Results

```bash
cat results/simulation/simulation_results.json
```

Compare max communication times across different imbalance levels.

---

## Common Commands

### Run specific experiment
```bash
python experiments/profile_expert_assignment.py
python experiments/simulate_network.py
```

### Use different model
```bash
python experiments/profile_expert_assignment.py \
    --model mistralai/Mixtral-8x7B-v0.1
```

### Adjust sample size
```bash
python experiments/profile_expert_assignment.py --max_samples 200
```

### Get help
```bash
python experiments/profile_expert_assignment.py --help
```

---

## Minimal Working Example

If you want to test just the core profiling functionality:

```python
from src.profiling.expert_profiler import ExpertAssignmentProfiler

# Initialize
profiler = ExpertAssignmentProfiler("microsoft/Phi-3.5-MoE-instruct")
profiler.load_model()

# Profile
profiler.register_hooks()
stats = profiler.profile_text("Hello, this is a test.")
profiler.remove_hooks()

# View results
summary = profiler.get_imbalance_summary(stats)
print(f"Average Gini: {summary['average_gini']:.3f}")
```

---

## Troubleshooting

### "CUDA out of memory"
- Use smaller batch size: `--batch_size 2`
- Use fewer samples: `--max_samples 50`
- Use CPU: Set `device='cpu'` in code

### "Model not found"
- Check internet connection
- Try: `huggingface-cli login`
- Use different model with `--model`

### Import errors
- Activate venv: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

### Scripts won't run
- Make executable: `chmod +x experiments/*.py`
- Use python directly: `python experiments/...`

---

## Next Steps

After running experiments:

1. **Read the Report**: Check `results/profiling/profiling_report.txt`
2. **Analyze Plots**: Review visualizations in `results/*/plots/`
3. **Compare Results**: Look at simulation vs profiling data
4. **Customize**: Modify parameters and re-run
5. **Write Report**: Document your findings

See `IMPLEMENTATION_GUIDE.md` for detailed documentation.

---

## Project Objectives Checklist

- [ ] Profile expert assignments → `profile_expert_assignment.py`
- [ ] Measure imbalance → Check Gini coefficients
- [ ] Test assignment changes → `test_assignment_changes.py`
- [ ] Fine-tune with loss → `finetune_with_imbalance_loss.py`
- [ ] Simulate network → `simulate_network.py`
- [ ] Analyze results → Review all outputs
- [ ] Document findings → Write report

Good luck with your project! 🚀
