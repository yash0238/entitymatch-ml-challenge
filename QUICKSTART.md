# Quick Start Guide

This guide will get you up and running with the Amazon ML Challenge 2026 Entity Resolution project.

## Prerequisites

- Python 3.10 or higher
- Git
- 8GB+ RAM recommended
- Windows/Linux/Mac

## Step 1: Navigate to Project

```powershell
cd "c:\Users\DELL\Desktop\Amazon ML\entity-resolution-challenge"
```

## Step 2: Create Data Link

The dataset is in the parent `dataset` folder. Create a symbolic link:

**On Windows (Run PowerShell as Administrator):**
```powershell
New-Item -ItemType SymbolicLink -Path "data" -Target "..\dataset\student_resource\dataset"
```

**On Linux/Mac:**
```bash
ln -s ../dataset/student_resource/dataset data
```

**Alternative (if symlink fails):** Copy the dataset folder:
```powershell
Copy-Item -Recurse "..\dataset\student_resource\dataset" -Destination "data"
```

## Step 3: Set Up Python Environment

### Option A: Using Conda (Recommended)

```powershell
conda env create -f environment.yml
conda activate entity-resolution
```

### Option B: Using venv

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows
# source venv/bin/activate    # On Linux/Mac
pip install -r requirements.txt
```

## Step 4: Run Data Profiling (Optional but Recommended)

Understand your data first:

```powershell
python src/data_profiling.py
```

This will show:
- Dataset statistics
- Noise patterns
- Match distributions
- Country breakdowns

## Step 5: Run Complete Pipeline

### Train and Predict in One Command

```powershell
python src/pipeline.py --mode both --data-dir data --output-dir output
```

This will:
1. Generate training candidate pairs (blocking)
2. Extract features
3. Train XGBoost model
4. Generate test candidate pairs
5. Predict matches
6. Save `matching_results.tsv` and `candidate_pairs.tsv` to `output/`

### Or Run Stages Separately

**Training only:**
```powershell
python src/pipeline.py --mode train --data-dir data
```

**Prediction only:**
```powershell
python src/pipeline.py --mode predict --data-dir data
```

## Step 6: Validate Submission

Before submitting to the leaderboard:

```powershell
python ..\dataset\student_resource\utils\validate_submission.py `
    --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv `
    --test-dir data\test
```

✅ You should see `PASS` with exit code 0.

## Step 7: Submit to Leaderboard

1. Go to the Amazon ML Challenge portal
2. Upload `output/matching_results.tsv`
3. Check your score!

## Project Structure

```
entity-resolution-challenge/
├── data/                    # Symlink to dataset (create this!)
├── src/                     # Source code
│   ├── pipeline.py          # Main entry point
│   ├── blocking.py          # Candidate generation
│   ├── feature_engineering.py
│   ├── matching_model.py
│   └── ...
├── output/                  # Results (generated)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── models/                  # Saved models (generated)
├── experiments/             # Experiment logs
└── tests/                   # Unit tests
```

## Troubleshooting

### Issue: `No module named 'src'`

**Solution:** Make sure you're in the project root directory.

### Issue: `FileNotFoundError: data/train/...`

**Solution:** Create the data symlink (Step 2).

### Issue: Import errors

**Solution:** Install dependencies:
```powershell
pip install -r requirements.txt
```

### Issue: Memory errors

**Solution:** Reduce TF-IDF top_k in `src/blocking.py`:
```python
# Line ~140
self._tfidf_block(..., top_k=20, threshold=0.4)  # Reduce from 50
```

## Running Tests

```powershell
pytest tests/ -v
```

Or:
```powershell
python tests/test_pipeline.py
```

## Next Steps

1. **Experiment with blocking strategies** in `src/blocking.py`
2. **Add more features** in `src/feature_engineering.py`
3. **Tune model hyperparameters** in `src/matching_model.py`
4. **Analyze errors** by comparing predictions with validation ground truth
5. **Iterate and improve** your F0.5 score!

## Tips for Improvement

- **Blocking recall is critical**: If true matches aren't in candidates, model can't recover them
- **Feature engineering matters**: Try name component matching, n-gram features
- **Threshold tuning**: Adjust for F0.5 (precision-heavy) vs F1
- **Singleton detection**: Don't forget empty match lists score perfectly when correct
- **Country-specific patterns**: India vs US vs France may need different handling

## Resources

- [Problem Statement](../dataset/student_resource/README.md)
- [Documentation Template](Documentation.md)
- [Full README](README.md)

---

**Need help?** Check the [README.md](README.md) or review the code comments.

Good luck! 🚀
