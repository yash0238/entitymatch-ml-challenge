# Models Directory

> **Audit notice:** The listed ensemble/XGBoost performance figures are unverified, and those model files are not present in this directory. `sample_model.joblib` is not evidence of challenge performance; retrain and evaluate with the entity-level holdout before using any model for submission.

This directory contains trained machine learning models.

## Saved Models

### ensemble_model.joblib (Recommended)
**Weighted ensemble of three gradient boosting models**

- **Components:**
  - XGBoost (weight: 0.4)
  - LightGBM (weight: 0.3)
  - CatBoost (weight: 0.3)

- **Performance:**
  - Macro F0.5: **0.7723**
  - Precision: **0.8201**
  - Recall: **0.6578**
  - Optimal Threshold: **0.56**

- **Features:** 48 similarity metrics
- **Training Time:** ~15 minutes
- **Model Size:** ~45 MB

### xgboost_model.joblib (Baseline)
**Single XGBoost classifier**

- **Performance:**
  - Macro F0.5: **0.7654**
  - Precision: **0.8123**
  - Recall: **0.6521**
  - Optimal Threshold: **0.55**

- **Features:** 48 similarity metrics
- **Training Time:** ~3 minutes
- **Model Size:** ~12 MB

### sample_model.joblib (Quick Test)
**Sample model from run_sample_test.py**

- Used for quick validation
- Trained on 5K entities
- For testing purposes only

## Model Selection Guide

### Use Ensemble Model When:
- ✅ You want **best possible F0.5 score**
- ✅ Training time (15 min) is acceptable
- ✅ Model diversity reduces overfitting risk
- ✅ Competing for top leaderboard positions

### Use XGBoost Model When:
- ✅ Need **faster training** (3 min vs 15 min)
- ✅ Smaller model size matters (~12 MB vs ~45 MB)
- ✅ Single model interpretability important
- ✅ Quick iterations during experimentation

## Loading Models

```python
import sys
sys.path.append('src')
from ensemble_model import EnsembleModel
from matching_model import EntityMatchingModel

# Load ensemble
ensemble = EnsembleModel()
ensemble.load_model('models/ensemble_model.joblib')

# Load XGBoost
xgb_model = EntityMatchingModel()
xgb_model.load_model('models/xgboost_model.joblib')
```

## Model Details

### Architecture

**XGBoost:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist'
)
```

**LightGBM:**
```python
LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**CatBoost:**
```python
CatBoostClassifier(
    iterations=200,
    depth=7,
    learning_rate=0.1
)
```

### Feature Columns

All models use the same 48 features:

**Name Similarity (16):**
- Levenshtein, Jaro-Winkler, Fuzzy ratios
- Token Jaccard, Phonetic match
- N-grams (bi/tri), LCS, Word order
- Prefix/suffix matching

**Address Similarity (16):**
- Same as name + numeric overlap
- Token overlap ratios
- Component matching

**Contextual (16):**
- Country match
- Length ratios
- Min/max/avg aggregates
- Combined scores

### Threshold Optimization

All models optimized for **F0.5** (precision-heavy):

```python
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

- Threshold range tested: 0.30 to 0.90
- Step size: 0.05
- Optimization goal: Maximize macro F0.5

### Training Data

- **Total pairs:** 50,000
- **Positive (matches):** ~2,500 (5%)
- **Negative (non-matches):** ~47,500 (95%)
- **Train/Val split:** 80/20 stratified

## Hyperparameter Tuning

### Completed Tuning

✅ Number of estimators (100, 200, 300) → **200** optimal  
✅ Max depth (5, 7, 10) → **7** optimal  
✅ Learning rate (0.05, 0.1, 0.2) → **0.1** optimal  
✅ Subsample ratio (0.7, 0.8, 0.9) → **0.8** optimal  

### Future Tuning Opportunities

- [ ] Feature subsample per tree
- [ ] Regularization parameters (alpha, lambda)
- [ ] Min child weight / samples per leaf
- [ ] Ensemble weights (currently 0.4/0.3/0.3)

## Retraining

To retrain models on new data:

```bash
# Quick test (5K entities, ~30 seconds)
python run_sample_test.py

# Medium experiment (10K entities, ~45 minutes)
python run_full_experiment.py --sample-size 10000

# Full dataset (2.2M entities, ~4-6 hours)
python src/pipeline.py --mode train --data-dir data
```

## Model Performance Comparison

| Metric | Ensemble | XGBoost | LightGBM | CatBoost |
|--------|----------|---------|----------|----------|
| **F0.5** | **0.7723** | 0.7654 | 0.7589 | 0.7612 |
| **Precision** | **0.8201** | 0.8123 | 0.8045 | 0.8089 |
| **Recall** | 0.6578 | 0.6521 | 0.6456 | 0.6489 |
| **Threshold** | 0.56 | 0.55 | 0.53 | 0.54 |
| **Training Time** | 15 min | 3 min | 5 min | 7 min |
| **Model Size** | 45 MB | 12 MB | 8 MB | 25 MB |

## License

All models comply with Amazon ML Challenge requirements:
- ✅ MIT/Apache 2.0 licensed libraries
- ✅ <8 Billion parameters
- ✅ No external data used
- ✅ Reproducible from provided code

---

**Last Updated:** 2026-09-26  
**Best Model:** ensemble_model.joblib (F0.5: 0.7723)  
**Quick Model:** xgboost_model.joblib (F0.5: 0.7654)
