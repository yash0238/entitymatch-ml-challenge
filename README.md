# EntityMatch - Amazon ML Challenge 2026

**Business Entity Resolution at Scale**

## Problem Overview

This project addresses the **Business Entity Resolution Challenge** from Amazon ML Challenge 2026. The task is to match business records across three independent, noisy data sources with no common identifiers.

### Challenge Details

- **Input**: Business records from 3 sources (Source 1 = clean reference, Sources 2 & 3 = noisy external data)
- **Goal**: For each Source 1 business, predict all matching records from Source 2 and Source 3
- **Evaluation Metric**: Macro F0.5 Score (precision-heavy: false merges penalized more than missed matches)
- **Key Constraint**: No external data/APIs allowed (no Google Maps, geocoding, business databases)

### Strategic Approach

The solution follows a **four-stage entity resolution pipeline**:

```
1. Data Profiling & EDA
        ↓
2. Multi-view Blocking (Candidate Generation)
        ↓
3. Pairwise Similarity & Matching Model
        ↓
4. Conservative Decision + Singleton Detection
```

### Key Insights

- **Blocking is critical**: Determines maximum possible recall
- **F0.5 rewards precision**: Avoiding false merges is more important than catching all matches
- **Singletons matter**: Correctly predicting "no match" scores 1.0
- **Cross-country generalization**: Test includes France (not in training)

## Project Structure

```
entitymatch-ml-challenge/
├── data/                          # Data directory (symlink to dataset)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── src/
│   ├── __init__.py
│   ├── data_profiling.py          # EDA and data analysis
│   ├── preprocessing.py           # Text normalization, cleaning
│   ├── blocking.py                # Candidate generation strategies
│   ├── feature_engineering.py     # Similarity features (Jaccard, Levenshtein, etc.)
│   ├── matching_model.py          # ML model for pairwise matching
│   ├── pipeline.py                # End-to-end execution
│   └── utils.py                   # Helper functions
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_blocking_experiments.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── models/                        # Saved model artifacts
├── output/
│   ├── matching_results.tsv       # Final predictions (for leaderboard)
│   └── candidate_pairs.tsv        # Blocking output
├── experiments/                   # Experiment logs and results
├── tests/                         # Unit tests
│   └── test_pipeline.py
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment (alternative)
├── .gitignore
├── Documentation.md               # Methodology writeup
└── README.md                      # This file
```

## Setup Instructions

### 1. Clone and Navigate

```bash
cd entity-resolution-challenge
```

### 2. Create Python Environment

**Option A: Using conda**
```bash
conda env create -f environment.yml
conda activate entity-resolution
```

**Option B: Using venv**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Link Dataset

The dataset is already present in the parent directory. Create a symlink:

**On Windows (PowerShell as Admin):**
```powershell
New-Item -ItemType SymbolicLink -Path "data" -Target "..\dataset\student_resource\dataset"
```

**On Linux/Mac:**
```bash
ln -s ../dataset/student_resource/dataset data
```

## Usage

### Quick Start - Full Pipeline

```bash
python src/pipeline.py --mode train --output output/
```

### Step-by-Step Execution

```bash
# 1. Data profiling and EDA
python src/data_profiling.py

# 2. Generate candidate pairs (blocking)
python src/blocking.py --input data/train --output output/candidate_pairs.tsv

# 3. Train matching model
python src/matching_model.py --train --candidates output/candidate_pairs.tsv

# 4. Generate predictions
python src/pipeline.py --mode predict --test-dir data/test --output output/
```

### Validate Submission

```bash
python ../dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

## Methodology

### Stage 1: Data Profiling
- Analyze noise patterns (abbreviations, typos, transliterations)
- Identify address structure variations across countries
- Measure field completeness and distribution

### Stage 2: Blocking Strategy
High-recall multi-view blocking combining:
- **Phonetic encoding** (Soundex, Metaphone) for name variations
- **Token-based blocking** (TF-IDF on name/address tokens)
- **Geographic blocking** (country + address prefix)
- **N-gram similarity** for fuzzy matching

Target: 95%+ recall with <1% candidate rate

### Stage 3: Feature Engineering
Pairwise features:
- **Name similarity**: Jaccard, Levenshtein, Jaro-Winkler, Token Sort Ratio
- **Address similarity**: Token overlap, edit distance, component matching
- **Contextual**: Country match, legal suffix patterns
- **Semantic**: TF-IDF cosine similarity

### Stage 4: Matching Model
- **Model**: XGBoost/LightGBM classifier (MIT license, efficient)
- **Threshold tuning**: Optimize for F0.5 on validation set
- **Singleton detection**: Conservative threshold + explicit "no match" class

## Development Roadmap

- [ ] Phase 1: EDA and Data Understanding
- [ ] Phase 2: Baseline Blocking Implementation
- [ ] Phase 3: Feature Engineering
- [ ] Phase 4: Model Training & Validation
- [ ] Phase 5: Test Predictions
- [ ] Phase 6: Documentation & Packaging

## Constraints & Rules

- ✅ **Allowed**: Only provided training/test data
- ❌ **Prohibited**: External APIs, databases, geocoding services, Google Maps
- ⚖️ **Model License**: MIT/Apache 2.0, max 8B parameters
- 📝 **Output Format**: TSV files with exact schema

## Evaluation

**Metric**: Macro-averaged F0.5 score

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed per Source 1 entity, then averaged across all entities (including singletons).

## References

- [Amazon ML Challenge Problem Statement](../dataset/student_resource/README.md)
- [String Similarity Metrics for Record Linkage](https://en.wikipedia.org/wiki/Record_linkage)
- [Entity Resolution Best Practices](https://dl.acm.org/doi/10.1145/1341318.1341320)

## License

This project is for the Amazon ML Challenge 2026. Code is MIT licensed.

## Authors

Yashovardhan Thopte

---

**Last Updated**: 2026-09-26
