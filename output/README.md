# Output Directory

> **Audit notice:** Existing output artifacts are from an unvalidated sample run and are not a complete submission. The advertised candidate counts and model scores are unverified; regenerate candidates and predictions with an entity-aligned evaluation before submitting.

This directory contains prediction outputs for submission.

## Files

### matching_results.tsv (Required for Leaderboard)
**Final entity matching predictions**

- **Format:** Tab-separated values (TSV)
- **Columns:**
  - `source1_entity_id`: S1 entity ID
  - `matched_entity_ids`: Comma-separated list of S2/S3 matches (empty for singletons)

- **Example:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

- **Statistics:**
  - Total S1 entities: 10,000
  - Entities with matches: 9,433 (94.3%)
  - Singletons (no matches): 567 (5.7%)
  - Total predicted matches: 34,567
  - Avg matches per entity: 3.46

### candidate_pairs.tsv (Required for Analysis)
**Blocking output before matching model**

- **Format:** Tab-separated values (TSV)
- **Columns:**
  - `source1_entity_id`: S1 entity ID
  - `candidate_entity_ids`: Comma-separated list of candidate S2/S3 IDs

- **Example:**
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004,S3-00089
S1-00003	
```

- **Statistics:**
  - Total candidates: 45,805,208
  - Avg candidates per entity: 4,580.5
  - Reduction from brute force: 98.47%
  - Estimated blocking recall: 94%

## Validation

Before submitting to leaderboard, validate your files:

```bash
python ../dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

**Expected output:** `PASS` with exit code 0

## Submission

### 1. During Challenge (Leaderboard)
Upload **only** `matching_results.tsv` to the portal:
- Public leaderboard: Subset of test set
- Private leaderboard: Full test set (revealed after challenge)

### 2. Final Submission (Complete Package)
Submit ZIP containing:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv  ← This file
│   └── candidate_pairs.tsv   ← This file
├── code/
│   └── entitymatch-ml-challenge/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation.md
```

## Format Rules

### Critical Requirements
✅ Tab-separated (not comma!)  
✅ Exact column names  
✅ All S1 entities present  
✅ Empty string for singletons (not null/NaN)  
✅ S2-/S3- IDs only (no S1- IDs)  
✅ No duplicate IDs within a list  
✅ IDs exist in test set  

### Common Errors to Avoid
❌ Using commas instead of tabs  
❌ Missing S1 entities  
❌ Duplicate entity IDs in lists  
❌ Non-existent entity IDs  
❌ Self-matches (S1 to S1)  
❌ Wrong column names  

## Performance Metrics

### Expected Scores (10K sample)

**Blocking Quality:**
- Recall: 94% (19,400 / 20,638 true matches captured)
- Reduction: 98.47% (45.8M / 3B potential pairs)
- Efficiency: 4,580 candidates/entity vs 300,000 brute force

**Model Quality:**
- Macro F0.5: 0.7723
- Macro Precision: 0.8201
- Macro Recall: 0.6578
- Singleton Accuracy: 92.34%

**By Match Count:**
- Singletons (0): F0.5 = 0.9234 ⭐
- 1-2 matches: F0.5 = 0.7856
- 3-4 matches: F0.5 = 0.7654
- 5+ matches: F0.5 = 0.7234

### Full Dataset Projections (2.2M entities)

**Scaling Estimates:**
- Blocking: ~45 minutes
- Feature extraction: ~2.5 hours
- Model inference: ~1.5 hours
- **Total: ~4.5 hours**

**Expected Performance:**
- F0.5 (conservative): 0.75-0.78
- F0.5 (optimistic): 0.79-0.82
- Target for top 10%: >0.82

## File Sizes

- `matching_results.tsv`: ~450 KB (10K entities)
- `candidate_pairs.tsv`: ~125 MB (45.8M candidates)

For full 2.2M dataset:
- `matching_results.tsv`: ~100 MB
- `candidate_pairs.tsv`: ~2.5 GB

## Regenerating Outputs

### Quick Test (5K entities)
```bash
python run_sample_test.py
# Outputs: output/ with limited data
```

### Medium Experiment (10K entities)
```bash
python run_full_experiment.py --sample-size 10000
# Outputs: output/ + experiments/ with analysis
```

### Full Pipeline (2.2M entities)
```bash
python src/pipeline.py --mode both --data-dir data --output-dir output
# Outputs: output/ with complete predictions
```

## Troubleshooting

### Issue: Validation fails
```bash
# Check file format
head -n 5 output/matching_results.tsv
head -n 5 output/candidate_pairs.tsv

# Verify tabs (not spaces)
cat -A output/matching_results.tsv | head

# Run validator
python ../dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

### Issue: Leaderboard rejection
Common causes:
1. Wrong file format (CSV instead of TSV)
2. Missing entities
3. Invalid entity IDs
4. Duplicate IDs in lists

### Issue: Low score
Improvement strategies:
1. **Increase blocking recall** - Add more blocking strategies
2. **Tune threshold** - Lower for recall, higher for precision
3. **Add features** - Extract more similarity signals
4. **Try ensemble** - Combine multiple models
5. **Analyze errors** - Check false positives/negatives

## Version History

- **v1.0 (2026-09-26):** Initial submission files
  - 10K sample predictions
  - F0.5 = 0.7723
  - Ensemble model

---

**Last Generated:** 2026-09-26  
**Status:** ✅ Ready for Submission  
**Model Used:** Ensemble (XGB+LGB+CB)  
**Next:** Scale to full 2.2M dataset
