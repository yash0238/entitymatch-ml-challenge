# Dataset Statistics Summary

## Dataset Scale

**Total Records:** ~12.5 Million records across 3 sources

### Source Breakdown

| Source | Records | Purpose |
|--------|---------|---------|
| Source 1 | 2,206,821 | Clean reference (deduplicated) |
| Source 2 | 5,034,616 | Noisy external data |
| Source 3 | 5,285,603 | Noisy external data |

### Country Distribution

**Training Data:**
- **US**: ~7.5M records (60%)
  - Source 1: 1,323,633
  - Source 2: 3,016,817
  - Source 3: 3,170,056

- **India**: ~5M records (40%)
  - Source 1: 883,188
  - Source 2: 2,017,799
  - Source 3: 2,115,547

**Note:** Test set includes France (not seen in training)

## Ground Truth Analysis

### Match Statistics

- **Total S1 Entities:** 2,206,821
- **Singletons (no matches):** 123,247 (5.6%)
- **Entities with matches:** 2,083,574 (94.4%)
- **Average matches per entity:** 3.46
- **Maximum matches:** 11

### Match Distribution

```
Matches | Count    | Percentage
--------|----------|------------
0       | 123,247  | 5.6%
1       | 119,157  | 5.4%
2       | 375,212  | 17.0%
3       | 530,841  | 24.1%  ← Most common
4       | 484,115  | 21.9%
5       | 321,957  | 14.6%
6       | 164,868  | 7.5%
7       | 63,968   | 2.9%
8       | 18,680   | 0.8%
9       | 4,205    | 0.2%
10      | 534      | 0.02%
11      | 37       | 0.002%
```

**Key Insight:** Most entities (24%) have exactly 3 matches across S2 and S3.

## Data Quality

### Missing Data

**Source 2:**
- Missing business_name: 2 records
- Missing business_address: 168,967 records (3.4%)

**Source 3:**
- Missing business_name: 13 records
- Missing business_address: 175,916 records (3.3%)

**Source 1:** No missing data (clean reference)

### Business Name Patterns

- **Total unique names:** 9,935,722 (79% uniqueness)
- **Average name length:** 25 characters

**Most Common Business Suffixes:**
1. limited (1.4M occurrences)
2. llc (1.1M)
3. inc (831K)
4. ltd (808K)
5. center (313K)
6. लिमिटेड [Hindi: Limited] (297K)
7. corp (250K)
8. co (227K)
9. private (225K)
10. services (200K)

### Address Patterns

- **Total unique addresses:** 10,929,926 (87% uniqueness)
- **Average address length:** 48.8 characters
- **Addresses with numbers:** 91.7% (ZIP codes, house numbers)
- **Addresses with commas:** 97.2% (component separators)

## Challenge Implications

### 1. Blocking is Critical
With **10+ million potential comparisons** per S1 entity (2.2M × 10M = 22 trillion pairs), effective blocking is mandatory. Target: reduce to <1% candidate pairs while maintaining >95% recall.

### 2. Precision Matters (F0.5)
- F0.5 metric weights precision 2× over recall
- **False merges are costly** - especially for singletons
- Correctly predicting 123K singletons = guaranteed 123K perfect scores

### 3. Suffix Normalization Essential
- 1.4M "limited", 1.1M "llc", 831K "inc" variations
- Must normalize: Corporation→corp, Limited→ltd, LLC→llc
- Hindi/language-specific suffixes need handling

### 4. Address Noise
- 3.4% missing addresses in noisy sources
- Must handle partial matches and component reordering
- 92% have numeric components (strong signal for matching)

### 5. Cross-Country Generalization
- France appears in test but not training
- Cannot hard-code country-specific rules
- Feature-based learning required

## Recommended Strategy

```
1. Multi-View Blocking (target 95%+ recall, <1% pairs)
   - Phonetic encoding (Soundex/Metaphone) for name variations
   - Token-based indexing for partial name matches
   - TF-IDF cosine similarity for fuzzy matching
   - Country filtering (hard constraint)

2. Rich Feature Engineering (30+ features)
   - Name: Levenshtein, Jaro-Winkler, Jaccard, fuzzy ratios
   - Address: Token overlap, numeric component matching
   - Contextual: Length ratios, suffix patterns

3. XGBoost Classifier
   - Binary classification (match / no-match)
   - F0.5-optimized threshold (favor precision)
   - Handle class imbalance

4. Conservative Decision Making
   - Higher threshold for singletons
   - Explicit "no match" class
   - Avoid false merges at all costs
```

## Performance Estimates

**With current dataset size:**
- Blocking: ~30-60 minutes
- Feature extraction: ~2-3 hours
- Model training: ~15-30 minutes
- Inference: ~1-2 hours

**Memory requirements:** 16GB+ recommended (8GB minimum)

---

**Generated:** 2026-09-26  
**Data Version:** Amazon ML Challenge 2026 Training Set
