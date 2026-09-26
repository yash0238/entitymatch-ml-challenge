"""Calibrate candidate models against a saved sample candidate checkpoint."""
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def macro_f05(entity_ids, labels, groups, candidates, probabilities, threshold):
    predictions = {entity_id: set() for entity_id in entity_ids}
    for group, candidate, probability in zip(groups, candidates, probabilities):
        if probability >= threshold:
            predictions[group].add(candidate)
    scores = []
    for entity_id in entity_ids:
        truth, guessed = labels[entity_id], predictions[entity_id]
        if not truth and not guessed:
            scores.append(1.0)
        elif not truth or not guessed:
            scores.append(0.0)
        else:
            true_positives = len(truth & guessed)
            if true_positives == 0:
                scores.append(0.0)
            else:
                precision = true_positives / len(guessed)
                recall = true_positives / len(truth)
                scores.append(1.25 * precision * recall / (0.25 * precision + recall))
    return float(np.mean(scores)), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="experiments/verified_sample_candidates_top40.npz")
    parser.add_argument("--output", default="experiments/verified_sample_model_sweep_top40.json")
    args = parser.parse_args()
    seed = 42
    sample_size = 1200
    rng = np.random.default_rng(seed)
    reservoir, seen = [], 0
    for chunk in pd.read_csv("data/train/train_source1.tsv", sep="\t", chunksize=25000):
        for row in chunk.itertuples(index=False):
            seen += 1
            item = (row.entity_id, row.business_name, row.business_address, row.country)
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                position = int(rng.integers(seen))
                if position < sample_size:
                    reservoir[position] = item
    entity_ids = [row[0] for row in reservoir]
    selected = set(entity_ids)
    labels = {entity_id: set() for entity_id in entity_ids}
    for chunk in pd.read_csv("data/train/train_ground_truth.tsv", sep="\t", chunksize=25000):
        for row in chunk[chunk.source1_entity_id.isin(selected)].itertuples(index=False):
            value = row.matched_entity_ids
            labels[row.source1_entity_id] = set() if pd.isna(value) or not str(value).strip() else set(str(value).split(","))
    entity_flags = [bool(labels[entity_id]) for entity_id in entity_ids]
    train_ids, remainder = train_test_split(entity_ids, test_size=0.4, random_state=seed, stratify=entity_flags)
    calibration_ids, test_ids = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=[bool(labels[entity_id]) for entity_id in remainder],
    )

    cache = np.load(args.cache, allow_pickle=True)
    features = cache["features"]
    groups = cache["groups"].astype(str)
    candidates = cache["candidates"].astype(str)
    pair_labels = cache["labels"].astype(np.uint8)
    assert set(groups).issubset(selected)
    train_mask = np.isin(groups, train_ids)
    calibration_mask = np.isin(groups, calibration_ids)
    test_mask = np.isin(groups, test_ids)

    model_configs = [
        ("hist_180_l15", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)),
        ("hist_300_l31", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=2.0, min_samples_leaf=25, random_state=seed)),
        ("xgb_350_d6", XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.06, min_child_weight=4, subsample=0.85, colsample_bytree=0.9, reg_lambda=5, tree_method="hist", n_jobs=4, eval_metric="logloss", random_state=seed)),
        ("xgb_500_d5", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.045, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, reg_lambda=8, tree_method="hist", n_jobs=4, eval_metric="logloss", random_state=seed)),
    ]
    sweep, best = [], None
    for name, model in model_configs:
        model.fit(features[train_mask], pair_labels[train_mask])
        probabilities = model.predict_proba(features[calibration_mask])[:, 1]
        calibration_groups = groups[calibration_mask]
        calibration_candidates = candidates[calibration_mask]
        best_score, best_threshold = -1.0, 1.0
        thresholds = np.unique(np.quantile(probabilities, np.linspace(0, 1, 201)))
        for threshold in thresholds:
            score, _ = macro_f05(calibration_ids, labels, calibration_groups, calibration_candidates, probabilities, threshold)
            if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
                best_score, best_threshold = score, float(threshold)
        sweep.append({"model": name, "calibration_macro_f05": best_score, "threshold": best_threshold})
        logger.info("%s calibration macro F0.5 %.5f at %.5f", name, best_score, best_threshold)
        if best is None or best_score > best["score"]:
            best = {"name": name, "score": best_score, "threshold": best_threshold, "model": model}

    test_probabilities = best["model"].predict_proba(features[test_mask])[:, 1]
    test_score, predictions = macro_f05(test_ids, labels, groups[test_mask], candidates[test_mask], test_probabilities, best["threshold"])
    precision = []
    recall = []
    for entity_id in test_ids:
        truth, guessed = labels[entity_id], predictions[entity_id]
        precision.append(len(truth & guessed) / len(guessed) if guessed else (1.0 if not truth else 0.0))
        recall.append(len(truth & guessed) / len(truth) if truth else (1.0 if not guessed else 0.0))
    all_true_pairs = sum(map(len, labels.values()))
    candidate_true_pairs = sum(candidate in labels[group] for group, candidate in zip(groups, candidates))
    singleton_ids = [entity_id for entity_id in test_ids if not labels[entity_id]]
    result = {
        "method": "full-source sparse token blocking; entity-disjoint train/calibration/test; model selected using calibration macro F0.5",
        "sample_size": sample_size,
        "partitions": {"train": len(train_ids), "calibration": len(calibration_ids), "test": len(test_ids)},
        "candidate_count": int(len(features)),
        "positive_candidate_pairs": int(pair_labels.sum()),
        "candidate_recall": float(candidate_true_pairs / max(1, all_true_pairs)),
        "calibration_sweep": sweep,
        "selected_model": best["name"],
        "selected_threshold": best["threshold"],
        "selected_calibration_macro_f05": best["score"],
        "test_macro_f05": test_score,
        "test_macro_precision": float(np.mean(precision)),
        "test_macro_recall": float(np.mean(recall)),
        "test_singleton_count": len(singleton_ids),
        "test_singleton_accuracy": float(np.mean([not predictions[entity_id] for entity_id in singleton_ids])) if singleton_ids else None,
        "seed": seed,
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()