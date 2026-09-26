"""Run a reproducible, memory-bounded entity-held-out training evaluation."""
import argparse
import heapq
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from preprocessing import TextNormalizer

logger = logging.getLogger(__name__)
STOP_TOKENS = {
    "and", "the", "for", "inc", "corp", "corporation", "company", "co",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services", "service",
    "group", "holdings", "holding", "india", "usa", "united", "states",
}
FEATURES = [
    "name_ratio", "name_token_sort", "name_partial", "name_jaccard",
    "name_overlap", "name_exact", "name_prefix", "addr_ratio",
    "addr_token_sort", "addr_jaccard", "addr_overlap", "addr_exact",
    "number_jaccard", "number_overlap", "shared_name_tokens",
    "shared_address_tokens", "name_length_ratio", "address_length_ratio",
]


def normalize_record(row):
    name = TextNormalizer.normalize_business_name(row.business_name)
    address = TextNormalizer.normalize_address(row.business_address)
    name_tokens = set(name.split())
    address_tokens = set(address.split())
    numbers = set(re.findall(r"\d+", str(row.business_address)))
    block_tokens = {t for t in name_tokens if len(t) >= 4 and t not in STOP_TOKENS}
    block_tokens.update(f"n:{n}" for n in numbers if len(n) >= 4)
    return name, address, name_tokens, address_tokens, numbers, block_tokens


def pair_features(left, right):
    name, address, name_tokens, address_tokens, numbers = left
    other_name, other_address, other_name_tokens, other_address_tokens, other_numbers = right
    shared_names = name_tokens & other_name_tokens
    shared_addresses = address_tokens & other_address_tokens
    number_intersection = numbers & other_numbers
    return [
        fuzz.ratio(name, other_name) / 100,
        fuzz.token_sort_ratio(name, other_name) / 100,
        fuzz.partial_ratio(name, other_name) / 100,
        len(name_tokens & other_name_tokens) / max(1, len(name_tokens | other_name_tokens)),
        len(shared_names) / max(1, min(len(name_tokens), len(other_name_tokens))),
        float(bool(name) and name == other_name),
        float(bool(name) and bool(other_name) and name[:5] == other_name[:5]),
        fuzz.ratio(address, other_address) / 100,
        fuzz.token_sort_ratio(address, other_address) / 100,
        len(address_tokens & other_address_tokens) / max(1, len(address_tokens | other_address_tokens)),
        len(shared_addresses) / max(1, min(len(address_tokens), len(other_address_tokens))),
        float(bool(address) and address == other_address),
        len(number_intersection) / max(1, len(numbers | other_numbers)),
        len(number_intersection),
        len(shared_names),
        len(shared_addresses),
        min(len(name), len(other_name)) / max(1, max(len(name), len(other_name))),
        min(len(address), len(other_address)) / max(1, max(len(address), len(other_address))),
    ]


def macro_f05(entity_ids, labels, group_ids, candidate_ids, probabilities, threshold):
    true_by_entity = {entity_id: set(values) for entity_id, values in labels.items()}
    predicted = {entity_id: set() for entity_id in entity_ids}
    for group_id, candidate_id, probability in zip(group_ids, candidate_ids, probabilities):
        if probability >= threshold:
            predicted[group_id].add(candidate_id)
    scores = []
    for entity_id in entity_ids:
        truth = true_by_entity.get(entity_id, set())
        guess = predicted[entity_id]
        if not truth and not guess:
            scores.append(1.0)
        elif not truth or not guess:
            scores.append(0.0)
        else:
            tp = len(truth & guess)
            if tp == 0:
                scores.append(0.0)
            else:
                precision = tp / len(guess)
                recall = tp / len(truth)
                scores.append(1.25 * precision * recall / (0.25 * precision + recall))
    return float(np.mean(scores)), predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--top-per-source", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    logger.info("Reservoir-sampling %d Source 1 records", args.sample_size)
    reservoir = []
    seen = 0
    for chunk in pd.read_csv(data_dir / "train/train_source1.tsv", sep="\t", chunksize=args.chunk_size):
        for row in chunk.itertuples(index=False):
            seen += 1
            item = (row.entity_id, row.business_name, row.business_address, row.country)
            if len(reservoir) < args.sample_size:
                reservoir.append(item)
            else:
                position = int(rng.integers(seen))
                if position < args.sample_size:
                    reservoir[position] = item
    reference = pd.DataFrame(reservoir, columns=["entity_id", "business_name", "business_address", "country"])
    entity_ids = reference.entity_id.tolist()
    reference["normalized"] = [normalize_record(row)[:5] for row in reference.itertuples(index=False)]
    reference["country_key"] = reference.country.astype(str).str.strip().str.casefold()
    entity_has_match = {}

    selected_ids = set(entity_ids)
    label_sets = {entity_id: set() for entity_id in entity_ids}
    for chunk in pd.read_csv(data_dir / "train/train_ground_truth.tsv", sep="\t", chunksize=args.chunk_size):
        for row in chunk[chunk.source1_entity_id.isin(selected_ids)].itertuples(index=False):
            value = row.matched_entity_ids
            label_sets[row.source1_entity_id] = set() if pd.isna(value) or not str(value).strip() else set(str(value).split(","))
    entity_has_match = np.array([bool(label_sets[entity_id]) for entity_id in entity_ids])
    if len(set(entity_has_match)) != 2:
        raise RuntimeError("Sample must include matched and singleton entities")
    train_ids, remainder = train_test_split(entity_ids, test_size=0.4, random_state=args.seed, stratify=entity_has_match)
    train_has_match = np.array([bool(label_sets[entity_id]) for entity_id in train_ids])
    calibration_ids, test_ids = train_test_split(remainder, test_size=0.5, random_state=args.seed, stratify=[bool(label_sets[entity_id]) for entity_id in remainder])
    logger.info("Entity partitions: train=%d calibration=%d test=%d", len(train_ids), len(calibration_ids), len(test_ids))

    token_to_entities = {}
    normalized_by_entity = {}
    country_by_entity = {}
    for row in reference.itertuples(index=False):
        normalized = row.normalized
        normalized_by_entity[row.entity_id] = normalized
        country_by_entity[row.entity_id] = row.country_key
        for token in normalize_record(row)[5]:
            token_to_entities.setdefault((row.country_key, token), []).append(row.entity_id)

    heaps = {(entity_id, source): [] for entity_id in entity_ids for source in (2, 3)}
    logger.info("Scanning full Source 2/3 files with indexed name/address tokens")
    for source in (2, 3):
        path = data_dir / f"train/train_source{source}.tsv"
        total = 0
        for chunk in pd.read_csv(path, sep="\t", chunksize=args.chunk_size):
            for row in chunk.itertuples(index=False):
                total += 1
                country = str(row.country).strip().casefold()
                normalized = normalize_record(row)
                if not normalized[5]:
                    continue
                potential = set()
                for token in normalized[5]:
                    potential.update(token_to_entities.get((country, token), ()))
                for entity_id in potential:
                    name, address, name_tokens, address_tokens, numbers = normalized_by_entity[entity_id]
                    shared_name = len(name_tokens & normalized[2])
                    shared_addr = len(address_tokens & normalized[3])
                    shared_number = len(numbers & normalized[4])
                    coarse = shared_name * 10 + shared_addr * 2 + min(shared_number, 3)
                    if coarse == 0:
                        continue
                    heap = heaps[(entity_id, source)]
                    item = (coarse, row.entity_id, normalized[:5])
                    if len(heap) < args.top_per_source:
                        heapq.heappush(heap, item)
                    elif coarse > heap[0][0]:
                        heapq.heapreplace(heap, item)
            if total % (args.chunk_size * 20) == 0:
                logger.info("Source %d scanned: %d records", source, total)
        logger.info("Source %d complete: %d records", source, total)

    global candidate_ids
    candidate_ids = []
    pair_rows = []
    candidate_count_by_entity = {}
    for entity_id in entity_ids:
        count = 0
        left = normalized_by_entity[entity_id]
        for source in (2, 3):
            for _, candidate_id, right in heaps[(entity_id, source)]:
                pair_rows.append((entity_id, candidate_id, pair_features(left, right)))
                candidate_ids.append(candidate_id)
                count += 1
        candidate_count_by_entity[entity_id] = count

    candidate_frame = pd.DataFrame(pair_rows, columns=["group", "candidate", "features"])
    if candidate_frame.empty:
        raise RuntimeError("Blocking generated no candidates")
    feature_matrix = np.asarray(candidate_frame.features.tolist(), dtype=np.float32)
    group_ids = candidate_frame.group.to_numpy()
    candidate_ids = candidate_frame.candidate.to_numpy()
    y = np.fromiter((candidate_id in label_sets[group_id] for group_id, candidate_id in zip(group_ids, candidate_ids)), dtype=np.uint8)
    cache_path = Path(f"experiments/verified_sample_candidates_top{args.top_per_source}.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feature_matrix, groups=group_ids, candidates=candidate_ids, labels=y)
    logger.info("Saved candidate checkpoint to %s", cache_path)

    train_mask = np.isin(group_ids, train_ids)
    calibration_mask = np.isin(group_ids, calibration_ids)
    test_mask = np.isin(group_ids, test_ids)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=args.seed)
    model.fit(feature_matrix[train_mask], y[train_mask])
    calibration_probabilities = model.predict_proba(feature_matrix[calibration_mask])[:, 1]
    test_probabilities = model.predict_proba(feature_matrix[test_mask])[:, 1]
    calibration_groups = group_ids[calibration_mask]
    calibration_candidates = candidate_ids[calibration_mask]
    test_groups = group_ids[test_mask]
    test_candidates = candidate_ids[test_mask]

    best_threshold, best_calibration_score = 1.0, -1.0
    for threshold in np.unique(np.quantile(calibration_probabilities, np.linspace(0, 1, 201))):
        score, _ = macro_f05(calibration_ids, label_sets, calibration_groups, calibration_candidates, calibration_probabilities, threshold)
        if score > best_calibration_score or (np.isclose(score, best_calibration_score) and threshold > best_threshold):
            best_threshold, best_calibration_score = float(threshold), score
    test_score, test_predictions = macro_f05(test_ids, label_sets, test_groups, test_candidates, test_probabilities, best_threshold)

    block_recall = sum(
        len(label_sets[entity_id] & {candidate for key in ((entity_id, 2), (entity_id, 3)) for _, candidate, _ in heaps[key]})
        for entity_id in entity_ids
    ) / max(1, sum(map(len, label_sets.values())))
    result = {
        "method": "full-source token blocking; entity-disjoint train/calibration/test; HistGradientBoosting; macro F0.5 threshold tuned on calibration only",
        "sample_size": len(entity_ids),
        "partitions": {"train": len(train_ids), "calibration": len(calibration_ids), "test": len(test_ids)},
        "candidates": int(len(candidate_frame)),
        "mean_candidates_per_entity": float(np.mean(list(candidate_count_by_entity.values()))),
        "candidate_recall_on_sample": float(block_recall),
        "positive_candidate_pairs": int(y.sum()),
        "train_positive_pairs": int(y[train_mask].sum()),
        "calibration_macro_f05": float(best_calibration_score),
        "selected_threshold": best_threshold,
        "test_macro_f05": test_score,
        "test_singletons": int(sum(not label_sets[entity_id] for entity_id in test_ids)),
        "test_singleton_accuracy": float(np.mean([not test_predictions[entity_id] for entity_id in test_ids if not label_sets[entity_id]])) if any(not label_sets[entity_id] for entity_id in test_ids) else None,
        "test_precision_macro": float(np.mean([len(test_predictions[e] & label_sets[e]) / len(test_predictions[e]) if test_predictions[e] else (1.0 if not label_sets[e] else 0.0) for e in test_ids])),
        "test_recall_macro": float(np.mean([len(test_predictions[e] & label_sets[e]) / len(label_sets[e]) if label_sets[e] else (1.0 if not test_predictions[e] else 0.0) for e in test_ids])),
        "seed": args.seed,
    }
    output_path = Path(args.output or f"experiments/verified_sample_evaluation_top{args.top_per_source}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()