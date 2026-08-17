"""Deterministic condition-by-gender-stratified speaker evaluation folds."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vaaniswitch.m0_data.config import TorgoConfig
from vaaniswitch.m0_data.schemas import (
    Condition,
    Gender,
    ManifestRecord,
)


class SplitError(RuntimeError):
    """Raised when speaker-group folds cannot satisfy leakage constraints."""


@dataclass(frozen=True, slots=True)
class SpeakerStats:
    """Metadata and record counts used only to balance whole speakers."""

    speaker_id: str
    condition: Condition
    gender: Gender
    total_records: int
    balance_records: int

    @property
    def stratum(self) -> tuple[str, str]:
        return self.condition.value, self.gender.value


def create_speaker_folds(
    records: Sequence[ManifestRecord],
    config: TorgoConfig,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Create the largest feasible fold count preserving all hard strata."""

    if not records:
        raise SplitError("cannot create folds from an empty manifest")
    speakers = _aggregate_speakers(records, config)
    strata: dict[tuple[str, str], list[SpeakerStats]] = defaultdict(list)
    for speaker in speakers:
        strata[speaker.stratum].append(speaker)
    for values in strata.values():
        values.sort(key=lambda item: item.speaker_id)

    if not strata:
        raise SplitError("no speaker strata were found")
    smallest_stratum = min(len(values) for values in strata.values())
    selected_folds = min(config.evaluation.requested_max_folds, smallest_stratum)
    if selected_folds < config.evaluation.minimum_folds:
        raise SplitError(
            "insufficient speakers per condition-by-gender stratum: "
            f"minimum={smallest_stratum}, required={config.evaluation.minimum_folds}"
        )

    assignment = _optimize_partition(
        strata,
        fold_count=selected_folds,
        seed=config.evaluation.seed,
    )
    all_speakers = sorted(assignment)
    folds: list[dict[str, Any]] = []
    for fold_index in range(selected_folds):
        evaluation_speakers = sorted(
            speaker for speaker, assigned_fold in assignment.items() if assigned_fold == fold_index
        )
        evaluation_set = set(evaluation_speakers)
        train_speakers = [speaker for speaker in all_speakers if speaker not in evaluation_set]
        folds.append(
            {
                "fold_id": fold_index + 1,
                "train_speakers": train_speakers,
                "evaluation_speakers": evaluation_speakers,
                "train_counts": _partition_counts(records, set(train_speakers), config),
                "evaluation_counts": _partition_counts(
                    records,
                    evaluation_set,
                    config,
                ),
            }
        )

    requested = config.evaluation.requested_max_folds
    if selected_folds < requested:
        selection_reason = (
            f"Selected {selected_folds} folds instead of {requested}: {selected_folds} is the "
            "largest count for which every evaluation fold contains all observed "
            "condition-by-gender strata."
        )
    else:
        selection_reason = (
            f"Selected the requested {selected_folds} folds while preserving every observed "
            "condition-by-gender stratum."
        )

    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": config.dataset_id,
        "strategy": {
            "name": "condition_gender_stratified_speaker_group_cv",
            "requested_max_folds": requested,
            "selected_folds": selected_folds,
            "minimum_folds": config.evaluation.minimum_folds,
            "seed": config.evaluation.seed,
            "group_key": config.evaluation.group_by,
            "hard_strata": list(config.evaluation.hard_strata),
            "balance_measure": "non_excluded_primary_logical_records",
            "balance_statuses": sorted(
                status.value for status in config.evaluation.balance_statuses
            ),
            "selection_reason": selection_reason,
            "stratum_speaker_counts": {
                f"{condition}|{gender}": len(values)
                for (condition, gender), values in sorted(strata.items())
            },
        },
        "source": {
            "manifest_path": config.relative_path(config.outputs["manifest_jsonl"]),
            "manifest_sha256": manifest_sha256,
            "logical_record_count": len(records),
        },
        "speaker_to_evaluation_fold": {
            speaker: fold_index + 1 for speaker, fold_index in sorted(assignment.items())
        },
        "folds": folds,
    }
    document["validation"] = validate_speaker_folds(records, document)
    return document


def validate_speaker_folds(
    records: Sequence[ManifestRecord],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Validate speaker, logical-record, and microphone-path disjointness."""

    speakers = {record.speaker_id for record in records}
    folds_value = document.get("folds")
    if not isinstance(folds_value, list) or not folds_value:
        raise SplitError("split document has no folds")
    fold_count = len(folds_value)
    map_value = document.get("speaker_to_evaluation_fold")
    if not isinstance(map_value, dict):
        raise SplitError("split document has no speaker_to_evaluation_fold mapping")
    if set(map_value) != speakers:
        raise SplitError("speaker-to-fold mapping does not cover exactly all speakers")

    evaluation_occurrences: Counter[str] = Counter()
    logical_evaluation_occurrences: Counter[str] = Counter()
    expected_speaker_map: dict[str, int] = {}
    fold_results: list[dict[str, Any]] = []
    condition_supported = True
    stratum_supported = True
    for expected_index, fold in enumerate(folds_value, start=1):
        if not isinstance(fold, dict) or fold.get("fold_id") != expected_index:
            raise SplitError("fold IDs must be consecutive and one-indexed")
        train = _string_set(fold.get("train_speakers"), "train_speakers")
        evaluation = _string_set(
            fold.get("evaluation_speakers"),
            "evaluation_speakers",
        )
        overlap = train & evaluation
        if overlap:
            raise SplitError(f"speaker leakage in fold {expected_index}: {sorted(overlap)}")
        if train | evaluation != speakers:
            raise SplitError(f"fold {expected_index} does not cover all speakers")
        if train != speakers - evaluation:
            raise SplitError(f"fold {expected_index} train set is not eval complement")

        evaluation_occurrences.update(evaluation)
        expected_speaker_map.update({speaker_id: expected_index for speaker_id in evaluation})
        train_records = [record for record in records if record.speaker_id in train]
        evaluation_records = [record for record in records if record.speaker_id in evaluation]
        logical_evaluation_occurrences.update(record.logical_id for record in evaluation_records)
        train_logical = {record.logical_id for record in train_records}
        evaluation_logical = {record.logical_id for record in evaluation_records}
        logical_overlap = train_logical & evaluation_logical
        if logical_overlap:
            raise SplitError(
                f"logical ID leakage in fold {expected_index}: {sorted(logical_overlap)}"
            )

        train_audio = _audio_paths(train_records)
        evaluation_audio = _audio_paths(evaluation_records)
        audio_overlap = train_audio & evaluation_audio
        if audio_overlap:
            raise SplitError(
                f"microphone path leakage in fold {expected_index}: {sorted(audio_overlap)}"
            )

        conditions = {record.condition.value for record in evaluation_records}
        strata = {(record.condition.value, record.gender.value) for record in evaluation_records}
        has_conditions = conditions == {
            Condition.DYSARTHRIC.value,
            Condition.CONTROL.value,
        }
        expected_strata = {(record.condition.value, record.gender.value) for record in records}
        has_strata = strata == expected_strata
        condition_supported &= has_conditions
        stratum_supported &= has_strata
        fold_results.append(
            {
                "fold_id": expected_index,
                "speaker_overlap_count": 0,
                "logical_id_overlap_count": 0,
                "audio_path_overlap_count": 0,
                "both_conditions_present": has_conditions,
                "all_condition_gender_strata_present": has_strata,
            }
        )

    if any(count != 1 for count in evaluation_occurrences.values()):
        raise SplitError("each speaker must be evaluated exactly once")
    if set(evaluation_occurrences) != speakers:
        raise SplitError("not every speaker appears in an evaluation fold")
    if map_value != expected_speaker_map:
        raise SplitError("speaker_to_evaluation_fold does not match fold evaluation membership")
    if any(count != 1 for count in logical_evaluation_occurrences.values()):
        raise SplitError("each logical record must be evaluated exactly once")
    if len(logical_evaluation_occurrences) != len(records):
        raise SplitError("not every logical record appears in evaluation exactly once")
    if not condition_supported:
        raise SplitError("at least one evaluation fold lacks a condition")
    if not stratum_supported:
        raise SplitError("at least one evaluation fold lacks a condition-by-gender stratum")

    return {
        "speaker_disjoint": True,
        "all_speakers_evaluated_once": True,
        "all_logical_records_evaluated_once": True,
        "microphone_paths_disjoint": True,
        "condition_supported_in_every_fold": True,
        "condition_gender_supported_in_every_fold": True,
        "fold_count": fold_count,
        "folds": fold_results,
    }


def _aggregate_speakers(
    records: Sequence[ManifestRecord],
    config: TorgoConfig,
) -> list[SpeakerStats]:
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record)

    result: list[SpeakerStats] = []
    for speaker_id, speaker_records in sorted(grouped.items()):
        conditions = {record.condition for record in speaker_records}
        genders = {record.gender for record in speaker_records}
        groups = {record.speaker_group for record in speaker_records}
        if len(conditions) != 1 or len(genders) != 1 or len(groups) != 1:
            raise SplitError(f"inconsistent metadata for speaker {speaker_id}")
        result.append(
            SpeakerStats(
                speaker_id=speaker_id,
                condition=next(iter(conditions)),
                gender=next(iter(genders)),
                total_records=len(speaker_records),
                balance_records=sum(
                    record.usability_status in config.evaluation.balance_statuses
                    for record in speaker_records
                ),
            )
        )
    return result


def _optimize_partition(
    strata: dict[tuple[str, str], list[SpeakerStats]],
    *,
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    ordered_strata = sorted(strata.items())
    option_sets: list[list[tuple[int, ...]]] = []
    for index, (_, speakers) in enumerate(ordered_strata):
        if index == 0 and len(speakers) == fold_count:
            option_sets.append([tuple(range(fold_count))])
        else:
            option_sets.append(_balanced_onto_assignments(len(speakers), fold_count))

    total_by_condition: Counter[str] = Counter()
    for values in strata.values():
        for speaker in values:
            total_by_condition[speaker.condition.value] += speaker.balance_records
    condition_targets = {
        condition: total / fold_count for condition, total in total_by_condition.items()
    }
    total_balance = sum(total_by_condition.values())
    total_target = total_balance / fold_count
    speaker_target = sum(len(values) for values in strata.values()) / fold_count

    best_score: tuple[float | int | str, ...] | None = None
    best_assignment: dict[str, int] | None = None
    for selected_options in itertools.product(*option_sets):
        assignment: dict[str, int] = {}
        for (_, speakers), option in zip(ordered_strata, selected_options, strict=True):
            assignment.update(
                {speaker.speaker_id: fold for speaker, fold in zip(speakers, option, strict=True)}
            )

        speaker_counts = Counter(assignment.values())
        if set(speaker_counts) != set(range(fold_count)):
            continue
        if max(speaker_counts.values()) - min(speaker_counts.values()) > 1:
            continue
        if any(abs(count - speaker_target) > 1 for count in speaker_counts.values()):
            continue

        condition_speaker_counts: dict[str, Counter[int]] = defaultdict(Counter)
        condition_record_counts: dict[str, Counter[int]] = defaultdict(Counter)
        total_record_counts: Counter[int] = Counter()
        for values in strata.values():
            for speaker in values:
                fold = assignment[speaker.speaker_id]
                condition_speaker_counts[speaker.condition.value][fold] += 1
                condition_record_counts[speaker.condition.value][fold] += speaker.balance_records
                total_record_counts[fold] += speaker.balance_records
        if any(
            max(counts.values()) - min(counts.values()) > 1
            for counts in condition_speaker_counts.values()
        ):
            continue

        relative_deviations: list[float] = []
        for condition, target in condition_targets.items():
            denominator = target if target else 1.0
            for fold in range(fold_count):
                value = condition_record_counts[condition][fold]
                relative_deviations.append(abs(value - target) / denominator)
        total_deviations = [
            abs(total_record_counts[fold] - total_target) for fold in range(fold_count)
        ]
        signature = "|".join(
            ",".join(sorted(speaker for speaker, value in assignment.items() if value == fold))
            for fold in range(fold_count)
        )
        tie_break = hashlib.sha256(f"{seed}|{signature}".encode()).hexdigest()
        score: tuple[float | int | str, ...] = (
            round(max(relative_deviations, default=0.0), 12),
            round(sum(value * value for value in relative_deviations), 12),
            max(total_record_counts.values()) - min(total_record_counts.values()),
            round(sum(value * value for value in total_deviations), 6),
            tie_break,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise SplitError("no balanced speaker partition satisfies the hard constraints")
    return best_assignment


def _balanced_onto_assignments(item_count: int, fold_count: int) -> list[tuple[int, ...]]:
    if item_count < fold_count:
        return []
    lower = item_count // fold_count
    upper = math.ceil(item_count / fold_count)
    assignments: list[tuple[int, ...]] = []
    for assignment in itertools.product(range(fold_count), repeat=item_count):
        counts = Counter(assignment)
        if set(counts) != set(range(fold_count)):
            continue
        if all(lower <= counts[fold] <= upper for fold in range(fold_count)):
            assignments.append(assignment)
    return assignments


def _partition_counts(
    records: Sequence[ManifestRecord],
    speaker_ids: set[str],
    config: TorgoConfig,
) -> dict[str, Any]:
    selected = [record for record in records if record.speaker_id in speaker_ids]
    return {
        "speaker_count": len(speaker_ids),
        "logical_records": len(selected),
        "balance_records": sum(
            record.usability_status in config.evaluation.balance_statuses for record in selected
        ),
        "by_condition": _counter_dict(record.condition.value for record in selected),
        "by_gender": _counter_dict(record.gender.value for record in selected),
        "by_condition_gender": _counter_dict(
            f"{record.condition.value}|{record.gender.value}" for record in selected
        ),
        "by_usability_status": _counter_dict(record.usability_status.value for record in selected),
        "by_prompt_type": _counter_dict(record.prompt_type.value for record in selected),
    }


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _audio_paths(records: Iterable[ManifestRecord]) -> set[str]:
    paths: set[str] = set()
    for record in records:
        if record.primary_audio_path:
            paths.add(record.primary_audio_path)
        if record.alternate_audio_path:
            paths.add(record.alternate_audio_path)
    return paths


def _string_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SplitError(f"{label} must be a list of speaker IDs")
    if len(value) != len(set(value)):
        raise SplitError(f"{label} contains duplicate speakers")
    return set(value)
