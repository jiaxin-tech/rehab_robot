"""Causal policy replay with every executed attempt accounted for.

A failed/invalid attempt consumes one slot and terminates this version. It is
never dropped from the ledger or retried for free. Replaying success prefixes
uses one policy instance so probe roles, stop decisions and model updates are
reconstructed before validating the next historical execution.
"""
from __future__ import annotations

from dataclasses import asdict


def replay_attempts(policy, attempts, load_measurement):
    if not isinstance(attempts, list):
        raise ValueError("ordered attempts ledger required")
    if len(attempts) > policy.config.total_budget:
        raise ValueError("HISTORY_EXCEEDS_TOTAL_ATTEMPT_BUDGET")
    measurements, transcript = [], []
    seen_episodes, seen_candidates = set(), set()
    action = policy.next_action(measurements)
    failed = False
    for index, attempt in enumerate(attempts, 1):
        if action.role == "STOP" or action.candidate_id is None:
            raise ValueError("ATTEMPT_AFTER_POLICY_STOP")
        if attempt.get("attempt_index") != index:
            raise ValueError("CONTIGUOUS_ONE_BASED_ATTEMPT_INDEX_REQUIRED")
        episode = attempt.get("episode_id")
        candidate = attempt.get("candidate_id")
        if not isinstance(episode, str) or not episode.strip() or episode in seen_episodes or candidate in seen_candidates:
            raise ValueError("DUPLICATE_OR_INVALID_ATTEMPT_IDENTITY")
        if candidate != action.candidate_id or attempt.get("role") != action.role:
            raise ValueError("HISTORICAL_ACTION_MISMATCH")
        seen_episodes.add(episode)
        seen_candidates.add(candidate)
        transcript.append(dict(attempt_index=index, episode_id=episode, candidate_id=candidate,
                               proposed_action=asdict(action), outcome=attempt.get("outcome")))
        if attempt.get("outcome") == "FAILED":
            reason = attempt.get("failure_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("FAILED_ATTEMPT_REASON_REQUIRED")
            if any(key in attempt for key in ("measured_path", "measured_sha256", "endpoints")):
                raise ValueError("FAILED_ATTEMPT_MUST_NOT_SUPPLY_TRAINING_DATA")
            action = type(action)(None, "STOP", "ATTEMPT_FAILED", dict(
                attempted_trials=index, total_budget=policy.config.total_budget,
                remaining_trials=max(0, policy.config.total_budget-index), failure_reason=reason))
            failed = True
            continue
        if attempt.get("outcome") != "COMPLETED":
            raise ValueError("ATTEMPT_OUTCOME_MUST_BE_COMPLETED_OR_FAILED")
        measurement = load_measurement(attempt, index)
        observation = measurement.observation
        if (observation.candidate_id != candidate or observation.episode_id != episode
                or not observation.valid):
            raise ValueError("ATTEMPT_MEASUREMENT_IDENTITY_OR_VALIDITY_MISMATCH")
        measurements.append(measurement)
        # Fit the new prefix even if budget/domain exhaustion stops selection.
        # Surrogate.fit ignores an unchanged prefix, so the policy's subsequent
        # internal fit does not duplicate prequential observations.
        policy.surrogate.fit(measurements)
        action = policy.next_action(measurements)
    return dict(measurements=measurements, action=action, transcript=transcript,
                policy_roles=dict(policy.roles), attempted_trials=len(attempts),
                completed_trials=len(measurements), failed_trials=int(failed),
                remaining_trials=max(0, policy.config.total_budget-len(attempts)))
