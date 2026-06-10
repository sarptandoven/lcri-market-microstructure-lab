"""Prototype validator for Preview Ladder DiT public submissions.

This is intentionally dependency-light so it can be promoted into the target repo
without adding benchmark infrastructure dependencies. It validates the parts of a
submission that are easy to cheat accidentally or intentionally: preview/final
phase separation, immutable artifacts, hidden-test hygiene, hardware-normalized
latency records, and reproducibility attestations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
OCI_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
ALLOWED_HARDWARE_PROFILES = {
    "cpu-reference-v1",
    "a100-80gb-v1",
    "h100-80gb-v1",
    "rtx4090-v1",
    "mps-m3-ultra-v1",
}
REQUIRED_ATTESTATION_KEYS = {
    "git_commit",
    "container_digest",
    "source_tree_sha256",
    "dependency_lock_sha256",
    "model_weight_sha256",
    "seed_policy",
    "determinism_policy",
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


def validate_submission_bundle(bundle: Mapping[str, Any], *, artifact_root: str | Path | None = None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _validate_preview_final_separation(bundle, issues)
    _validate_artifact_immutability(bundle, issues, artifact_root)
    _validate_hidden_test_hygiene(bundle, issues)
    _validate_hardware_normalization(bundle, issues)
    _validate_reproducibility_attestations(bundle, issues)
    return issues


def load_and_validate(path: str | Path, *, artifact_root: str | Path | None = None) -> list[ValidationIssue]:
    return validate_submission_bundle(json.loads(Path(path).read_text(encoding="utf-8")), artifact_root=artifact_root)


def _validate_preview_final_separation(bundle: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    preview = _map(bundle.get("preview_submission"))
    final = _map(bundle.get("final_submission"))
    if not preview or not final:
        issues.append(ValidationIssue("missing_phase", "$", "bundle must contain preview_submission and final_submission"))
        return

    preview_closed_at = _num(preview.get("sealed_at_s"))
    final_started_at = _num(final.get("started_at_s"))
    if preview_closed_at is None or final_started_at is None:
        issues.append(ValidationIssue("missing_phase_time", "$.preview_submission.sealed_at_s", "preview seal and final start times are required"))
    elif preview_closed_at > final_started_at:
        issues.append(ValidationIssue("final_leakage_time", "$.preview_submission.sealed_at_s", "preview must be sealed before final starts"))

    forbidden = {"final_uri", "final_sha256", "final_latent_sha256", "final_metric", "final_score"}
    for key in forbidden.intersection(preview.keys()):
        issues.append(ValidationIssue("final_leakage_field", f"$.preview_submission.{key}", "preview submission may not contain final-derived fields"))

    preview_inputs = set(preview.get("declared_inputs", [])) if isinstance(preview.get("declared_inputs"), list) else set()
    for item in preview_inputs:
        if isinstance(item, str) and item.startswith("hidden_final/"):
            issues.append(ValidationIssue("final_leakage_input", "$.preview_submission.declared_inputs", "preview declared a hidden-final input"))


def _validate_artifact_immutability(bundle: Mapping[str, Any], issues: list[ValidationIssue], artifact_root: str | Path | None) -> None:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issues.append(ValidationIssue("missing_artifacts", "$.artifacts", "at least preview and final artifacts are required"))
        return
    seen_roles: set[str] = set()
    for idx, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            issues.append(ValidationIssue("bad_artifact", f"$.artifacts[{idx}]", "artifact must be an object"))
            continue
        role = str(artifact.get("role", ""))
        seen_roles.add(role)
        sha = str(artifact.get("sha256", ""))
        uri = str(artifact.get("uri", ""))
        if not SHA256_RE.match(sha):
            issues.append(ValidationIssue("bad_sha256", f"$.artifacts[{idx}].sha256", "artifact sha256 must be lowercase hex"))
        if sha and sha not in uri:
            issues.append(ValidationIssue("not_content_addressed", f"$.artifacts[{idx}].uri", "artifact URI should include its sha256 digest"))
        if artifact.get("mutable", False):
            issues.append(ValidationIssue("mutable_artifact", f"$.artifacts[{idx}].mutable", "submitted artifacts must be immutable"))
        if artifact_root and uri.startswith("file://") and SHA256_RE.match(sha):
            candidate = Path(artifact_root) / uri.removeprefix("file://")
            if candidate.exists():
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if actual != sha:
                    issues.append(ValidationIssue("sha256_mismatch", f"$.artifacts[{idx}].sha256", f"expected {sha}, got {actual}"))
    for role in ("preview_video", "final_video", "report_json"):
        if role not in seen_roles:
            issues.append(ValidationIssue("missing_artifact_role", "$.artifacts", f"missing required role {role}"))


def _validate_hidden_test_hygiene(bundle: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    task = _map(bundle.get("task_manifest"))
    if not task:
        issues.append(ValidationIssue("missing_task_manifest", "$.task_manifest", "task manifest is required"))
        return
    if task.get("mask_split") not in {"public", "hidden-eval"}:
        issues.append(ValidationIssue("bad_mask_split", "$.task_manifest.mask_split", "mask_split must be public or hidden-eval"))
    if task.get("mask_split") == "hidden-eval":
        for key in ("mask_uri", "mask_sha256", "mask_pixels", "hidden_mask_digest"):
            if key in task:
                issues.append(ValidationIssue("hidden_mask_leak", f"$.task_manifest.{key}", "hidden-eval manifests must not expose hidden mask material"))
    if bundle.get("debug_exports"):
        issues.append(ValidationIssue("debug_exports_forbidden", "$.debug_exports", "debug exports are forbidden in public leaderboard submissions"))


def _validate_hardware_normalization(bundle: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    hw = _map(bundle.get("hardware"))
    latency = _map(bundle.get("latency"))
    if not hw or not latency:
        issues.append(ValidationIssue("missing_hardware_latency", "$", "hardware and latency blocks are required"))
        return
    if hw.get("normalization_profile") not in ALLOWED_HARDWARE_PROFILES:
        issues.append(ValidationIssue("unknown_hardware_profile", "$.hardware.normalization_profile", "hardware profile must be from the benchmark allowlist"))
    for key in ("accelerator_name", "accelerator_count", "driver_version", "runtime_version"):
        if key not in hw:
            issues.append(ValidationIssue("missing_hardware_field", f"$.hardware.{key}", "required for latency normalization"))
    for key in ("time_to_first_preview_s", "final_latency_s", "normalized_preview_s", "normalized_final_s"):
        value = _num(latency.get(key))
        if value is None or value <= 0:
            issues.append(ValidationIssue("bad_latency", f"$.latency.{key}", "latency must be a positive number"))


def _validate_reproducibility_attestations(bundle: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    att = _map(bundle.get("reproducibility"))
    if not att:
        issues.append(ValidationIssue("missing_reproducibility", "$.reproducibility", "reproducibility attestations are required"))
        return
    for key in sorted(REQUIRED_ATTESTATION_KEYS):
        if key not in att:
            issues.append(ValidationIssue("missing_attestation", f"$.reproducibility.{key}", "required reproducibility attestation missing"))
    for key in ("source_tree_sha256", "dependency_lock_sha256", "model_weight_sha256"):
        if key in att and not SHA256_RE.match(str(att[key])):
            issues.append(ValidationIssue("bad_attestation_sha", f"$.reproducibility.{key}", "attestation must be a sha256 hex digest"))
    if "container_digest" in att and not OCI_DIGEST_RE.match(str(att["container_digest"])):
        issues.append(ValidationIssue("bad_container_digest", "$.reproducibility.container_digest", "container digest must be an OCI sha256 digest"))


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("submission_json")
    args = parser.parse_args()
    issues = load_and_validate(args.submission_json)
    print(json.dumps([issue.__dict__ for issue in issues], indent=2, sort_keys=True))
    raise SystemExit(1 if issues else 0)
