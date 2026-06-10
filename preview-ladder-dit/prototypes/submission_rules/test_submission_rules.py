import unittest

from submission_rules import validate_submission_bundle

D = "a" * 64
OCI = "sha256:" + "b" * 64


def valid_bundle():
    return {
        "task_manifest": {"task_id": "hidden-001", "mask_split": "hidden-eval"},
        "preview_submission": {"sealed_at_s": 10.0, "declared_inputs": ["source/video.mp4", "public/prompt.json"]},
        "final_submission": {"started_at_s": 11.0},
        "artifacts": [
            {"role": "preview_video", "uri": f"s3://pld/{D}/preview.mp4", "sha256": D, "mutable": False},
            {"role": "final_video", "uri": f"s3://pld/{D}/final.mp4", "sha256": D, "mutable": False},
            {"role": "report_json", "uri": f"s3://pld/{D}/report.json", "sha256": D, "mutable": False},
        ],
        "hardware": {
            "normalization_profile": "h100-80gb-v1",
            "accelerator_name": "H100 SXM 80GB",
            "accelerator_count": 1,
            "driver_version": "550.54",
            "runtime_version": "cuda-12.4",
        },
        "latency": {
            "time_to_first_preview_s": 2.3,
            "final_latency_s": 41.0,
            "normalized_preview_s": 2.1,
            "normalized_final_s": 39.5,
        },
        "reproducibility": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "container_digest": OCI,
            "source_tree_sha256": D,
            "dependency_lock_sha256": D,
            "model_weight_sha256": D,
            "seed_policy": "fixed task seed plus disclosed per-run seed",
            "determinism_policy": "torch deterministic when available; disclose nondeterministic kernels",
        },
    }


class SubmissionRulesTest(unittest.TestCase):
    def test_valid_bundle_passes(self):
        self.assertEqual(validate_submission_bundle(valid_bundle()), [])

    def test_detects_preview_final_leakage(self):
        bundle = valid_bundle()
        bundle["preview_submission"]["sealed_at_s"] = 12.0
        bundle["preview_submission"]["final_sha256"] = D
        codes = {issue.code for issue in validate_submission_bundle(bundle)}
        self.assertIn("final_leakage_time", codes)
        self.assertIn("final_leakage_field", codes)

    def test_detects_hidden_mask_and_mutable_artifacts(self):
        bundle = valid_bundle()
        bundle["task_manifest"]["mask_uri"] = "s3://private/hidden-mask.mp4"
        bundle["artifacts"][0]["mutable"] = True
        bundle["artifacts"][0]["uri"] = "s3://pld/latest/preview.mp4"
        codes = {issue.code for issue in validate_submission_bundle(bundle)}
        self.assertIn("hidden_mask_leak", codes)
        self.assertIn("mutable_artifact", codes)
        self.assertIn("not_content_addressed", codes)

    def test_requires_hardware_and_reproducibility(self):
        bundle = valid_bundle()
        del bundle["hardware"]["driver_version"]
        bundle["hardware"]["normalization_profile"] = "unknown-gpu"
        bundle["reproducibility"]["container_digest"] = "ubuntu:latest"
        codes = {issue.code for issue in validate_submission_bundle(bundle)}
        self.assertIn("missing_hardware_field", codes)
        self.assertIn("unknown_hardware_profile", codes)
        self.assertIn("bad_container_digest", codes)


if __name__ == "__main__":
    unittest.main()
