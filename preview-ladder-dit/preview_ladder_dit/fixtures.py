from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

Video = list[list[list[float]]]
Mask = list[list[list[bool]]]

FIXTURE_CASES = (
    "clean",
    "identity_drift",
    "boundary_halo",
    "background_leak",
    "motion_mismatch",
    "temporal_flicker",
    "occlusion_failure",
    "mask_instability",
    "thin_structure_boundary",
    "illumination_pulse",
    "shadow_leak",
    "parallax_mismatch",
)

FIXTURE_METADATA: dict[str, dict[str, object]] = {
    "clean": {"failure_axis": "control", "expected_detector_metrics": ("preview_final_l1",), "generation_seed": 0, "difficulty": "control"},
    "identity_drift": {"failure_axis": "appearance", "expected_detector_metrics": ("preview_final_l1", "low_frequency_drift"), "generation_seed": 101, "difficulty": "easy"},
    "boundary_halo": {"failure_axis": "boundary", "expected_detector_metrics": ("boundary_consistency_error", "boundary_signed_bias"), "generation_seed": 102, "difficulty": "easy"},
    "background_leak": {"failure_axis": "background", "expected_detector_metrics": ("background_preservation_error", "background_temporal_leak"), "generation_seed": 103, "difficulty": "easy"},
    "motion_mismatch": {"failure_axis": "motion", "expected_detector_metrics": ("trajectory_center_error", "occupancy_iou_error"), "generation_seed": 104, "difficulty": "medium"},
    "temporal_flicker": {"failure_axis": "temporal", "expected_detector_metrics": ("temporal_flicker_delta", "local_temporal_residual_flicker"), "generation_seed": 105, "difficulty": "easy"},
    "occlusion_failure": {"failure_axis": "occlusion", "expected_detector_metrics": ("mask_occupancy_delta", "occupancy_iou_error"), "generation_seed": 106, "difficulty": "medium"},
    "mask_instability": {"failure_axis": "mask", "expected_detector_metrics": ("temporal_edge_jitter", "mask_occupancy_delta"), "generation_seed": 107, "difficulty": "medium"},
    "thin_structure_boundary": {"failure_axis": "thin_structure", "expected_detector_metrics": ("boundary_consistency_error", "occupancy_iou_error"), "generation_seed": 108, "difficulty": "hard"},
    "illumination_pulse": {"failure_axis": "illumination", "expected_detector_metrics": ("temporal_flicker_delta", "low_frequency_drift"), "generation_seed": 109, "difficulty": "medium"},
    "shadow_leak": {"failure_axis": "background", "expected_detector_metrics": ("background_preservation_error", "boundary_consistency_error"), "generation_seed": 110, "difficulty": "medium"},
    "parallax_mismatch": {"failure_axis": "camera_motion", "expected_detector_metrics": ("trajectory_center_error", "temporal_flicker_delta"), "generation_seed": 111, "difficulty": "hard"},
}


EXPECTED_METRIC_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "clean": {
        "preview_final_l1": (0.0, 0.0),
        "boundary_consistency_error": (0.0, 0.0),
        "background_preservation_error": (0.0, 0.0),
    },
    "identity_drift": {"preview_final_l1": (0.08, 0.30), "low_frequency_drift": (0.08, 0.30)},
    "boundary_halo": {"boundary_consistency_error": (0.05, 0.35)},
    "background_leak": {"background_preservation_error": (0.05, 0.35)},
    "motion_mismatch": {"trajectory_center_error": (0.03, 0.40), "occupancy_iou_error": (0.15, 1.0)},
    "temporal_flicker": {"temporal_flicker_delta": (0.10, 0.50), "local_temporal_residual_flicker": (0.10, 0.60)},
    "occlusion_failure": {"mask_occupancy_delta": (0.05, 0.40), "occupancy_iou_error": (0.05, 0.60)},
    "mask_instability": {"temporal_edge_jitter": (0.01, 0.50), "mask_occupancy_delta": (0.05, 0.50)},
    "thin_structure_boundary": {"boundary_consistency_error": (0.02, 0.35), "occupancy_iou_error": (0.02, 0.80)},
    "illumination_pulse": {"temporal_flicker_delta": (0.03, 0.35), "low_frequency_drift": (0.02, 0.30)},
    "shadow_leak": {"background_preservation_error": (0.01, 0.25), "boundary_consistency_error": (0.01, 0.30)},
    "parallax_mismatch": {"trajectory_center_error": (0.01, 0.30), "temporal_flicker_delta": (0.01, 0.30)},
}


@dataclass(frozen=True)
class FixtureBundle:
    case: str
    source: Video
    preview: Video
    final: Video
    mask: Mask
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_fixture(case: str, *, frames: int = 8, height: int = 32, width: int = 32) -> FixtureBundle:
    if case not in FIXTURE_CASES:
        raise ValueError(f"unknown fixture case {case!r}; expected one of {FIXTURE_CASES}")
    if frames < 2 or height < 8 or width < 8:
        raise ValueError("fixtures require frames>=2, height>=8, width>=8")
    mask = _moving_box_mask(frames, height, width)
    source = _base_source(frames, height, width)
    preview = _make_preview(source, mask)
    final = _copy_video(preview)
    description = "final matches accepted preview"
    if case == "identity_drift":
        _inside(mask, final, lambda v, t, y, x: _clip(v - 0.22 + 0.03 * math.sin(t + x)))
        description = "replacement identity/color drifts inside the approved mask"
    elif case == "boundary_halo":
        band = _boundary_band(mask, 1)
        _where(band, final, lambda v, t, y, x: _clip(v + 0.25))
        description = "visible halo appears on mask boundary"
    elif case == "background_leak":
        _outside(mask, final, lambda v, t, y, x: _clip(v + (0.28 if (x + y + t) % 5 == 0 else 0.08)))
        description = "source background changes outside editable region"
    elif case == "motion_mismatch":
        final = _make_preview(source, _moving_box_mask(frames, height, width, x_shift=3))
        description = "final replacement follows a different trajectory than preview"
    elif case == "temporal_flicker":
        _inside(mask, final, lambda v, t, y, x: _clip(v + (0.22 if t % 2 else -0.18)))
        description = "final replacement flickers despite stable preview"
    elif case == "occlusion_failure":
        for t in range(frames // 2, frames):
            for y, row in enumerate(final[t]):
                for x, _ in enumerate(row):
                    if mask[t][y][x] and (x + y) % 3 == 0:
                        row[x] = source[t][y][x]
        description = "replacement disappears intermittently under occlusion"
    elif case == "mask_instability":
        for t in range(1, frames, 2):
            for y in range(height):
                for x in range(width):
                    if mask[t][y][x] and x % 2 == 0:
                        final[t][y][x] = source[t][y][x]
        description = "final mask occupancy varies frame to frame"
    elif case == "thin_structure_boundary":
        for t in range(frames):
            for y in range(height):
                for x in range(width):
                    if mask[t][y][x] and (x + 2 * y + t) % 7 == 0:
                        preview[t][y][x] = 0.96
                        final[t][y][x] = source[t][y][x]
        description = "approved thin structures are dropped or softened near the replacement boundary"
    elif case == "illumination_pulse":
        _inside(mask, final, lambda v, t, y, x: _clip(v + 0.16 * math.sin(t * math.pi / 2.0)))
        description = "final introduces a coherent brightness pulse not present in the accepted preview"
    elif case == "shadow_leak":
        band = _boundary_band(mask, 2)
        for t in range(frames):
            for y in range(height):
                for x in range(width):
                    if band[t][y][x] and not mask[t][y][x]:
                        final[t][y][x] = _clip(final[t][y][x] - 0.18 + 0.03 * math.sin(x + t))
        description = "replacement casts an unapproved shadow into locked background pixels"
    elif case == "parallax_mismatch":
        shifted = _moving_box_mask(frames, height, width)
        final = _copy_video(source)
        for t in range(frames):
            dx = 1 + (t % 3)
            for y in range(height):
                for x in range(width):
                    if shifted[t][y][x]:
                        src_x = max(0, min(width - 1, x - dx))
                        final[t][y][x] = preview[t][y][src_x]
        description = "final follows a row-stable but preview-inconsistent parallax offset"
    return FixtureBundle(
        case=case,
        source=source,
        preview=preview,
        final=final,
        mask=mask,
        metadata={
            **{key: (list(value) if isinstance(value, tuple) else value) for key, value in FIXTURE_METADATA[case].items()},
            "failure_description": description,
            "frames": frames,
            "height": height,
            "width": width,
            "expected_metric_bands": EXPECTED_METRIC_BANDS.get(case, {}),
        },
    )


def write_fixtures(out_dir: str | Path, *, cases: Sequence[str] = FIXTURE_CASES, frames: int = 8, height: int = 32, width: int = 32) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in cases:
        bundle = make_fixture(case, frames=frames, height=height, width=width)
        path = out / f"{case}.json"
        path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        paths.append(path)
    manifest = {"cases": list(cases), "frames": frames, "height": height, "width": width, "files": [p.name for p in paths]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def _base_source(frames: int, height: int, width: int) -> Video:
    video: Video = []
    for t in range(frames):
        frame: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                gradient = 0.10 + 0.45 * x / max(1, width - 1) + 0.20 * y / max(1, height - 1)
                temporal = 0.04 * math.sin((x + t) * 0.35) + 0.03 * math.cos((y - t) * 0.22)
                row.append(_clip(gradient + temporal))
            frame.append(row)
        video.append(frame)
    return video


def _moving_box_mask(frames: int, height: int, width: int, *, x_shift: int = 0) -> Mask:
    box_h = max(3, height // 4)
    box_w = max(3, width // 5)
    out: Mask = []
    for t in range(frames):
        cx = width // 4 + x_shift + t * max(1, width // 2 // max(1, frames - 1))
        cy = height // 2 + int(round(math.sin(t * 0.7)))
        y0 = max(1, min(height - box_h - 1, cy - box_h // 2))
        x0 = max(1, min(width - box_w - 1, cx - box_w // 2))
        frame = [[False for _ in range(width)] for _ in range(height)]
        for y in range(y0, y0 + box_h):
            for x in range(x0, x0 + box_w):
                frame[y][x] = True
        out.append(frame)
    return out


def _make_preview(source: Video, mask: Mask) -> Video:
    out = _copy_video(source)
    _inside(mask, out, lambda v, t, y, x: _clip(0.78 + 0.04 * math.sin(0.4 * t + 0.2 * x) - 0.03 * math.cos(0.3 * y)))
    return out


def _copy_video(video: Video) -> Video:
    return [[list(row) for row in frame] for frame in video]


def _inside(mask: Mask, video: Video, fn) -> None:
    _where(mask, video, fn)


def _outside(mask: Mask, video: Video, fn) -> None:
    for t, frame in enumerate(video):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if not mask[t][y][x]:
                    row[x] = fn(value, t, y, x)


def _where(include: Mask, video: Video, fn) -> None:
    for t, frame in enumerate(video):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if include[t][y][x]:
                    row[x] = fn(value, t, y, x)


def _boundary_band(mask: Mask, radius: int) -> Mask:
    frames, height, width = len(mask), len(mask[0]), len(mask[0][0])
    out = [[[False for _ in range(width)] for _ in range(height)] for _ in range(frames)]
    for t in range(frames):
        for y in range(height):
            for x in range(width):
                here = mask[t][y][x]
                edge = False
                for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                    for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                        if mask[t][yy][xx] != here:
                            edge = True
                if edge:
                    out[t][y][x] = True
    return out


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
