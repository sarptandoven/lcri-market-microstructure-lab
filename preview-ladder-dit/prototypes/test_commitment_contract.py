from commitment_contract import extract_preview_commitment


def test_extract_preview_commitment_basic():
    preview = [
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.8, 0.9, 0.0], [0.0, 0.7, 0.6, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.7, 0.9, 0.0], [0.0, 0.7, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ]
    mask = [
        [[False, False, False, False], [False, True, True, False], [False, True, True, False], [False, False, False, False]],
        [[False, False, False, False], [False, True, True, False], [False, True, True, False], [False, False, False, False]],
    ]
    uncertainty = [
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.2, 0.0], [0.0, 0.7, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.2, 0.0], [0.0, 0.9, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ]

    commitment = extract_preview_commitment(preview, mask, uncertainty)
    data = commitment.to_dict()

    assert data["schema_version"] == "preview-commitment/v0.1-prototype"
    assert data["frames"] == 2
    assert data["frame_commitments"][0]["mask_occupancy"] == 4 / 16
    assert data["frame_commitments"][0]["centroid_x"] == 1.5
    assert data["frame_commitments"][0]["centroid_y"] == 1.5
    assert data["frame_commitments"][0]["lock_ratio"] == 0.5
    assert data["frame_commitments"][0]["refine_ratio"] == 0.5
