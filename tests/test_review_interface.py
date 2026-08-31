from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
import shutil

import pytest

from windblade_review.packet import ReviewPacketError, load_review_packet
from windblade_review.schema import ReviewSchemaError, load_pass_schema
from windblade_review.store import ReviewDataError, ReviewStore
from windblade_review.workflow import GateError, pass_b_access_allowed, validate_pass_a_lock


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/error_analysis.yaml"
PACKET = ROOT / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"
PASS_A_FORM = PACKET / "pass_a/pass_a_review_form.csv"
PASS_B_FORM = PACKET / "pass_b/pass_b_review_form.csv"
MAPPING = PACKET / "id_mapping/review_id_mapping.csv"
BLANK_PASS_A_HASH = "44a5200e8b921b65f55cd391943abfbd4ca600e9723fcd8c70a36eb6cf2b7d58"
COMPLETED_PASS_A_HASH = "3b6548d8e6a1240c224f156f9266c5025cc099816d73a8c81960173fe9c8423e"
BLANK_PASS_B_HASH = "7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8"
COMPLETED_PASS_B_HASH = "0f5258e06a4e854d338705bcf1d38ced048f0652a99ccc4639b18c3baae1cd96"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pass_a_schema():
    return load_pass_schema(CONFIG, "pass_a")


@pytest.fixture(scope="module")
def pass_b_schema():
    return load_pass_schema(CONFIG, "pass_b")


@pytest.fixture(scope="module")
def pass_a_packet():
    return load_review_packet(ROOT, PACKET, "pass_a")


@pytest.fixture(scope="module")
def pass_b_packet():
    return load_review_packet(ROOT, PACKET, "pass_b")


def copied_forms(tmp_path: Path) -> Path:
    root = tmp_path / "forms"
    (root / "pass_a").mkdir(parents=True)
    (root / "pass_b").mkdir(parents=True)
    shutil.copy2(PASS_A_FORM, root / "pass_a/pass_a_review_form.csv")
    shutil.copy2(PASS_B_FORM, root / "pass_b/pass_b_review_form.csv")
    for path in (
        root / "pass_a/pass_a_review_form.csv",
        root / "pass_b/pass_b_review_form.csv",
    ):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or ())
            rows = list(reader)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {
                    field: row["review_id"] if field == "review_id" else ""
                    for field in headers
                }
                for row in rows
            )
    return root


def first_choices(schema) -> dict[str, str]:
    return {
        field.name: (field.choices[0] if field.required else "")
        for field in schema.fields
    }


def complete(store: ReviewStore) -> None:
    values = first_choices(store.schema)
    for review_id in store.expected_ids:
        store.save_case(review_id, values)


def test_exact_frozen_schemas_and_choice_spellings(pass_a_schema, pass_b_schema):
    assert pass_a_schema.headers == (
        "review_id",
        "defect_visible",
        "corruption_obscures_diagnostic_detail",
        "dataset_label_visually_plausible",
        "visually_ambiguous_between_categories",
        "possible_crop_or_background_problem",
        "reviewer_notes",
    )
    assert pass_b_schema.headers == (
        "review_id",
        "activation_primarily_inside_annotation",
        "activation_concentrated_on_degradation_artifact",
        "pattern_consistent_across_cnn_seeds",
        "prediction_visually_understandable_after_reveal",
        "reviewer_notes",
    )
    assert pass_a_schema.definition("defect_visible").choices == (
        "yes",
        "partially",
        "no",
        "uncertain",
    )
    assert pass_b_schema.definition("pattern_consistent_across_cnn_seeds").choices == (
        "yes",
        "partly",
        "no",
        "uncertain",
    )


def test_canonical_forms_preserve_completed_pass_a_and_completed_pass_b(
    pass_a_schema, pass_b_schema, pass_a_packet, pass_b_packet
):
    a = ReviewStore(PASS_A_FORM, pass_a_schema, pass_a_packet.review_ids).load()
    b = ReviewStore(PASS_B_FORM, pass_b_schema, pass_b_packet.review_ids).load()
    assert len(a.rows) == len(b.rows) == 60
    assert a.total_required == 300 and a.answered_required in {0, 300}
    if a.answered_required == 0:
        assert sha256(PASS_A_FORM) == BLANK_PASS_A_HASH
    else:
        assert a.complete
        assert sha256(PASS_A_FORM) == COMPLETED_PASS_A_HASH
    assert b.answered_required == 240 and b.total_required == 240 and b.complete
    assert sha256(PASS_B_FORM) == COMPLETED_PASS_B_HASH


def test_packet_ids_are_exact_and_ordered(pass_a_packet, pass_b_packet):
    expected = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    assert pass_a_packet.review_ids == pass_b_packet.review_ids == expected


def test_pass_a_packet_contains_only_allowed_case_metadata_and_assets(pass_a_packet):
    for case in pass_a_packet.cases:
        assert case.metadata == f"Dataset true label: {case.true_label}"
        assert len(case.assets) in {2, 4}
        assert all(case.review_id in asset.path.parts for asset in case.assets)
        assert all("gradcam" not in asset.path.as_posix().lower() for asset in case.assets)
        assert all(asset.path.name in {
            "clean.png", "clean_annotation.png", "degraded.png", "degraded_annotation.png"
        } for asset in case.assets)


def test_pass_b_is_loaded_only_by_an_explicit_pass_b_call(monkeypatch, pass_a_schema):
    accessed: list[str] = []
    original = Path.open

    def spy(path: Path, *args, **kwargs):
        accessed.append(path.as_posix())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    packet = load_review_packet(ROOT, PACKET, "pass_a")
    ReviewStore(PASS_A_FORM, pass_a_schema, packet.review_ids).load()
    assert not any("/pass_b/" in path for path in accessed)
    assert not any("id_mapping" in path for path in accessed)
    assert not any("review_id_mapping.csv" in path for path in accessed)


def test_mapping_is_never_needed_to_load_either_packet(monkeypatch):
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        if path.resolve() == MAPPING.resolve():
            raise AssertionError("the separate ID mapping was opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert len(load_review_packet(ROOT, PACKET, "pass_a").cases) == 60
    assert len(load_review_packet(ROOT, PACKET, "pass_b").cases) == 60


def test_external_or_escaping_assets_are_rejected(tmp_path: Path):
    review = tmp_path / "packet/pass_a"
    review.mkdir(parents=True)
    cases = "".join(
        f'<section class="case"><h2>P9A-{index:03d}</h2><p>Dataset true label: craze</p>'
        f'<img src="https://example.com/{index}.png"><img src="https://example.com/{index}a.png"></section>'
        for index in range(1, 61)
    )
    (review / "index.html").write_text(f"<html><body>{cases}</body></html>", encoding="utf-8")
    with pytest.raises(ReviewPacketError, match="external"):
        load_review_packet(tmp_path, tmp_path / "packet", "pass_a")


def test_partial_save_resumes_at_first_incomplete_without_reordering(
    tmp_path, pass_a_schema, pass_a_packet
):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    expected_order = pass_a_packet.review_ids
    store.save_case(expected_order[0], first_choices(pass_a_schema))
    store.save_case(expected_order[1], {"defect_visible": "uncertain"})
    resumed = store.load()
    assert resumed.first_incomplete_index == 1
    assert resumed.completed_cases == 1
    assert tuple(row["review_id"] for row in resumed.rows) == expected_order


def test_complete_pass_is_detected_after_restart(tmp_path, pass_a_schema, pass_a_packet):
    forms = copied_forms(tmp_path)
    path = forms / "pass_a/pass_a_review_form.csv"
    complete(ReviewStore(path, pass_a_schema, pass_a_packet.review_ids))
    restarted = ReviewStore(path, pass_a_schema, pass_a_packet.review_ids).load(require_complete=True)
    assert restarted.complete and restarted.completed_cases == 60
    assert restarted.answered_required == 300


def test_partial_pass_b_resume_after_gate(tmp_path, pass_a_schema, pass_b_schema, pass_a_packet, pass_b_packet):
    forms = copied_forms(tmp_path)
    a_store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    b_store = ReviewStore(forms / "pass_b/pass_b_review_form.csv", pass_b_schema, pass_b_packet.review_ids)
    complete(a_store)
    b_store.save_case(pass_b_packet.review_ids[0], first_choices(pass_b_schema))
    assert validate_pass_a_lock(a_store, True)
    resumed = ReviewStore(b_store.path, pass_b_schema, pass_b_packet.review_ids).load()
    assert resumed.first_incomplete_index == 1
    assert pass_b_access_allowed(a_store.load(), pass_a_locked=True, pass_b_started=True)


def test_notes_round_trip_commas_quotes_and_unicode(tmp_path, pass_a_schema, pass_a_packet):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    note = 'Comma, quote "preserved" — 中文; no normalization.'
    store.save_case(pass_a_packet.review_ids[0], {"reviewer_notes": note})
    assert store.load().rows[0]["reviewer_notes"] == note


def test_invalid_response_is_never_silently_repaired(tmp_path, pass_a_schema, pass_a_packet):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    with pytest.raises(ReviewDataError, match="invalid response"):
        store.save_case(pass_a_packet.review_ids[0], {"defect_visible": "probably"})
    assert store.load().answered_required == 0


def test_changed_header_id_row_count_and_order_are_rejected(tmp_path, pass_a_schema, pass_a_packet):
    forms = copied_forms(tmp_path)
    path = forms / "pass_a/pass_a_review_form.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0], rows[1] = rows[1], rows[0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pass_a_schema.headers)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ReviewDataError, match="IDs or row order"):
        ReviewStore(path, pass_a_schema, pass_a_packet.review_ids).load()


def test_atomic_write_and_interrupted_replace_preserve_canonical(
    tmp_path, pass_a_schema, pass_a_packet
):
    forms = copied_forms(tmp_path)
    path = forms / "pass_a/pass_a_review_form.csv"
    before = path.read_bytes()

    def interrupted(_source, _destination):
        raise OSError("simulated replacement interruption")

    store = ReviewStore(path, pass_a_schema, pass_a_packet.review_ids, replace_file=interrupted)
    with pytest.raises(OSError, match="interruption"):
        store.save_case(pass_a_packet.review_ids[0], {"defect_visible": "yes"})
    assert path.read_bytes() == before
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_saving_one_pass_does_not_change_the_other(
    tmp_path, pass_a_schema, pass_a_packet
):
    forms = copied_forms(tmp_path)
    other = forms / "pass_b/pass_b_review_form.csv"
    before = sha256(other)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    store.save_case(pass_a_packet.review_ids[0], {"defect_visible": "yes"})
    assert sha256(other) == before == BLANK_PASS_B_HASH


def test_incomplete_invalid_or_unattested_pass_a_cannot_lock(
    tmp_path, pass_a_schema, pass_a_packet
):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    with pytest.raises(GateError, match="incomplete"):
        validate_pass_a_lock(store, True)
    complete(store)
    with pytest.raises(GateError, match="attestation"):
        validate_pass_a_lock(store, False)
    assert validate_pass_a_lock(store, True) == store.sha256()


def test_gate_enables_but_does_not_automatically_start_it(
    tmp_path, pass_a_schema, pass_a_packet
):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_a/pass_a_review_form.csv", pass_a_schema, pass_a_packet.review_ids)
    complete(store)
    snapshot = store.load()
    assert not pass_b_access_allowed(snapshot, pass_a_locked=False, pass_b_started=True)
    assert not pass_b_access_allowed(snapshot, pass_a_locked=True, pass_b_started=False)
    assert pass_b_access_allowed(snapshot, pass_a_locked=True, pass_b_started=True)


def test_pass_b_completion_requires_all_240_answers(
    tmp_path, pass_b_schema, pass_b_packet
):
    forms = copied_forms(tmp_path)
    store = ReviewStore(forms / "pass_b/pass_b_review_form.csv", pass_b_schema, pass_b_packet.review_ids)
    with pytest.raises(ReviewDataError, match="0/240"):
        store.load(require_complete=True)
    complete(store)
    assert store.load(require_complete=True).answered_required == 240


def test_interface_source_is_scientifically_isolated():
    paths = [ROOT / "app/review_app.py", *sorted((ROOT / "src/windblade_review").glob("*.py"))]
    imported: set[str] = set()
    combined = ""
    for path in paths:
        source = path.read_text(encoding="utf-8")
        combined += source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name.startswith("windblade_demo") for name in imported)
    assert not any(name in {"requests", "socket", "urllib.request"} for name in imported)
    assert "review_id_mapping.csv" not in combined
    assert "best_state_dict" not in combined
    assert "st.query_params" not in combined


def test_packet_images_and_mapping_are_unchanged_by_reading(pass_a_packet, pass_b_packet):
    before_mapping = sha256(MAPPING)
    hashes = {asset.path: sha256(asset.path) for case in pass_a_packet.cases for asset in case.assets}
    hashes.update({asset.path: sha256(asset.path) for case in pass_b_packet.cases for asset in case.assets})
    assert len(pass_a_packet.cases) == len(pass_b_packet.cases) == 60
    assert all(sha256(path) == digest for path, digest in hashes.items())
    assert sha256(MAPPING) == before_mapping == "46b19248797997e8aa7236b9c3fbf17f972977f2cdb2c3a958bd12513f25210b"
