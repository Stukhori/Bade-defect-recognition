from windblade.data.processed import calculate_processed_fingerprint


def _fingerprint(**changes):
    values = {
        "config_sha256": "config-a",
        "curation_manifest_sha256": "curation-a",
        "manifest_content": "manifest-a",
        "checksum_content": "checksums-a",
    }
    values.update(changes)
    return calculate_processed_fingerprint(**values)


def test_identical_processed_inputs_have_identical_fingerprint():
    assert _fingerprint() == _fingerprint()


def test_changed_crop_bytes_change_processed_fingerprint():
    assert _fingerprint() != _fingerprint(checksum_content="checksums-b")


def test_changed_crop_config_changes_processed_fingerprint():
    assert _fingerprint() != _fingerprint(config_sha256="config-b")


def test_changed_manifest_changes_processed_fingerprint():
    assert _fingerprint() != _fingerprint(manifest_content="manifest-b")
