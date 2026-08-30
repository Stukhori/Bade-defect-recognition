"""Pure two-pass gate rules for the local review interface."""

from __future__ import annotations

from windblade_review.store import ReviewDataError, ReviewSnapshot, ReviewStore


PASS_A_ATTESTATION = (
    "I completed Pass A without opening Pass B or the separate ID mapping. "
    "I understand that locking Pass A prevents later changes after model evidence is revealed."
)


class GateError(RuntimeError):
    """Raised when a reviewer attempts to cross an incomplete gate."""


def validate_pass_a_lock(store: ReviewStore, attested: bool) -> str:
    """Validate every response and the explicit attestation, returning the form hash."""

    try:
        store.load(require_complete=True)
    except ReviewDataError as exc:
        raise GateError(str(exc)) from exc
    if not attested:
        raise GateError("Pass A cannot be locked until the reviewer checks the attestation.")
    return store.sha256()


def pass_b_access_allowed(
    pass_a_snapshot: ReviewSnapshot,
    *,
    pass_a_locked: bool,
    pass_b_started: bool,
) -> bool:
    return pass_a_snapshot.complete and pass_a_locked and pass_b_started
