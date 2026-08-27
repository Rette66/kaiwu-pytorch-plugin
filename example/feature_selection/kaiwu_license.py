from __future__ import annotations

import os


def _init_kaiwu_license_from_env() -> None:
    """Initialize the Kaiwu license from environment variables.

    Raises:
        RuntimeError: If Kaiwu license initialization fails.
    """
    user_id = os.environ.get("LICENSE_USER_ID")
    sdk_code = os.environ.get("LICENSE_SDK_CODE")
    if not user_id or not sdk_code:
        return

    import kaiwu.license as license_manager

    try:
        license_manager.init(user_id, sdk_code)
    except Exception as exc:
        raise RuntimeError(
            "Kaiwu license initialization failed. Check whether LICENSE_USER_ID "
            "and LICENSE_SDK_CODE are correct, and whether the machine can reach "
            "the Kaiwu license server."
        ) from exc
