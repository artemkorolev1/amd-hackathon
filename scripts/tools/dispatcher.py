#!/usr/bin/env python3
"""Root dispatcher — routes to harness.py (legacy) or staging/entrypoint.py (parallel pool).

Entrypoint for the container image.  Controlled by the STAGING_ENABLED env var:

  STAGING_ENABLED=1  → run staging.entrypoint.main()
  otherwise          → run harness.main()
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("dispatcher")


def main() -> None:
    staging_enabled = os.environ.get("STAGING_ENABLED", "0") == "1"

    if staging_enabled:
        logger.info("STAGING_ENABLED=1 — routing to staging/entrypoint")
        try:
            from staging import entrypoint  # type: ignore[import-unchecked]
            entrypoint.main()
        except ImportError as exc:
            logger.error(
                "STAGING_ENABLED=1 but staging/entrypoint is not available (%s). "
                "Falling back to harness.", exc
            )
            from harness import main as harness_main  # type: ignore[import-unchecked]
            harness_main()
    else:
        logger.info("STAGING_ENABLED unset or 0 — routing to harness")
        from harness import main as harness_main  # type: ignore[import-unchecked]
        harness_main()


if __name__ == "__main__":
    main()
