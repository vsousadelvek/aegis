"""python -m aegis — inicia o servidor AEGIS."""

from __future__ import annotations

import logging
import uvicorn

from aegis.core.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_settings()
    uvicorn.run(
        "aegis.api.app:create_app",
        factory=True,
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
