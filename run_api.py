"""Launch the FastAPI server with configurable host/port.

Environment variables / CLI flags:
    API_HOST        Host address to bind (default: 0.0.0.0)
    API_PORT        Port to bind (default: 8000)
    API_RELOAD      Enable uvicorn reload (true/false, default: false)
    API_WORKERS     Number of Uvicorn workers (default: 1)

Usage examples:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8080 --workers 2
"""

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SHL Recommender API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host interface to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("API_PORT", "8000"))),
        help="Port to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of uvicorn workers (default: %(default)s)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=str_to_bool(os.getenv("API_RELOAD"), default=False),
        help="Enable auto-reload (useful for development)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()

    logger.info("=" * 70)
    logger.info("Starting SHL Recommender API Server")
    logger.info("=" * 70)
    logger.info("Server will listen on http://%s:%s", args.host, args.port)
    logger.info("OpenAPI docs available at /docs (if enabled)")
    logger.info("Stats endpoint available at /stats")
    logger.info("=" * 70)

    # An import string (not the app object) is required for reload and multiple workers
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )
