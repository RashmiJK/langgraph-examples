import subprocess
import sys
from pathlib import Path

from graph_examples.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the Streamlit app."""
    app_path = Path(__file__).parent / "product_review_app.py"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("Streamlit interrupted by user. Closing main process.")
    finally:
        logger.info("Closing main process.")


if __name__ == "__main__":
    main()
