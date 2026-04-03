"""Server entrypoint for HF multi-mode deployment."""

import os
import uvicorn
from server import app


def main():
    """Launch FastAPI server for Hugging Face Space."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
