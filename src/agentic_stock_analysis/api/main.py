import logging
from fastapi import FastAPI

from agentic_stock_analysis.core.log_config import setup_logging
from agentic_stock_analysis.api.routes import router
from agentic_stock_analysis.ml.model import get_model, MODEL_PATH
from agentic_stock_analysis.ml.training import ensure_model_trained


setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock Predictor API",
        description="Predicts next day stock movement and gets an AI explanation",
        version="1.0",
    )
    app.include_router(router)

    @app.on_event("startup")
    def warmup():
        """
        Warmup policy:
        - If model exists: load it (fast)
        - If missing: train it once using 5y data across a ticker universe,
        save model + dataset
        """
        try:
            if MODEL_PATH.exists():
                get_model()
                logger.info(f"Model warmup complete (loaded): {MODEL_PATH}")
            else:
                logger.warning(
                    f"Model not found at {MODEL_PATH}. Attempting training on startup..."
                )
                ensure_model_trained(
                    years=5,
                    max_tickers=400,
                    min_tickers=100,
                )
                get_model()
                logger.info("Model warmup complete (trained + loaded).")
        except Exception:
            logger.exception("Model warmup failed.")
            raise

    return app


app = create_app()
