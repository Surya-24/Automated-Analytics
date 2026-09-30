import os
import logging
from typing import Optional, List, Any
from dotenv import load_dotenv

load_dotenv()

# Setup standard structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AI_Data_Agent")

# 1. Pydantic Logfire Initialization
_logfire_initialized = False
try:
    import logfire
    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(token=logfire_token)
        try:
            logfire.instrument_pydantic()
        except Exception:
            pass
        _logfire_initialized = True
        logger.info("⚡ Pydantic Logfire successfully configured with cloud token.")
    else:
        # Local console instrumentation mode if no cloud token provided
        logfire.configure(send_to_logfire=False)
        try:
            logfire.instrument_pydantic()
        except Exception:
            pass
        _logfire_initialized = True
        logger.info("ℹ️ Logfire running in local capture mode (set LOGFIRE_TOKEN for cloud dashboard).")
except Exception as e:
    logger.warning(f"Logfire initialization skipped: {e}")


# 2. Langfuse Callback Handler Initialization
def get_langfuse_callback() -> Optional[Any]:
    """
    Returns a LangfuseCallbackHandler if keys are configured, otherwise None.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    
    if public_key and secret_key:
        try:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            return handler
        except Exception as e:
            logger.warning(f"Failed to instantiate Langfuse CallbackHandler: {e}")
            return None
    return None


# 3. LangSmith Tracing Verification
def check_langsmith_status() -> bool:
    """
    Returns True if LangSmith environment variables are enabled.
    """
    tracing_v2 = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ["true", "1"]
    api_key = bool(os.environ.get("LANGCHAIN_API_KEY"))
    return tracing_v2 and api_key


def get_active_callbacks() -> List[Any]:
    """
    Consolidates active callbacks (Langfuse, etc.) to pass into LangGraph / LangChain invocations.
    """
    callbacks = []
    lf_handler = get_langfuse_callback()
    if lf_handler:
        callbacks.append(lf_handler)
    return callbacks
