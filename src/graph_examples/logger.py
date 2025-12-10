import logging
import os


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Create a logger based on envronment variable LOG_LEVEL.

    Args:
        name (str, optional): Name of the logger. Defaults to __name__ which is the current module name.

    Returns:
        logging.logger: Configured logger object.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        log_format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s\n",
        )

        formatter = logging.Formatter(log_format)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger
