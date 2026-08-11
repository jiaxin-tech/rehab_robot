"""Project logging without import-time filesystem writes."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a console logger; experiment files are owned by episode loggers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加handler

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
