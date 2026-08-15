import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Return a plain logger. (Vendored from Matcha-TTS without the
    lightning rank_zero_only wrapper — Voicebox runs single-process.)"""
    return logging.getLogger(name)
