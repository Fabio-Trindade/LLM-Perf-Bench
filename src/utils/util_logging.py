import logging

from src.enums.enum_logging import EnumLogging

def config_logging(logging_str: str):
    if logging_str == EnumLogging.DEBUG.value:
        log = logging.DEBUG
    elif logging_str == EnumLogging.WARN.value:
        log = logging.WARN
    elif logging_str == EnumLogging.INFO.value:
        log = logging.INFO
    elif logging_str == EnumLogging.ERROR.value:
        log = logging.ERROR
    else:
        raise ValueError(f"Invalid logging value {log}")

    logging.basicConfig(
        level=log,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )