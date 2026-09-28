import functools
import logging
import os
from enum import Enum


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):

        self.logger = logging.getLogger("mesh_processor")
        self.logger.setLevel(logging.INFO)  # Default level
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler (optional)
            log_dir = os.environ.get("LOG_DIR", "")
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(
                    os.path.join(log_dir, "mesh_processor.log")
                )
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        self.logger.propagate = False



    @staticmethod
    def get_instance():
        if Logger._instance is None:
            return Logger()
        return Logger._instance

    def set_level(self, level):
        if isinstance(level, LogLevel):
            self.logger.setLevel(level.value)
        elif isinstance(level, int):
            self.logger.setLevel(level)
        elif isinstance(level, str):
            level_map = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL,
            }
            level_value = level_map.get(level.lower(), logging.INFO)
            self.logger.setLevel(level_value)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def log_function(level=LogLevel.INFO):
    """
    Decorator to log function entry and exit.

    Args:
        level: Logging level for the function (default: INFO)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = Logger.get_instance()
            function_name = func.__name__

            # Log function entry with arguments (truncated for readability)
            arg_str = ", ".join([f"{a}" for a in args[:2]]) + (
                ", ..." if len(args) > 2 else ""
            )
            kwarg_str = ", ".join([f"{k}={v}" for k, v in list(kwargs.items())[:2]]) + (
                ", ..." if len(kwargs) > 2 else ""
            )
            params = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"

            logger.debug(f"Entering {function_name}({params})")

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Log function exit
                if level == LogLevel.DEBUG:
                    result_str = (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )
                    logger.debug(f"Exiting {function_name} -> {result_str}")
                else:
                    logger.debug(f"Exiting {function_name}")

                return result
            except Exception as e:
                logger.error(f"Exception in {function_name}: {str(e)}")
                raise

        return wrapper

    return decorator


if "LOG_LEVEL" in os.environ:
    Logger.get_instance().set_level(os.environ["LOG_LEVEL"])
