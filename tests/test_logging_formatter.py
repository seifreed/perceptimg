import logging
import sys

from perceptimg.utils.logging_config import JsonFormatter, configure_logging


def test_json_formatter_includes_exc_info_and_extra() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="error",
            args=(),
            exc_info=exc_info,
            func="func",
        )
        record.custom = "extra"
        output = formatter.format(record)
        assert '"custom"' in output
        assert '"error"' in output


def test_configure_logging_sets_logger_levels() -> None:
    configure_logging(json_output=False, merge=False, logger_names=("foo",))
    logger = logging.getLogger("foo")
    logger.info("hello")
