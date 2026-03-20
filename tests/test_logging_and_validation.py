import logging

from perceptimg.utils import logging_config, validation


def test_configure_logging_plaintext() -> None:
    logging_config.configure_logging(json_output=False, merge=False)
    logging.getLogger("check").info("ok")


def test_validation_success_paths() -> None:
    validation.ensure_positive(1, "x")
    validation.ensure_between_0_1(0.5, "y")
    validation.ensure_non_empty([1], "z")
