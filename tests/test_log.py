# -*- coding: utf-8 -*-

import logging

import pytest

from hp_optimizer.utils.log import configure_logger


def create_log_records():
    hp_optimizer_logger = logging.getLogger('hp_optimizer')
    foo_logger = logging.getLogger('hp_optimizer.foo')
    foobar_logger = logging.getLogger('hp_optimizer.foo.bar')

    hp_optimizer_logger.info('Welcome to HP-Optimizer')
    hp_optimizer_logger.debug('Generating project from pytest-plugin')
    foo_logger.info('Loading user config from home dir')
    foobar_logger.debug("I don't know.")
    foobar_logger.debug('I wanted to save the world.')
    foo_logger.error('Aw, snap! Something went wrong')
    hp_optimizer_logger.debug('Successfully generated project')


@pytest.fixture
def info_messages():
    return [
        'INFO: Welcome to HP-Optimizer',
        'INFO: Loading user config from home dir',
        'ERROR: Aw, snap! Something went wrong',
    ]


@pytest.fixture
def debug_messages():
    return [
        'INFO hp_optimizer: '
        'Welcome to HP-Optimizer',

        'DEBUG hp_optimizer: '
        'Generating project from pytest-plugin',

        'INFO hp_optimizer.foo: '
        'Loading user config from home dir',

        "DEBUG hp_optimizer.foo.bar: "
        "I don't know.",

        'DEBUG hp_optimizer.foo.bar: '
        'I wanted to save the world.',

        'ERROR hp_optimizer.foo: '
        'Aw, snap! Something went wrong',

        'DEBUG hp_optimizer: '
        'Successfully generated project',
    ]


@pytest.fixture
def info_logger():
    return configure_logger(stream_level='INFO')


@pytest.fixture
def debug_logger():
    return configure_logger(stream_level='DEBUG')


@pytest.fixture
def log_file(tmpdir):
    return tmpdir / 'pytest-plugin.log'


@pytest.fixture
def info_logger_with_file(log_file):
    return configure_logger(
        stream_level='INFO',
        log_file=str(log_file),
    )


def test_info_stdout_logging(caplog, info_logger, info_messages):
    """Test that stdout logs use info format and level."""
    [stream_handler] = info_logger.handlers
    assert isinstance(stream_handler, logging.StreamHandler)
    assert stream_handler.level == logging.INFO

    create_log_records()

    stream_messages = [
        stream_handler.format(r)
        for r in caplog.records
        if r.levelno >= stream_handler.level
    ]

    assert stream_messages == info_messages


def test_debug_stdout_logging(caplog, debug_logger, debug_messages):
    """Test that stdout logs use debug format and level."""
    [stream_handler] = debug_logger.handlers
    assert isinstance(stream_handler, logging.StreamHandler)
    assert stream_handler.level == logging.DEBUG

    create_log_records()

    stream_messages = [
        stream_handler.format(r)
        for r in caplog.records
        if r.levelno >= stream_handler.level
    ]

    assert stream_messages == debug_messages


def test_log_file_logging(
        caplog, info_logger_with_file, log_file, debug_messages):
    """Test that logging to stdout uses a different format and level than
    the the file handler.
    """

    [file_handler, stream_handler] = info_logger_with_file.handlers
    assert isinstance(file_handler, logging.FileHandler)
    assert isinstance(stream_handler, logging.StreamHandler)
    assert stream_handler.level == logging.INFO
    assert file_handler.level == logging.DEBUG

    create_log_records()

    assert log_file.exists()

    # Last line in the log file is an empty line
    assert log_file.readlines(cr=False) == debug_messages + ['']
