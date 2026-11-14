"""Test-only sandbox selection; pure tests never require Linux namespaces.

reuse-checked: existing Session fixture and pytest addoption convention.
"""
import os

import pytest


def pytest_addoption(parser):
    parser.addoption('--rlm-sandbox', action='store_true',
                     default=os.environ.get('RLM_TEST_SANDBOX') == '1',
                     help='Use actual Linux bubblewrap, including isolation assertions')
    parser.addoption('--rlm-unsandboxed', action='store_true',
                     help='Explicitly select portable test workers without Linux namespaces')


def pytest_configure(config):
    if config.getoption('--rlm-sandbox') and config.getoption('--rlm-unsandboxed'):
        raise pytest.UsageError('choose either --rlm-sandbox or --rlm-unsandboxed')
