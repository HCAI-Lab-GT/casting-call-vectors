"""Session fixtures for the tiny-model tests: one random model and one fake Assistant Axis folder."""

import pytest

from test_persona_acts_tiny import make_aa, make_tiny


@pytest.fixture(scope="session")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("persona_acts")


@pytest.fixture(scope="session")
def model(tmp):
    return make_tiny(tmp)


@pytest.fixture(scope="session")
def aa(tmp):
    return make_aa(tmp)


@pytest.fixture(scope="session")
def roles_dir(aa):
    return aa[0]


@pytest.fixture(scope="session")
def qfile(aa):
    return aa[1]
