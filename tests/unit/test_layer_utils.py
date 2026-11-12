import pytest

from pvx.utils.layer_utils import layer_from_fraction, resolve_layers


def test_middle_of_32_layers_matches_legacy_layer_16():
    assert layer_from_fraction(0.5, 32) == 16


@pytest.mark.parametrize("n,expected", [(28, 14), (42, 21), (32, 16)])
def test_middle_layer_across_depths(n, expected):
    assert layer_from_fraction(0.5, n) == expected


def test_fraction_one_is_clamped_to_last_block():
    assert layer_from_fraction(1.0, 32) == 31


def test_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        layer_from_fraction(1.5, 32)


def test_resolve_layers_explicit_only_needs_no_config():
    assert resolve_layers("any/model", layers=[16, 8, 16]) == [8, 16]


def test_resolve_layers_requires_something():
    with pytest.raises(ValueError):
        resolve_layers("any/model")
