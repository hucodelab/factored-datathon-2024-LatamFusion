import pytest

# Example Test Suite


def test_addition():
    assert 1 + 1 == 2


def test_string_length():
    sample_string = "Hello, World!"
    assert len(sample_string) == 13


def test_list_contains_value():
    sample_list = [1, 2, 3, 4, 5]
    assert 3 in sample_list
