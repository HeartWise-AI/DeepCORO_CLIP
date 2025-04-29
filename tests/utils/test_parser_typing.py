import pytest
import argparse
import json
from utils.parser_typing import str2bool, parse_dict, parse_list, parse_optional_int, parse_optional_str


class TestStr2Bool:
    def test_boolean_inputs(self):
        assert str2bool(True) is True
        assert str2bool(False) is False

    def test_true_string_values(self):
        true_values = ['yes', 'true', 't', 'y', '1', 
                       'YES', 'TRUE', 'T', 'Y', 'True', 'Yes']
        for val in true_values:
            assert str2bool(val) is True

    def test_false_string_values(self):
        false_values = ['no', 'false', 'f', 'n', '0',
                        'NO', 'FALSE', 'F', 'N', 'False', 'No']
        for val in false_values:
            assert str2bool(val) is False

    def test_invalid_inputs(self):
        invalid_values = ['maybe', 'invalid', '2', 'null']
        for val in invalid_values:
            with pytest.raises(argparse.ArgumentTypeError, match='Boolean value expected.'):
                str2bool(val)


class TestParseDict:
    def test_valid_json(self):
        input_str = '{"key": "value", "number": 42}'
        expected = {"key": "value", "number": 42}
        assert parse_dict(input_str) == expected

    def test_empty_dict(self):
        input_str = '{}'
        assert parse_dict(input_str) == {}

    def test_nested_dict(self):
        input_str = '{"outer": {"inner": "value"}}'
        expected = {"outer": {"inner": "value"}}
        assert parse_dict(input_str) == expected

    def test_invalid_json(self):
        invalid_values = [
            '{key: value}',  # Missing quotes
            '{',  # Incomplete JSON
            'not a json',  # Not JSON at all
        ]
        for val in invalid_values:
            with pytest.raises(argparse.ArgumentTypeError, match='Invalid JSON string'):
                parse_dict(val)


class TestParseList:
    def test_list_input(self):
        input_list = [1, 2, 3]
        assert parse_list(input_list) == [1, 2, 3]

    def test_comma_separated_string(self):
        input_str = "1,2,3"
        assert parse_list(input_str) == [1, 2, 3]

    def test_bracketed_string(self):
        input_str = "[1, 2, 3]"
        assert parse_list(input_str) == [1, 2, 3]

    def test_quoted_string(self):
        input_str = "'1,2,3'"
        assert parse_list(input_str) == [1, 2, 3]

    def test_space_in_string(self):
        input_str = "1, 2, 3"
        assert parse_list(input_str) == [1, 2, 3]

    def test_empty_string(self):
        assert parse_list("") == []
        assert parse_list("[]") == []
        assert parse_list("''") == []

    def test_invalid_inputs(self):
        invalid_values = [
            "a,b,c",  # Non-integers
            "1,b,3",  # Mixed types
            "1.2,3.4",  # Floats
        ]
        for val in invalid_values:
            with pytest.raises(argparse.ArgumentTypeError, match='Invalid list of integers'):
                parse_list(val)


class TestParseOptionalInt:
    def test_valid_integer(self):
        assert parse_optional_int("42") == 42
        assert parse_optional_int("-5") == -5
        assert parse_optional_int("0") == 0

    def test_none_values(self):
        assert parse_optional_int("none") is None
        assert parse_optional_int("None") is None
        assert parse_optional_int("NONE") is None
        assert parse_optional_int("") is None

    def test_invalid_inputs(self):
        invalid_values = [
            "not_an_int",
            "3.14",
            "true",
        ]
        for val in invalid_values:
            with pytest.raises(argparse.ArgumentTypeError, match='Invalid integer or None value'):
                parse_optional_int(val)


class TestParseOptionalStr:
    def test_regular_string(self):
        assert parse_optional_str("hello") == "hello"
        assert parse_optional_str("42") == "42"
        assert parse_optional_str("True") == "True"

    def test_none_values(self):
        assert parse_optional_str("none") is None
        assert parse_optional_str("None") is None
        assert parse_optional_str("NONE") is None
        assert parse_optional_str("") is None 