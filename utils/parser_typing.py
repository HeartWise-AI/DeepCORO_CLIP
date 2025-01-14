import json
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_dict(string):
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError('Invalid JSON string')

def parse_list(string):
    try:
        if isinstance(string, list):
            return [int(x) for x in string]
        cleaned = string.strip('[]"\' ').strip()
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(',')]
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(f'Invalid list of integers: {string}. Error: {str(e)}')

def parse_optional_int(string):
    if string.lower() == 'none' or string == '':
        return None
    try:
        return int(string)
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(f'Invalid integer or None value: {string}. Error: {str(e)}') 

def parse_optional_str(string):
    if string.lower() == 'none' or string == '':
        return None
    return string
