"""Lossless coarse coronary-system remapping of explicit human vessel labels.

PDA/PLV is ambiguous without dominance/branch-origin evidence. It is never
automatically assigned to the right coronary system. Missing answers stay missing.
"""
LEFT = {'left_coronary', 'lm', 'lad', 'diagonal', 'lcx', 'om_ramus'}


def coronary_system(vessel):
    if vessel in LEFT:
        return 'left_coronary'
    if vessel in {'rca', 'right_coronary'}:
        return 'right_coronary'
    if vessel == 'graft':
        return 'graft'
    if not vessel or vessel != vessel:
        return ''
    return 'uncertain'
