import re

def convert_timespan_str_to_float(to_convert: str) -> float:
    total = sum([int(val[0:3]) * multiplier for val, multiplier in
                 zip(re.split('[:.]', to_convert), [3600, 60, 1, 1 / 1000])])
    return total