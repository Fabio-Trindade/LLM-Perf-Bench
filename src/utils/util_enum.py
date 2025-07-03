from enum import Enum

def get_string_values_list(enum: Enum):
    return [en.value for en in enum if isinstance(en.value, str)]