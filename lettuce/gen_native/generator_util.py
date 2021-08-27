import re


def pretty_print_c_(buffer: str):
    # remove spaces before new line
    buffer = re.sub(r' +\n', '\n', buffer)
    # remove multiple empty lines
    buffer = re.sub(r'\n\n\n+', '\n\n', buffer)
    # remove whitespace at end and begin
    buffer = re.sub(r'\n+$', '\n', buffer)
    buffer = re.sub(r'^\n*', '', buffer)
    # place preprocessor directives at start of line
    buffer = re.sub(r'\n +#', '\n#', buffer)
    # remove unnecessary whitespace between closures
    buffer = re.sub(r'{\n\n+', '{\n', buffer)
    buffer = re.sub(r'}\n\n+(\s*)}', r'}\n\1}', buffer)
    return buffer


def pretty_print_py_(buffer: str):
    # remove spaces before new line
    buffer = re.sub(r' +\n', '\n', buffer)
    # remove whitespace at end and begin
    buffer = re.sub(r'\n+$', '\n', buffer)
    buffer = re.sub(r'^\n*', '', buffer)
    # remove whitespace at end and begin
    buffer = re.sub(r'\s*\ndef ', '\n\n\ndef ', buffer)
    return buffer
