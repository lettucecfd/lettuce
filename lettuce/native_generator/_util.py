class AbstractMethodInvokedError(NotImplementedError):
    def __init__(self):
        import inspect
        method_name = inspect.getouterframes(inspect.currentframe())[1].function
        super().__init__(f"Abstract method `{method_name}` can not be invoked")


def _load_template(name: str):
    import os
    path = os.path.join(__file__, '..', f"_{name}.template")

    with open(path, 'r') as file:
        return file.read()


def _pretty_print_c(buffer: str):
    import re
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


def _pretty_print_py(buffer: str):
    import re
    # remove spaces before new line
    buffer = re.sub(r' +\n', '\n', buffer)
    # remove whitespace at end and begin
    buffer = re.sub(r'\n+$', '\n', buffer)
    buffer = re.sub(r'^\n*', '', buffer)
    # remove whitespace at end and begin
    buffer = re.sub(r'\s*\ndef ', '\n\n\ndef ', buffer)
    return buffer
