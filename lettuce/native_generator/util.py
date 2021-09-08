class AbstractMethodInvokedError(NotImplementedError):
    def __init__(self):
        import inspect
        method_name = inspect.getouterframes(inspect.currentframe())[1].function
        super().__init__(f"Abstract method `{method_name}` can not be invoked")
