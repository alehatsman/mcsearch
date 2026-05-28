class Handler:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return self.format(self.name)

    def format(self, x):
        return "Hello, " + x


def make_handler(name):
    return Handler(name)
