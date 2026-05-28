import utils.text
from app.handler import Handler, make_handler


def helper():
    return 1


def main():
    h = make_handler("world")
    msg = utils.text.upper("hi")
    Handler(msg)
    helper()
