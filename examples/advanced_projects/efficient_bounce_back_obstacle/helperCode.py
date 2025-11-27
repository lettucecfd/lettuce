import os
import sys


class Logger(object):
    def __init__(self, outdir):
        self.terminal = sys.stdout
        self.filename = os.path.join(outdir, "log")

    def write(self, message):
        self.terminal.write(message)
        f = open(self.filename, "a")
        f.write(message)
        f.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility. This handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass