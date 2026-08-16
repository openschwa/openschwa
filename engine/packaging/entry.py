"""PyInstaller entry script.

A bundle needs a real script to start from, not a console-script name. Keep it
empty of logic: everything it needs lives in the package so the frozen and
installed paths cannot drift.
"""

import multiprocessing

from openschwa_engine.app import main

if __name__ == "__main__":
    # Required before anything spawns: without it a frozen child process
    # re-executes the launcher instead of the worker, forking endlessly.
    multiprocessing.freeze_support()
    main()
