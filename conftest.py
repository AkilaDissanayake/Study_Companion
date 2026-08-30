"""
Pytest root conftest.

Its mere presence here makes pytest add this directory to sys.path, which lets
test modules do plain top-level imports (`import main`, `from utils...`,
`from models...`) exactly the way the app itself does. It also pins the
working directory to here, since several modules (logger, file handler,
config handler) resolve relative paths ("logs/", "configs/", "images/")
against the process cwd.
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
