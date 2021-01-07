#!/usr/bin/env python
import os
import py_compile
PYTHONOPTIMIZE = "OO"


py_compile.compile("Classes_Gillespie.py", "__pycache__/Classes_Gillespie.pyc", optimize=2)
py_compile.compile("Functions_Gillespie.py", "__pycache__/Functions_Gillespie.pyc", optimize=2)
py_compile.compile("main.py", "__pycache__/main.pyc", optimize=2)
py_compile.compile("PostProcessing_Functions.py", "__pycache__/PostProcessing_Functions.pyc", optimize=2)
os.system('python __pycache__/main.pyc')
exit()
