"""helper.py: 

Some helper functions which are compatible with both python2 and python3.
"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import os
import re
import subprocess

def execute(cmd):
    """execute: Execute a given command.

    :param cmd: string, given command.

    Return:
    ------
        Return a iterator over output.
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def find_files( dirname, name_contains=None, text_regex_search=None):
    files = []
    for d, sd, fs in os.walk(dirname):
        for f in fs:
            fpath = os.path.join(d,f)
            include = True
            if name_contains:
                if name_contains not in os.path.basename(f):
                    include = False
            if text_regex_search:
                with open(fpath, 'r' ) as f:
                    txt = f.read()
                    if re.search(text_regex_search, txt) is None:
                        include = False
            if include:
                files.append(fpath)
    return files
