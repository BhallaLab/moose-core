# -*- coding: utf-8 -*-


"""methods_utils.py:
    Some non-standard functions generic to moose.

    This library may not be exposed to end-users. Intended for development by
    the maintainer of this file.

Last modified: Sat Jan 18, 2014  05:01PM

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


import re

objPathPat = re.compile(r'(\/\w+\[\d+\])+?$')

def idPathToObjPath( idPath ):
    """ Append a [0] if missing from idPath.

    Id-paths do not have [0] at their end. This does not allow one to do
    algebra properly.
    """
    m = objPathPat.match( idPath )
    if m: return idPath
    else:
        return '{}[0]'.format(idPath)

# This one is from  https://stackoverflow.com/a/377028/1805129
# Checks if a command exists.
def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def main():
    p1 = '/cable[0]/comp_[1]/a'
    p2 = '/cab[1]/comp/com'
    p3 = '/cab[1]/p[2]/c[3]'
    p4 = '/ca__b[1]/_p[2]/c[122]'
    for p in [p1, p2, p3, p4]:
        m = objPathPat.match(p)
        if m:
            print(m.group(0))
        else:
            print(("{} is invalid Obj path in moose".format( p )))

if __name__ == '__main__':
    main()
