# /*******************************************************************
#  * File:            pymoose.py
#  * Description:      This is a wrapper over moose.py and apart from
#  *                   exposing the functions thereof, it adds some 
#  *                   utility functions.
#  * Author:          Subhasis Ray
#  * E-mail:          ray dot subhasis at gmail dot com
#  * Created:         2008-10-12 22:50:06
#  ********************************************************************/
# /**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Messaging Object Oriented Simulation Environment,
# ** also known as GENESIS 3 base code.
# **           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
# ** It is made available under the terms of the
# ** GNU General Public License version 2
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/

from moose import *

def listmsg(pymoose_object):
    """Prints the incoming and outgoing messages of the given object."""
    obj = pymoose_object
    ret = []
    if type(pymoose_object) is type(""):
        obj = Neutral(pymoose_object)
    for msg in obj.inMessages():
        ret.append(msg)
    for msg in obj.outMessages():
        ret.append(msg)
    return ret


def showmsg(pymoose_object):
    """Prints the incoming and outgoing messages of the given object."""
    obj = pymoose_object
    if type(pymoose_object) is type(""):
        obj = Neutral(pymoose_object)
    print 'INCOMING:'
    for msg in obj.inMessages():
        print msg
    print 'OUTGOING:'
    for msg in obj.outMessages():
        print msg


def readtable(table, filename, separator=None):
    """Reads the file specified by filename to fill the MOOSE table object.

    The file can either have one float on each line, in that case the
    table will be filled with values sequentially.
    Or, the file can have 
    index value
    on each line. If the separator between index and value is anything other than
    white space, you can specify it in the separator argument."""

    in_file = open(filename)
    ii = 0
    line_no = 0
    for line in in_file:
        line_no = line_no + 1
        tokens = split(line, separator)
        if len(token) is 0:
            continue
        elif len(token) == 1:
            table[ii] = float(token[0])
        elif len(token) == 2:
            table[int(token[0])] = float(token[1])
        else:
            print "pymoose.readTable(", table, ",", filename, ",", separator, ") - line#", line_no, " does not fit." 

def getfields(moose_object):
    """Returns a dictionary of the fields (ValueFinfo) in this
    object."""
    fields = {}
    c_dict = moose_object.__class__.__dict__
    for key, value in c_dict.items():
        if type(value) is property:
            try:
                getter = c_dict["_" + moose_object.__class__.__name__ + "__get_" + key]
                fields[key] = getattr(moose_object, key)
            except KeyError:
                pass

    return fields


def printtree(root, vchar='|', hchar='__', vcount=1, depth=0, prefix='', is_last=False):
    """Pretty-print a MOOSE tree.
    
    root - the root element of the MOOSE tree, must be some derivatine of Neutral.

    vchar - the character printed to indicate vertical continuation of
    a parent child relationship.

    hchar - the character printed just before the node name

    vcount - determines how many lines will be printed between two
    successive nodes.

    depth - for internal use - should not be explicitly passed.

    prefix - for internal use - should not be explicitly passed.

    is_last - for internal use - should not be explicitly passed.

    """
    if isinstance(root, str) or isinstance(root, Id):
        root = Neutral(root)

    for i in range(vcount):
        print prefix

    if depth != 0:
        print prefix + hchar,
        if is_last:
            index = prefix.rfind(vchar)
            prefix = prefix[:index] + ' ' * (len(hchar) + len(vchar)) + vchar
        else:
            prefix = prefix + ' ' * len(hchar) + vchar
    else:
        prefix = prefix + vchar

    print root.name
    
    children = [ Neutral(child) for child in root.children() ]
    for i in range(0, len(children) - 1):
        printtree(children[i],
                  vchar, hchar, vcount, depth + 1, 
                  prefix, False)
    if len(children) > 0:
        printtree(children[-1], vchar, hchar, vcount, depth + 1, prefix, True)


def df_traverse(root, operation, *args):
    """Traverse the tree in a depth-first manner and apply the
    operation using *args. The first argument is the root object by
    default.""" 
    if hasattr(root, '_visited'):
        return
    operation(root, *args)
    for child in root.children():
        childNode = Neutral(child)
        df_traverse(childNode, operation, *args)
    root._visited = True

def readcell_scrambled(filename, target):
    """A special version for handling cases where a .p file has a line
    with specified parent yet to be defined.

    It creates a temporary file with a sorted version based on
    connectivity, so that parent is always defined before child."""
    pfile = open(filename, "r")
    tmpfilename = filename + ".tmp"
    graph = defaultdict(list)
    data = {}
    error = None
    root = None
    for line in pfile:
        tmpline = line.strip()
        if tmpline.startswith("*") or tmpline.startswith("//"):
            continue
        elif tmpline.startswith("/*"):
            error = "Handling C style comments not implemented."
            break
        node, parent, rest, = tmpline.split(None, 2)
        print node, parent
        if (parent == "none"):
            if (root is None):
                root = node
            else:
                error = "Duplicate root elements: ", root, node, "> Cannot process any further."
                break
        graph[parent].append(node)
        data[node] = line
    if error is not None:
        print error
        return None

    tmpfile = open(tmpfilename, "w")
    stack = [root]
    while stack:
        current = stack.pop()
        children = graph[current]
        stack.extend(children)
        tmpfile.write(data[current])
    tmpfile.close()
    PyMooseBase.getContext().readCell(tmpfilename, target)
    return Cell(target)

if __name__ == "__main__": # test printtree
    s = Neutral('cell')
    soma = Neutral('soma', s)
    d1 = Neutral('d1', soma)
    d2 = Neutral('d2', soma)
    d3 = Neutral('d3', d1)
    d4 = Neutral('d4', d1)
    d5 = Neutral('d5', s)
    printtree(s)

    expected = 'cell            \
                |               \
                |__soma         \
                |  |            \
                |  |__d1        \
                |  |  |         \
                |  |  |__d3     \
                |  |  |         \
                |  |  |__d4     \
                |  |            \
                |  |__d2        \
                |               \
                |__d5'

    s1 = Neutral('cell1')
    c1 = Neutral('c1', s1)
    c2 = Neutral('c2', c1)
    c3 = Neutral('c3', c1)
    c4 = Neutral('c4', c2)
    c5 = Neutral('c5', c3)
    c6 = Neutral('c6', c3)
    c7 = Neutral('c7', c4)
    c8 = Neutral('c8', c5)
    printtree(s1)
    expected1 = 'cell1                  \
                 |                      \
                 |__c1                  \
                    |                   \
                    |__c2               \
                    |  |                \
                    |  |__c4            \
                    |     |             \
                    |     |__c7         \
                    |                   \
                    |__c3               \
                       |                \
                       |__c5            \
                       |  |             \
                       |  |__c8         \
                       |                \
                       |__c6'
