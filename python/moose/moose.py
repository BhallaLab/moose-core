# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Thu Oct  4 20:04:06 2012 (+0530)
#           By: subha
#     Update #: 2170
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 

# Code:

import cStringIO
import warnings
import platform
import pydoc
_py3k = False
if int(platform.python_version_tuple()[0]) >= 3:
    _py3k = True
from collections import defaultdict
from . import _moose
from ._moose import *
import __main__ as main

sequence_types = [ 'vector<double>',
                   'vector<int>',
                   'vector<long>',
                   'vector<unsigned int>',
                   'vector<float>',
                   'vector<unsigned long>',
                   'vector<short>',
                   'vector<Id>',
                   'vector<ObjId>' ]
known_types = ['void',
               'char',
               'short',
               'int',
               'unsigned int',
               'double',
               'float',
               'long',
               'unsigned long',
               'string',
               'ematrix',
               'melement'] + sequence_types

################################################################
# Wrappers for global functions
################################################################ 
    
def pwe():
    """Print present working element. Convenience function for GENESIS
    users. If you want to retrieve the element in stead of printing
    the path, use moose.getCwe()

    """
    pwe_ = _moose.getCwe()
    print pwe_.getPath()
    return pwe_
    
def le(el=None):
    """List elements. 
    
    Parameters
    ----------
    el: str/melement/ematrix/None
    The element or the path under which to look. If `None`, children
    of current working element are displayed.
    """
    if el is None:
        el = getCwe()[0]
    elif isinstance(el, str):
        if not exists(el):
            raise ValueError('no such element')
        el = element(el)
    elif isinstance(el, ematrix):
        el = el[0]    
    print 'Elements under', el.path
    for ch in el.children:
        print ch.path

ce = setCwe # ce is a GENESIS shorthand for change element.

def syncDataHandler(target):
    """Synchronize data handlers for target.

    Parameter:
    target -- target element or path or ematrix.
    """
    raise NotImplementedError('The implementation is not working for IntFire - goes to invalid objects. \
First fix that issue with SynBase or something in that line.')
    if isinstance(target, str):
        if not _moose.exists(target):
            raise ValueError('%s: element does not exist.' % (target))
        target = ematrix(target)
        _moose.syncDataHandler(target)

def showfield(elem, field='*', showtype=False):
    """Show the fields of the element, their data types and values in
    human readable format. Convenience function for GENESIS users.

    Parameters:

    elem: str/melement instance
    Element or path of an existing element.

    field: str
    Field to be displayed. If '*', all fields are displayed.

    showtype: bool
    If True show the data type of each field.

    """
    if isinstance(elem, str):
        if not exists(elem):
            raise ValueError('no such element')
        elem = element(elem)
    if field == '*':        
        value_field_dict = getFieldDict(elem.className, 'valueFinfo')
        max_type_len = max([len(dtype) for dtype in list(value_field_dict.values())])
        max_field_len = max([len(dtype) for dtype in list(value_field_dict.keys())])
        print '\n[', elem.path, ']'
        for key, dtype in list(value_field_dict.items()):
            if dtype == 'bad' or key == 'this' or key == 'dummy' or key == 'me' or dtype.startswith('vector') or 'ObjId' in dtype:
                continue
            value = elem.getField(key)
            if showtype:
                typestr = dtype.ljust(max_type_len + 4)
                # The following hack is for handling both Python 2 and
                # 3. Directly putting the print command in the if/else
                # clause causes syntax error in both systems.
                print typestr,
            print key.ljust(max_field_len + 4), '=', value
    else:
        try:
            print field, '=', elem.getField(field)
        except AttributeError:
            pass # Genesis silently ignores non existent fields

def showfields(element, showtype=False):
    """Convenience function. Should be deprecated if nobody uses it."""
    warnings.warn('Deprecated. Use showfield(element, field="*", showtype=True) instead.', DeprecationWarning)
    showfield(element, field='*', showtype=showtype)
    
finfotypes = [('valueFinfo', 'value field') , 
              ('srcFinfo', 'source message field'),
              ('destFinfo', 'destination message field'),
              ('sharedFinfo', 'shared message field'),
              ('lookupFinfo', 'lookup field')]

# 2012-01-11 19:20:39 (+0530) Subha: checked for compatibility with dh_branch
# 2012-09-27 19:26:30 (+0530) Subha: updated for compatibility with buildQ branch
def listmsg(pymoose_object):
    """Return a list containing the incoming and outgoing messages of
    the given object."""
    obj = pymoose_object
    ret = []
    if type(pymoose_object) is type(""):
        obj = _moose.Neutral(pymoose_object)
    for msg in obj.inMsg:
        ret.append(msg)
    for msg in obj.outMsg:
        ret.append(msg)
    return ret

# 2012-01-11 19:20:39 (+0530) Subha: checked for compatibility with dh_branch
# 2012-09-27 19:26:30 (+0530) Subha: updated for compatibility with buildQ branch
def showmsg(pymoose_object):
    """Prints the incoming and outgoing messages of the given object."""
    obj = pymoose_object
    if type(pymoose_object) is type(""):
        obj = _moose.Neutral(pymoose_object)
    print 'INCOMING:'
    for msg in obj.msgIn:
        print msg.e2.path, msg.destFieldsOnE2, '<---', msg.e1.path, msg.srcFieldsOnE1
    print 'OUTGOING:'
    for msg in obj.msgOut:
        print msg.e1.path, msg.srcFieldsOnE1, '--->', msg.e2.path, msg.destFieldsOnE2

def getfielddoc(tokens, indent=''):
    """Get the documentation for field specified by
    tokens.

    tokens should be a two element list/tuple where first element is a
    MOOSE class name and second is the field name.
    """
    assert(len(tokens) > 1)
    for ftype, rtype in finfotypes:
        cel = _moose.element('/classes/'+tokens[0])
        numfinfo = getField(cel, 'num_'+ftype, 'unsigned')
        finfo = element('/classes/%s/%s' % (tokens[0], ftype))
        for ii in range(numfinfo):
            oid = melement(finfo.getId(), 0, ii, 0)
            if oid.name == tokens[1]:
                return '%s%s.%s: %s - %s\n\t%s\n' % \
                    (indent, tokens[0], tokens[1], 
                     oid.type, rtype, oid.docs)    
    raise NameError('`%s` has no field called `%s`' 
                    % (tokens[0], tokens[1]))
                    
    
def getmoosedoc(tokens):
    """Retrieve MOOSE builtin documentation for tokens.
    
    tokens is a list or tuple containing: (classname, [fieldname])"""
    indent = '    '
    docstring = cStringIO.StringIO()
    if not tokens:
        return ""
    class_path = '/classes/%s' % (tokens[0])
    if exists(class_path):
        if len(tokens) == 1:
            docstring.write('%s\n' % (Cinfo(class_path).docs))
    else:
        raise NameError('name \'%s\' not defined.' % (tokens[0]))
    class_id = ematrix('/classes/%s' % (tokens[0]))
    if len(tokens) > 1:
        docstring.write(getfielddoc(tokens))
    else:
        for ftype, rname in finfotypes:
            docstring.write('\n*%s*\n' % (rname.capitalize()))
            numfinfo = getField(class_id[0], 'num_'+ftype, 'unsigned')
            finfo = ematrix('/classes/%s/%s' % (tokens[0], ftype))
            for ii in range(numfinfo):
                oid = melement(finfo, 0, ii, 0)
                docstring.write('%s%s: %s\n' % 
                                (indent, oid.name, oid.type))
    ret = docstring.getvalue()
    docstring.close()
    return ret

# the global pager is set from pydoc even if the user asks for paged
# help once. this is to strike a balance between GENESIS user's
# expectation of control returning to command line after printing the
# help and python user's expectation of seeing the help via more/less.
pager=None

def doc(arg, paged=False):
    """Display the documentation for class or field in a class.
    
    Parameters
    ----------
    arg: str or moose class or instance of melement or instance of ematrix

    argument can be a string specifying a moose class name and a field
    name separated by a dot. e.g., 'Neutral.name'. Prepending `moose.`
    is allowed. Thus moose.doc('moose.Neutral.name') is equivalent to
    the above.
    
    argument can also be string specifying just a moose class name or
    a moose class or a moose object (instance of melement or ematrix
    or there subclasses). In that case, the builtin documentation for
    the corresponding moose class is displayed.

    paged: bool
    
    Whether to display the docs via builtin pager or print and
    exit. If not specified, it defaults to False and moose.doc(xyz)
    will print help on xyz and return control to command line.

    """
    # There is no way to dynamically access the MOOSE docs using
    # pydoc. (using properties requires copying all the docs strings
    # from MOOSE increasing the loading time by ~3x). Hence we provide a
    # separate function.
    global pager
    if paged and pager is None:
        pager = pydoc.pager
    tokens = []
    text = ''
    if isinstance(arg, str):
        tokens = arg.split('.')
        if tokens[0] == 'moose':
            tokens = tokens[1:]
    elif isinstance(arg, type):
        tokens = [arg.__name__]
    elif isinstance(arg, melement) or isinstance(arg, ematrix):
        text = '%s: %s\n\n' % (arg.path, arg.className)
        tokens = [arg.className]
    if tokens:
        text += getmoosedoc(tokens)
    else:
        text += pydoc.gethelp(arg)
    if pager:
        pager(text)
    else:
        print text
                

# 
# moose.py ends here
