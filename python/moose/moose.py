# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Tue Sep 18 14:29:22 2012 (+0530)
#           By: subha
#     Update #: 2020
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

######################################################################
# Special function to generate objects of the right class from
# a given path.
#
# As of 2012-08-22, element() function has been renamed to_el because
# ObjId is now called element in pymoose. This function is not
# entirely deprecated as we do not yet have a way to call the
# destFields as functions from the base class.
######################################################################

def to_el(path):
    """Return a reference to an existing object as an instance of the
    appropriate class. If path does not exist, raises NameError.

    ematrix or element can be provided in stead of path"""
    if isinstance(path, ematrix) or isinstance(path, melement):
        classObj = eval(path.class_)
    elif isinstance(path, str):
        if not _moose.exists(path):
            raise NameError('Object %s not defined' % (path))
        oid = _moose.element(path)
        classObj = eval(oid.class_)
    else:
        raise TypeError('expected argument: ematrix/element/str')
    return classObj(path)

def arrayelement(path, className='Neutral'):
    """Return a reference to an existing object as an instance of the
    right class. If path does not exist, className is used for
    creating an instance of that class with the given path"""
    warnings.warn('use element.ematrix() to retrieve its container. \
ematrix instances can be used directly for getting \
tuple of the field values of its elements.', 
                  DeprecationWarning)
    if not exists(path):
        raise NameError('Object %s not defined' % (path))
    return ematrix(path)

################################################################
# Wrappers for global functions
################################################################ 
    
def pwe():
    """Print present working element. Convenience function for GENESIS
    users."""
    print _moose.getCwe().getPath()
    
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
        value_field_dict = getFieldDict(elem.class_, 'valueFinfo')
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
    
def doc(arg):
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

    """
    # There is no way to dynamically access the MOOSE docs using
    # pydoc. (using properties requires copying all the docs strings
    # from MOOSE increasing the loading time by ~3x). Hence we provide a
    # separate function.
    docstring = cStringIO.StringIO()
    indent = '    '
    tokens = None
    print_field = False
    if isinstance(arg, str):
        tokens = arg.split('.')
        if tokens[0] == 'moose':
            tokens = tokens[1:]
        print_field = len(tokens) > 1
    elif isinstance(arg, type):
        tokens = [arg.__name__]
    elif isinstance(arg, melement) or isinstance(arg, ematrix):
        tokens = [arg.class_]
    else:
        raise TypeError('Require a string or a moose class or a moose object')
    class_path = '/classes/%s' % (tokens[0])
    if exists(class_path):
        if not print_field:
            docstring.write('%s\n' % (Cinfo(class_path).docs))
    else:
        raise NameError('name \'%s\' not defined.' % (tokens[0]))
    class_id = ematrix('/classes/%s' % (tokens[0]))
    finfotypes = ['valueFinfo', 'srcFinfo', 'destFinfo', 'lookupFinfo']
    readablefinfotypes = ['value field', 'source field', 'destination field', 'lookup field']
    if print_field:
        fieldfound = False
        for rtype, ftype in zip(readablefinfotypes, finfotypes):
            numfinfo = getField(class_id[0], 'num_'+ftype, 'unsigned')
            finfo = ematrix('/classes/%s/%s' % (tokens[0], ftype))
            for ii in range(numfinfo):
                oid = melement(finfo, 0, ii, 0)
                if oid.name == tokens[1]:
                    docstring.write('%s%s.%s: %s - %s\n' % (indent, tokens[0], tokens[1], oid.docs, rtype))
                    fieldfound = True
                    break
            if fieldfound:
                break
    else:
        finfoheaders = ['* Value Fields *', '* Source Fields *', '* Destination Fields *', '* Lookup Fields *']
        for header, ftype in zip(finfoheaders, finfotypes):
            docstring.write('\n%s\n' % (header))
            numfinfo = getField(class_id[0], 'num_'+ftype, 'unsigned')
            finfo = ematrix('/classes/%s/%s' % (tokens[0], ftype))
            for ii in range(numfinfo):
                oid = melement(finfo, 0, ii, 0)
                docstring.write('%s%s: %s\n' % (indent, oid.name, oid.type))
            
    pydoc.pager(docstring.getvalue())
    docstring.close()
                

# 
# moose.py ends here
