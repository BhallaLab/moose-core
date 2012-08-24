# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Fri Aug 24 12:39:23 2012 (+0530)
#           By: subha
#     Update #: 1885
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


import warnings
import platform
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
               'element'] + sequence_types

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
    if isinstance(path, ematrix) or isinstance(path, element):
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
    print(_moose.getCwe().getPath())
    
def le(el=None):
    """List elements. """
    if el is None:
        el = getCwe()[0]
    elif isinstance(el, str):
        if not exists(el):
            raise ValueError('no such element')
        el = element(el)
    elif isinstance(el, ematrix):
        el = el[0]
    print('Elements under', el.path)
    for ch in el.children:
        print(ch.path)

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

    element -- Element or path of an existing element.

    field -- Field to be displayed. If '*', all fields are displayed.

    showtype -- If True show the data type of each field.

    """
    if isinstance(elem, str):
        if not exists(elem):
            raise ValueError('no such element')
        elem = element(elem)
    if field == '*':        
        value_field_dict = getFieldDict(elem.class_, 'valueFinfo')
        max_type_len = max([len(dtype) for dtype in list(value_field_dict.values())])
        max_field_len = max([len(dtype) for dtype in list(value_field_dict.keys())])
        print('\n[', elem.path, ']')
        for key, dtype in list(value_field_dict.items()):
            if dtype == 'bad' or key == 'this' or key == 'dummy' or key == 'me' or dtype.startswith('vector') or 'ObjId' in dtype:
                continue
            value = elem.getField(key)
            if showtype:
                typestr = dtype.ljust(max_type_len + 4)
                # The following hack is for handling both Python 2 and
                # 3. Directly putting the print command in the if/else
                # clause causes syntax error in both systems.
                print(typestr, end=" ")
            print(key.ljust(max_field_len + 4), '=', value)
    else:
        try:
            print(field, '=', elem.getField(field))
        except AttributeError:
            pass # Genesis silently ignores non existent fields

def showfields(element, showtype=False):
    """Convenience function. Should be deprecated if nobody uses it."""
    warnings.warn('Deprecated. Use showfield(element, field="*", showtype=True) instead.', DeprecationWarning)
    showfield(element, field='*', showtype=showtype)
    
def doc(arg):
    """Display the documentation for class or field in a class.

    There is no way to dynamically access the MOOSE docs using
    pydoc. (using properties requires copying all the docs strings
    from MOOSE increasing the loading time by ~3x). Hence we provide a
    separate function.
    """
    if isinstance(arg, str):
        tokens = arg.split('.')
        print_field = len(tokens) > 1
        class_path = '/classes/%s' % (tokens[0])
        if exists(class_path):
            if not print_field:
                print(Cinfo(class_path).docs)
        else:
            print('No such class:', tokens[0])
            return
    class_id = ematrix('/classes/%s' % (tokens[0]))
    num_finfo = getField(class_id[0], 'num_valueFinfo', 'unsigned')
    finfo = ematrix('/classes/%s/valueFinfo' % (tokens[0]))
    print('\n* Value Field *\n')
    for ii in range(num_finfo):
        oid = element(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print(oid.name, ':', oid.docs, '\n')
                return
        else:
            print(oid.name, ':', oid.docs, '\n')            
    num_finfo = getField(class_id[0], 'num_srcFinfo', 'unsigned')
    finfo = ematrix('/classes/%s/srcFinfo' % (tokens[0]))
    print('\n* Source Field *\n')
    for ii in range(num_finfo):
        oid = element(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print(oid.name, ':', oid.docs, '\n')
                return
        else:
            print(oid.name, ':', oid.docs, '\n')            
    num_finfo = getField(class_id[0], 'num_destFinfo', 'unsigned')
    finfo = ematrix('/classes/%s/destFinfo' % (tokens[0]))
    print('\n* Destination Field *\n')
    for ii in range(num_finfo):
        oid = element(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print(oid.name, ':', oid.docs, '\n')
                return
        else:
            print(oid.name, ':', oid.docs, '\n')
    num_finfo = getField(class_id[0], 'num_lookupFinfo', 'unsigned')    
    finfo = ematrix('/classes/%s/lookupFinfo' % (tokens[0]))
    print('\n* Lookup Field *\n')
    for ii in range(num_finfo):
        oid = element(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print(oid.name, ':', oid.docs)
                return
        else:
            print(oid.name, ':', oid.docs, '\n')
    

# 
# moose.py ends here
