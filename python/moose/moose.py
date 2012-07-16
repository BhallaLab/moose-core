# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Mon Jul 16 22:52:53 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 1842
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
from collections import defaultdict
import _moose
from _moose import *
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
               'Id',
               'ObjId'] + sequence_types

# Store a set of moose builtin classes
moose_classes = set([ch[0].name for ch in ObjId('/classes').children])

class NeutralArray(_moose.Id):
    """
    The base class. Each NeutralArray object has a unique Id (field
    id_) and that is the only data directly visible under Python. All
    operation are done on the objects by calling functions on the Id.
    
    A NeutralArray object is actually an array. The individual
    elements in a NeutralArray are of class Neutral. To access these
    individual elements, you can index the NeutralArray object.

    """
    def __init__(self, *args, **kwargs):
        """
        A NeutralArray object can be constructed in many ways. The
        most basic one being:

        neutral = moose.NeutralArray('my_neutral_object', (3))

        This will create a NeutralArray object with name
        'my_neutral_object' containing 3 elements. The object will be
        created as a child of the current working entity. Any class
        derived from NeutralArray can also be created using the same
        constructor. Actually it takes keyword parameters to do that:

        intfire = moose.NeutralArray(path='/my_neutral_object', dims=(3), type='IntFire')

        will create an IntFire object of size 3 as a child of the root entity.

        If the above code is already executed,

        duplicate = moose.NeutralArray(intfire)

        will create a duplicate reference to the existing intfire
        object. They will share the same Id and any changes made via
        the MOOSE API to one will be effective on the other.
        
        """
        try:
            Id.__init__(self, *args, **kwargs)
        except TypeError:
            path = None
            dims = (1)
            className = None
            if kwargs:
                try:
                    path = kwargs['path']
                except KeyError:
                    pass
                try:
                    dims = kwargs['dims']
                except KeyError:
                    pass
                try:
                    className = kwargs['dtype']
                except KeyError:
                    pass
                
            if path is None:
                if not args:
                    raise TypeError("NeutralArray.__init__(path, dims=(1), dtype='Neutral'): required"
                                    " argument `path` not specified.")
                path = args[0]
            if len(args) > 1:
                dims = args[1]
            if len(args) > 2:
                className = args[2]
            # Create a new ArrayElement
            _base_class = self.__class__
            if className is None:
                className = self.__class__.__name__
            if className.endswith('Array'):
                className = className[:-len('Array')]
                # For classes extended in Python get to the first MOOSE base class
                while className not in moose_classes and className != 'ObjId':
                    _base_class = self.__base__
                    if _base_class == object:
                        raise TypeError('Class %s does not inherit any MOOSE class.' % (self.__class__.__name__))                    
                    className = _base_class.__name__
                    if className.endswith('Array'):
                        className = className[:-len('Array')]
            Id.__init__(path=path, dims=dims, type=className)

    def getFieldNames(self, ftype='valueFinfo'):
        """Return a list of fields available in this object.

        Parameters:

        str ftype -- (default 'valueFinfo') category of field, valid values are:
        valueFinfo, lookupFinfo, srcFinfo, destFinfo or sharedFinfo.

        If empty string or not specified, returns names of fields from
        all categories.
        """        
        return self[0].getFieldNames(ftype)

    def getFieldType(self, field, ftype=''):
        """Return the data type of the field as a string."""
        return self[0].getFieldType(field, ftype)

    path = property(lambda self: self.getPath())
    fieldNames = property(lambda self: self.getFieldNames(ftype='valueFinfo'))
    name = property(lambda self: self[0].name)
    shape = property(lambda self: self.getShape())
    
################################################################
# Special function to generate objects of the right class from
# a given path.
################################################################

def element(path):
    """Return a reference to an existing object as an instance of the
    right class. If path does not exist, raises NameError.

    Id or ObjId can be provided in stead of path"""
    if isinstance(path, Id):
        oid = path[0]
        path = path.getPath()
    elif isinstance(path, ObjId):
        oid = path
    elif isinstance(path, str):
        if not _moose.exists(path):
            raise NameError('Object %s not defined' % (path))
        oid = _moose.ObjId(path)
    else:
        raise TypeError('expected argument: Id/ObjId/str')
    classObj = eval(oid.class_)
    return classObj(oid)

def arrayelement(path, className='Neutral'):
    """Return a reference to an existing object as an instance of the
    right class. If path does not exist, className is used for
    creating an instance of that class with the given path"""
    if not exists(path):
        raise NameError('Object %s not defined' % (path))
    oid = ObjId(path)
    className = oid.getField('class')
    return eval('%sArray("%s")' % (className, path))
    

################################################################
# Wrappers for global functions
################################################################ 


    
def pwe():
    """Print present working element. Convenience function for GENESIS
    users."""
    print _moose.getCwe().getPath()
    
def le(element=None):
    """List elements. """
    if element is None:
        element = getCwe()[0]
    elif isinstance(element, str):
        element = Neutral(element)
    print 'Elements under', element.path
    for ch in element.children:
        print ch

ce = setCwe

def syncDataHandler(target):
    """Synchronize data handlers for target.

    Parameter:
    target -- target element or path or Id.
    """
    raise NotImplementedError('The implementation is not working for IntFire - goes to invalid objects. First fix that issue with SynBase or something in that line.')
    if isinstance(target, str):
        if not _moose.exists(target):
            raise ValueError('%s: element does not exist.' % (target))
        target = Id(target)
        _moose.syncDataHandler(target)

def showfield(elem, field='*', showtype=False):
    """Show the fields of the element, their data types and values in
    human readable format. Convenience function for GENESIS users.

    Parameters:

    element -- Element or path of an existing element or ObjId of an element.

    field -- Field to be displayed. If '*', all fields are displayed.

    showtype -- If True show the data type of each field.

    """
    target = element(elem)
    if field == '*':        
        value_field_dict = getFieldDict(target.class_, 'valueFinfo')
        max_type_len = max([len(dtype) for dtype in value_field_dict.values()])
        max_field_len = max([len(dtype) for dtype in value_field_dict.keys()])
        print 
        print '[', target.path, ']'
        for key, dtype in value_field_dict.items():
            if dtype == 'bad' or key == 'this' or key == 'dummy' or key == 'me' or dtype.startswith('vector') or 'ObjId' in dtype:
                continue
            value = target.getField(key)
            if showtype:
                print dtype.ljust(max_type_len + 4),
            print key.ljust(max_field_len + 4), '=', value
    else:
        try:
            print field, '=', target.getField(field)
        except AttributeError:
            pass # Genesis silently ignores non existent fields

def showfields(element, showtype=False):
    """Convenience function. Should be deprecated if nobody uses it."""
    warnings.warn('Deprecated. Use showfield(element, field="*", showtype=True) instead.', DeprecationWarning)
    showfield(element, field='*', showtype=showtype)
    
def wildcardFind(cond):
    """Search for objects that match condition cond."""
    return [element(id_) for id_ in _moose.wildcardFind(cond)]

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
                print Cinfo(class_path).docs
        else:
            print 'No such class:', tokens[0]
            return
    class_id = Id('/classes/%s' % (tokens[0]))
    num_finfo = getField(class_id[0], 'num_valueFinfo', 'unsigned')
    finfo = Id('/classes/%s/valueFinfo' % (tokens[0]))
    print '\n* Value Field *'
    for ii in range(num_finfo):
        oid = ObjId(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print oid.name, ':', oid.docs
                return
        else:
            print oid.name, ':', oid.docs
            print
    num_finfo = getField(class_id[0], 'num_srcFinfo', 'unsigned')
    finfo = Id('/classes/%s/srcFinfo' % (tokens[0]))
    print '\n* Source Field *'
    for ii in range(num_finfo):
        oid = ObjId(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print oid.name, ':', oid.docs
                return
        else:
            print oid.name, ':', oid.docs
            print
    num_finfo = getField(class_id[0], 'num_destFinfo', 'unsigned')
    finfo = Id('/classes/%s/destFinfo' % (tokens[0]))
    print '\n* Destination Field *'
    for ii in range(num_finfo):
        oid = ObjId(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print oid.name, ':', oid.docs
                return
        else:
            print oid.name, ':', oid.docs
            print
    num_finfo = getField(class_id[0], 'num_lookupFinfo', 'unsigned')    
    finfo = Id('/classes/%s/lookupFinfo' % (tokens[0]))
    print '\n* Lookup Field *'
    for ii in range(num_finfo):
        oid = ObjId(finfo, 0, ii, 0)
        if print_field:
            if oid.name == tokens[1]:
                print oid.name, ':', oid.docs
                return
        else:
            print oid.name, ':', oid.docs
            print
    

# 
# moose.py ends here
