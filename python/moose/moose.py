# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Thu Mar  8 16:10:57 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 1370
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

"""
MOOSE = Multiscale Object Oriented Simulation Environment.

Classes:

Id:

this is the unique identifier of a MOOSE object. Note that you
can create multiple references to the same MOOSE object in Python, but
as long as they have the same Id, they all point to the same entity in
MOOSE.

Methods:

getValue() -- unsigned integer representation of id

getPath() -- string representing the path corresponding this id

getShape -- tuple containing the dimensions of this id

Id also implements part of the sequence protocol:

len(id) -- the first dimension of id.

id[n] -- the n-th ObjId in id.

id[n1:n2] -- a tuple containing n1 to n2-th (exclusive) ObjId in id.

objid in id -- True if objid is contained in id.



ObjId:

Unique identifier of an element in a MOOSE object. It has three components:

Id id - the Id containing this element

unsigned integer dataIndex - index of this element in the container

unsigned integer fieldIndex - if this is a tertiary object, i.e. acts
as a field in another element (like synapse[0] in IntFire[1]), then
the index of this field in the containing element.

Methods:

getId -- Id object containing this ObjId.

getDataIndex() -- unsigned integer representing the index of this
element in containing MOOSE object.

getFieldIndex() -- unsigned integer representing the index of this
element as a field in the containing Element.

getFieldType(field) -- human readable datatype information of field

getField(field) -- get value of field

setField(field, value) -- assign value to field

getFieldNames(fieldType) -- tuple containing names of all the fields
of type fieldType. fieldType can be valueFinfo, lookupFinfo, srcFinfo,
destFinfo and sharedFinfo. If nothing is passed, a union of all of the
above is used and all the fields are returned.

connect(srcField, destObj, destField, msgType) -- connect srcField of
this element to destField of destObj.

getMsgSrc(fieldName) -- return a tuple containing the ObjIds of all
the elements from which a message is entering the field specified by
fieldName.

getMsgDesr(fieldName) -- return a tuple containing list of ObjIds of
elements that recieve messages from the fieldName of this element.


NeutralArray:

The base class. Each NeutralArray object has an unique Id (field _id) and
that is the only data directly visible under Python. All operation are
done on the objects by calling functions on the Id.

A NeutralArray object is actually an array. The individual elements in a
NeutralArray are of class Neutral. To access these individual
elements, you can index the NeutralArray object.

A NeutralArray object can be constructed in many ways. The most basic one
being:

neutral = moose.NeutralArray('my_neutral_object', [3])

This will create a NeutralArray object with name 'my_neutral_object'
containing 3 elements. The object will be created as a child of the
current working entity. Any class derived from NeutralArray can also be
created using the same constructor. Actually it takes keyword
parameters to do that:

intfire = moose.NeutralArray(path='/my_neutral_object', dims=[3], type='IntFire')

will create an IntFire object of size 3 as a child of the root entity.

If the above code is already executed,

duplicate = moose.NeutralArray(intfire)

will create a duplicate reference to the existing intfire object. They
will share the same Id and any changes made via the MOOSE API to one
will be effective on the other.

Neutral -- The base class for all elements in object of class
NeutralArray or derivatives of NeutralArray. A Neutral will always point
to an index in an existing entity. The underlying data is ObjId (field
oid_) - a triplet of id, dataIndex and fieldIndex. Here id is the Id
of the NeutralArray object containing this element. dataIndex is the index
of this element in the container. FieldIndex is a tertiary index and
used only when this element acts as a field of another
element. Otherwise fieldIndex is 0.

Indexing a NeutralArray object returns a Neutral.

i_f = intfire[0] will return a reference to the first element in the
IntFire object we created earlier. All field-wise operations are done
on Neutrals.

A neutral object (and its derivatives) can also be created in the
older way by specifying a path to the constructor. This path may
contain an index. If there is a pre-existing NeutralArray object with
the given path, then the index-th item of that array is returned. If
the target object does not exist, but all the objects above it exist,
then a new Array object is created and its first element is
returned. If an index > 0 is specified in this case, that results in
an IndexOutOfBounds exception. If any of the objects higher in the
hierarchy do not exist (thus the path up to the parent is invalid), a
NameError is raised.

a = Neutral('a') # creates /a
b = IntFire(a/b') # Creates /a/b
c = IntFire(c/b') # Raises NameError.
d = NeutralArray('c', 10)
e = Neutral('c[9]') # Last element in d

Fields:

childList - a list containing the children of this object.

className - class of the underlying MOOSE object. The corresponding
field in MOOSE is 'class', but in Python that is a keyword, so we use
className instead. This is same as Neutral.getField('class')


dataIndex - data index of this object. This should not be needed for
normal use.

dimensions - a tuple representation dimensions of the object. If it is
empty, this is a singleton object.

fieldIndex - fieldIndex for this object. Should not be needed for
ordinary use.

fieldNames - list fields available in the underlying MOOSE object.



Methods:

children() - return the list of Ids of the children

connect(srcField, destObj, destField) - a short hand and backward
compatibility function for moose.connect(). It creates a message
connecting the srcField on the calling object to destField on the dest
object.

getField(fieldName) - return the value of the specified field.

getFieldNames() - return a list of the available field names on this object

getFieldType(fieldName) - return the data type of the specified field.



More generally, Neutral and all its derivatives will have a bunch of methods that are
for calling functions via destFinfos. help() for these functions
should show something like:

<lambda> lambda self, arg_0_{type}, arg_1_{type} unbound moose.{ClassName} method

These are dynamically defined methods, and calling them with the right
parameters will cause the corresponding moose function to be
called. Note that all parameters are converted to strings, so you may
loose some precision here.

[Comment - subha: This explanation is no less convoluted than the
implementation itself. Hopefully I'll have the documentation
dynamically dragged out of Finfo documentation in future.]

module functions:

element(path) - returns a reference to an existing object converted to
the right class. Raises NameError if path does not exist.

arrayelement(path) - returns a reference to an existing object
converted to the corresponding Array class. Raises NameError if path
does not exist.

copy(src=<src>, dest=<dest>, name=<name_of_the_copy>, n=<num_copies>,
copyMsg=<whether_to_copy_messages) -- make a copy of source object as
a child of the destination object.


move(src, dest) -- move src object under dest object.

useClock(tick, path, field) -- schedule <field> of every object that
matches <path> on clock no. <tick>.

setClock(tick, dt) -- set dt of clock no <tick>.

start(runtime) -- start simulation of <runtime> time.

reinit() -- reinitialize simulation.

stop() -- stop simulation

isRunning() -- true if simulation is in progress, false otherwise.

exists(path) -- true if there is a pre-existing object with the specified path.

loadModel(filepath, modelpath) -- load file in <filepath> into node
<modelpath> of the moose model-tree.

setCwe(obj) -- set the current working element to <obj> - which can be
either a string representing the path of the object in the moose
model-tree, or an Id.
cwe(obj) -- an alias for setCwe.

getCwe() -- returns Id of the current working element.
pwe() -- an alias for getCwe.

showfields(obj) -- print the fields in object in human readable format

le(obj) -- list element under object, if no parameter specified, list
elements under current working element

"""

from collections import defaultdict
import _moose
from _moose import __version__, VERSION, SVN_REVISION, useClock, setClock, start, reinit, stop, isRunning, loadModel, getFieldDict, Id, ObjId, exists, seed
from _moose import wildcardFind as _wildcardFind # We override the original

# Dict of available MOOSE class names. This is faster for look-up
_moose_classes = dict([(child.getPath().rpartition('/')[-1], True) for child in Id('/classes')[0].getField('children')])

class _MooseDescriptor(object):
    """Descriptor to give access to MOOSE class' ValueFinfo attributes"""
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.oid_.getField(self.name)

    def __set__(self, obj, value):
        obj.oid_.setField(self.name, value)
        
    def __delete__(self, obj):
        raise AttributeError('ValueFinfos cannot be deleted.')

class _LFDescriptor(object):
    def __init__(self, name):
        self.name = name
    def __get__(self, obj, objtype=None):
        return _LookupField(obj.oid_, self.name)
    
class _LookupField(object):
    def __init__(self, oid, name):
        self.name = name
        self.oid = oid
    def __getitem__(self, key):
        return self.oid.getLookupField(self.name, key)
    def __setitem__(self, key, value):
        self.oid.setLookupField(self.name, key, value)
    
class _MooseMeta(type):
    """Simple metaclass to create class definitions with attributes
    created from that in the MOOSE underlying classes."""
    def __init__(cls, name, bases, classdict):        
        super(_MooseMeta, cls).__init__(name, bases, classdict)
        try:
            field_dict = getFieldDict(name, 'valueFinfo')
            for field in field_dict.keys():
                setattr(cls, field, _MooseDescriptor(field))
        except NameError:
            pass
        try:
            field_dict = getFieldDict(name, 'lookupFinfo')
            for field in field_dict.keys():
                setattr(cls, field, _LFDescriptor(field))
        except NameError:
            pass
        

class NeutralArray(object):
    """
    The base class. Each NeutralArray object has an unique Id (field
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

        neutral = moose.NeutralArray('my_neutral_object', [3])

        This will create a NeutralArray object with name
        'my_neutral_object' containing 3 elements. The object will be
        created as a child of the current working entity. Any class
        derived from NeutralArray can also be created using the same
        constructor. Actually it takes keyword parameters to do that:

        intfire = moose.NeutralArray(path='/my_neutral_object', dims=[3], type='IntFire')

        will create an IntFire object of size 3 as a child of the root entity.

        If the above code is already executed,

        duplicate = moose.NeutralArray(intfire)

        will create a duplicate reference to the existing intfire
        object. They will share the same Id and any changes made via
        the MOOSE API to one will be effective on the other.
        
        """
        path = None
        dims = None
        self.id_ = None
        try:
            className = kwargs['type']
            self.className = className
        except KeyError:
            # This code is messy and depends on the class name. I
            # could not find a way to pass the element data type to
            # the class definition dynamically
            if not hasattr(self, 'className'):
                self.className = 'Neutral'
        try:
            dims = kwargs['dims']
        except KeyError:
            pass
        try:
            path = kwargs['path']
        except KeyError:
            pass
        if len(args) > 0:
            if isinstance(args[0], str):
                path = args[0]
            elif isinstance(args[0], Id):
                self.id_ = args[0]
            elif isinstance(args[0], int):
                self.id_ = Id(args[0])
        if len(args) > 1:
            dims = args[1]
        if len(args) > 2:
            self.className = args[2]
        # No existing Array element ot Id specified, create new
        # ArrayElement
        if self.id_ is None:
            if path is None:
                raise TypeError('A string path or an existing Id or an int value for existing Id must be the first argument to __init__')
            if exists(path):
                self.id_ = _moose.Id(path=path)
                # Check if specified dimensions match the existing
                # object's dimensions
                if dims is not None:
                    shape = self.id_.getShape()
                    if isinstance(dims, int):
                        if shape[0] != dims:
                            raise ValueError('Specified dimensions do not match that of existing array object')
                    else:
                        if len(shape) != len(dims):
                            raise ValueError('Specified dimensions do not match that of existing array object')
                        for ii in range(len(shape)):
                            if shape[ii] != dims[ii]:
                                raise ValueError('Specified dimensions do not match that of existing array object')
                else:
                    dims = (1)
            # Create a new ArrayElement
            _base_class = self.__class__
            _class_name = self.__class__.__name__
            if _class_name.endswith('Array'):
                _class_name = _class_name[:-len('Array')]
            # For classes extended in Python get to the first MOOSE base class
            while _class_name not in _moose_classes:
                _base_class = self.__base__
                if _base_class == object:
                    raise TypeError('Class %s does not inherit any MOOSE class.' % (self.__class__.__name__))                    
                _class_name = _base_class.__name__
                if _class_name.endswith('Array'):
                    _class_name = _class_name[:-len('Array')]
            self.id_ = _moose.Id(path=path, dims=dims, type=_class_name)
        # Check for conversion from instance of incompatible MOOSE
        # class.
        orig_classname = self.id_[0].getField('class') + 'Array'
        if self.__class__.__name__ != orig_classname:
            orig_class = eval(orig_classname)
            self_class = self.__class__
            while self_class != object and self_class not in orig_class.mro():
                self_class = self_class.__base__
            if self_class == object:
                self.id_ = None
                raise TypeError('Cannot convert %s to %s' % (orig_class, self.__class__))

    def getFieldNames(self, ftype=''):
        """Return a list of fields available in this object.

        Parameters:

        str ftype -- (default '') category of field, valid values are:
        valueFinfo, lookupFinfo, srcFinfo, destFinfo or sharedFinfo.

        If empty string or not specified, returns names of fields from
        all categories.
        """
        
        return self.id_[0].getFieldNames(ftype)

    def getFieldType(self, field, ftype=''):
        """Return the data type of the field as a string."""
        return self.id_[0].getFieldType(field, ftype)

    def __getitem__(self, index):
        objid = self.id_[index]
        retclass = eval(self.id_[0].getField('class'))
        ret = retclass(objid)
        return ret

    def __len__(self):
        return len(self.id_)

    def __contains__(self, other):
        if isinstance(other, Neutral):
            return other.oid_ in self.id_
        elif isinstance(other, ObjId):
            return other in self.id_
        else:
            return False

    def __repr__(self):
        return self.id_.getPath()

    path = property(lambda self: self.id_.getPath())
    fieldNames = property(lambda self: self.id_[0].getFieldNames('valueFinfo'))
    name = property(lambda self: self.id_[0].getField('name'))
    shape = property(lambda self: self.id_.getShape())
    

        
class Neutral(object):
    """Corresponds to a single entry in a NeutralArray. Base class for
    all other MOOSE classes for single entries in Array elements.

    A Neutral object wraps an ObjId (field oid_) - a triplet of id,
    dataIndex and fieldIndex. Here id is the Id of the NeutralArray object
    containing this element. dataIndex is the index of this element in the
    container. FieldIndex is a tertiary index and used only when this
    element acts as a field of another element. Otherwise fieldIndex is 0.

    Indexing a NeutralArray object returns a Neutral.

    A neutral object (and its derivatives) can also be created in the
    older way by specifying a path to the constructor. This path may
    contain an index. If there is a pre-existing NeutralArray object
    with the given path, then the index-th item of that array is
    returned. If the target object does not exist, but all the objects
    above it exist, then a new Array object is created and its first
    element is returned. If an index > 0 is specified in this case,
    that results in an IndexOutOfBounds exception. If any of the
    objects higher in the hierarchy do not exist (thus the path up to
    the parent is invalid), a NameError is raised.

    a = Neutral('a') # creates /a
    b = IntFire(a/b') # Creates /a/b
    c = IntFire(c/b') # Raises NameError.
    d = NeutralArray('c', 10)
    e = Neutral('c[9]') # Last element in d
    """
    # _MooseMeta creates field access via getters and setters.
    __metaclass__ = _MooseMeta
    def __init__(self, *args, **kwargs):
        """Initialize a Neutral object.

        Arguments:

        arg1 : A path or an existing ObjId or an Id or a NeutralArray
        or another Neutral object. 
        
        path -- a string specifying the path for the Neutral
        object. If there is already a Neutral object with the given
        path, a reference to that object is created. Otherwise, a new
        NeutralArray is created with the given path. In the latter
        case, the path must be valid up to the last '/'. For example,
        if you specify '/a/b/c', then for correct operation, there
        must be an element named 'a' at the top level and a child
        element of 'a' named 'b'. This works like 'mkdir' or 'md'
        command in some operating systems.

        ObjId -- if the first argument is an ObjId, then the Neutral
        object refers to the existing object with the specified ObjId.

        Id -- if the first argument is an Id then the Neutral object
        will refer to some entry in the ArrayElement with this Id. The
        exact index pointed to will be determined by the second
        argument, or the first entry if no second argument is
        specified.

        NeutralArray -- Same as Id (as if the Id of the NeutralArray
        was passed).

        Neutral -- create a new reference to the existing Neutral
        object.

        arg2 : if there is a second argument, then this is taken as
        the dataindex into an existing array element.

        arg3: if there is a third argument, this is taken as the
        fieldindex into an existing array field.
        """
        id_ = None
        self.oid_ = None
        dindex = None
        findex = None
        numFieldBits = None
        if len(args) >= 1:
            if isinstance(args[0], ObjId):
                self.oid_ = args[0]
            elif isinstance(args[0], Id):
                id_ = args[0].getValue()
            elif isinstance(args[0], Neutral):
                self.oid_ = args[0].oid_
            elif isinstance(args[0], NeutralArray):
                id_ = args[0].id_
            elif isinstance(args[0], str):
                try:
                    self.oid_ = _moose.ObjId(args[0])
                except ValueError:
                    # A non-existing path has been provided. Try to
                    # construct a singleton array element and create
                    # reference to the first element.
                    self_class = self.__class__                    
                    while (self_class.__name__ not in _moose_classes) and (self_class != object): # Handle class extension in Python. 
                        self_class = self_class.__base__
                    if self_class == object:
                        raise TypeError('Class %s does not inherit any MOOSE class.' % (self.__class__.__name__))
                    id_ = Id(path=args[0], dims=(1,), type=self_class.__name__)
            else:
                raise TypeError('First non-keyword argument must be a number or an existing Id/ObjId/Neutral/NeutralArray object or a path.')
        if len(args) >= 2:
            dindex = args[1]
        if len(args) >= 3:
            findex = args[2]
        if len(args) >= 4:
            numFieldBits = args[3]
        if (kwargs):
            try:
                id_ = kwargs['id']
            except KeyError:
                pass
            try:
                dindex = kwargs['dataIndex']
            except KeyError:
                pass
            try:
                findex = kwargs['fieldIndex']
            except KeyError:
                pass
            try:
                numFieldBits = kwargs['numFieldBits']
            except KeyError:
                pass
        if self.oid_ is None:
            if id_ is not None:
                if dindex is None:
                    self.oid_ = _moose.ObjId(id_)
                elif findex is None:
                    self.oid_ = _moose.ObjId(id_, dindex)
                elif numFieldBits is None:
                    self.oid_ = _moose.ObjId(id_, dindex, findex)
                else:
                    self.oid_ = _moose.ObjId(id_, dindex, findex, numFieldBits)
        # Check for conversion from instance of incompatible MOOSE
        # class.        
        orig_classname = self.oid_.getField('class')
        if self.__class__.__name__ != orig_classname:
            orig_class = eval(orig_classname)
            self_class = self.__class__
            while self_class != object and self_class not in orig_class.mro():
                self_class = self_class.__base__
            if self_class == object:
                self.oid_ = None
                raise TypeError('Cannot convert %s to %s' % (orig_class, self.__class__))

    def getField(self, field):
        """Return the field value"""
        return self.oid_.getField(field)

    def getFieldType(self, field, ftype=''):
        """Return the type of the specified field in human readable format"""
        return self.oid_.getFieldType(field, ftype)

    def getFieldNames(self, ftype=''):
        """Return a list of the fields of specified fieldType."""
        return self.oid_.getFieldNames(ftype)

    def getNeighbors(self, fieldName):
        if fieldName in getFieldDict(self.className).keys():
            return [eval('%s("%s")' % (id_[0].getField('class'), id_.getPath())) for id_ in self.oid_.getNeighbors(fieldName)]
        raise ValueError('%s: no such field on %s' % (fieldName, self.path))
        
    def connect(self, srcField, dest, destField, msgType='Single'):
        return self.oid_.connect(srcField, dest.oid_, destField, msgType)

    
    def inMessages(self):
        msg_list = []
        for msg in self.msgIn:
            e1 = msg.getField('e1')
            e2 = msg.getField('e2')
            for (f1, f2) in zip(msg.getField('srcFieldsOnE2'), msg.getField('destFieldsOnE1')):
                msg_str = '[%s].%s <- [%s].%s' % (e1.getPath(), f1, e2.getPath(), f2)
                msg_list.append(msg_str)
        return msg_list

    def outMessages(self):
        msg_list = []
        for msg in self.msgOut:
            e1 = msg.getField('e1')
            e2 = msg.getField('e2')
            for (f1, f2) in zip(msg.getField('srcFieldsOnE1'), msg.getField('destFieldsOnE2')):
                msg_str = '[%s].%s -> [%s].%s' % (e1.getPath(), f1, e2.getPath(), f2)
                msg_list.append(msg_str)
        return msg_list
    
    
    className = property(lambda self: self.oid_.getField('class'))
    fieldNames = property(lambda self: self.oid_.getFieldNames())
    name = property(lambda self: self.oid_.getField('name'))
    path = property(lambda self: '%s[%d]' % (self.oid_.getField('path'), self.oid_.getDataIndex()))
    id_ = property(lambda self: self.oid_.getId())
    fieldIndex = property(lambda self: self.oid_.getFieldIndex())
    dataIndex = property(lambda self: self.oid_.getDataIndex())
    # We have a lookup field called neighbors already, this should override that
    @property
    def neighborDict(self):
        """defaultdict whose keys are field names and values are list
        of objects that are connected to that field"""
        neighbors = defaultdict(list)
        for finfotype in ['srcFinfo', 'destFinfo', 'sharedFinfo']:
            for field in self.oid_.getFieldNames(finfotype):
                tmp = self.oid_.getNeighbors(field)
                neighbors[field] += [eval('%s("%s")' % (nid[0].getField('class'), nid.getPath())) for nid in tmp]
        return neighbors
        
    
    childList = property(lambda self: [eval('%s("%s")' % (ch[0].getField('class'), ch.getPath())) for ch in self.oid_.getField('children')])

################################################################
# Special function to generate objects of the right class from
# a given path.
################################################################

def element(path):
    """Return a reference to an existing object as an instance of the
    right class. If path does not exist, className is used for
    creating an instance of that class with the given path"""
    if not _moose.exists(path):
        raise NameError('Object %s not defined' % (path))
    oid = _moose.ObjId(path)
    className = oid.getField('class')
    return eval('%s("%s")' % (className, path))

def arrayelement(path, className='Neutral'):
    """Return a reference to an existing object as an instance of the
    right class. If path does not exist, className is used for
    creating an instance of that class with the given path"""
    if not _moose.exists(path):
        raise NameError('Object %s not defined' % (path))
    oid = _moose.ObjId(path)
    className = oid.getField('class')
    return eval('%sArray("%s")' % (className, path))
    

################################################################
# Wrappers for global functions
################################################################ 

def copy(src, dest, name, n=1, toGlobal=False, copyExtMsg=False):
    if isinstance(src, NeutralArray):
        src = src.id_
    if isinstance(dest, NeutralArray):
        dest = dest.id_
    new_id = _moose.copy(src, dest, name, n=n, toGlobal=toGlobal, copyExtMsg=copyExtMsg)
    return new_id

def move(src, dest):
    if isinstance(src, NeutralArray):
        src = src.id_
    if isinstance(dest, NeutralArray):
        dest = dest.id_
    _moose.move(src, dest)

def delete(target):
    """Explicitly delete a MOOSE object. This will invalidate all
    existing references. They will all point to the default root
    object."""
    if isinstance(target, NeutralArray):
        target = target.id_
    if not isinstance(target, Id):
        raise TypeError('Only Id or Array objects can be deleted: received %s' % (target.__class__.__name__))
    _moose.delete(target)
    
def setCwe(element):
    if isinstance(element, NeutralArray):
        _moose.setCwe(element.id_)
    elif isinstance(element, Neutral):
        _moose.setCwe(element.oid_)
    else:
        _moose.setCwe(element)

def getCwe():
    _id = _moose.getCwe()
    obj = NeutralArray(_id)
    return obj

def connect(src, srcMsg, dest, destMsg, msgType='Single'):
    """Connect src object's source field specified by srcMsg to
    destMsg field of target object."""
    if isinstance(src, Neutral):
        src = src.oid_
    if isinstance(dest, Neutral):
        dest = dest.oid_
    return src.connect(srcMsg, dest, destMsg, msgType)

def le(element=None):
    if element is None:
        element = getCwe()[0]
    elif isinstance(element, str):
        element = Neutral(element)
    print 'Elements under', element.path
    for ch in element.children:
        print ch
    
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

def showfields(element):
    """Show the fields of the element, their data types and values in
    human readable format.

    Parameters:

    element -- Element or path of an existing element or ObjId of an element.

    showtype -- show the data type of each field or not.

    """
    if isinstance(element, str):
        if not moose.exists(element):
            raise ValueError('%s -- no such moose object exists.' % (element))
        element = Neutral(element)
    if not isinstance(element, Neutral):
        if not isinstance(element, ObjId):
            raise TypeError('Expected argument of type ObjId or Neutral or a path to an existing object. Found %s' % (type(element)))
        element = Neutral(element)
    value_field_dict = getFieldDict(element.className, 'valueFinfo')
    print '------------------------------------'
    print 'Value Fields of ', element.path
    print '------------------------------------'
    for key, dtype in value_field_dict.items():
        if dtype == 'bad':
            continue
        value = element.oid_.getField(key)        
        print dtype, key, ' = ', value
        
def wildcardFind(path):
    return [eval('%s("%s")' % (id_[0].getField('class'), id_.getPath())) for id_ in _wildcardFind(path)]
    
#######################################################
# This is to generate class definitions automatically
#######################################################
def define_class(classId):
    """Define a class based on Cinfo element with Id=classId."""
    class_name = classId[0].getField('name')
    if class_name in globals().keys():
        return
    base = classId[0].getField('baseClass')
    if base != 'none':
        try:
            base_class = globals()[base]
        except KeyError:
            define_class(Id('/classes/'+base))
            base_class = globals()[base]
    else:
        base_class = object
    class_obj = type(class_name, (base_class,), {})
    # Create the attributes for ValueFinfos. These will appear to be
    # ordinary Python attributes, but actually accessed via calls to
    # getField and setField.
    field_dict = getFieldDict(class_name, 'valueFinfo')
    for field in field_dict.keys():
        setattr(class_obj, field, _MooseDescriptor(field))
    # Go through the destFinfos and make them look like methods
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
    for finfoName, finfoArgs in getFieldDict(class_name, 'destFinfo').items():
        cont_flag = False
        # 'get_...' and 'set_...' are internal destFinfos - not to be accessed directly
        if finfoName.startswith('get_') or finfoName.startswith('set_'):
            continue
        destfnArgs = finfoArgs.split(',')
        fndef = 'lambda self'
        fnargs = "'%s'" % (finfoName)
        index = 0
        # Generate the string representation of the lambda expression
        # for calling the destination function.
        for arg in destfnArgs:
            if arg not in known_types:
                # Unhandled argument type, skip this destFinfo
                cont_flag = True
                break
            # Extend the string for the lambda expression with formal
            # arguments
            if arg != 'void':
                fndef += ', arg_%d_%s' % (index, arg.replace(' ', '_').replace('<', '_').replace('>', ''))
                fnargs += ', arg_%d_%s' % (index, arg.replace(' ', '_').replace('<', '_').replace('>', ''))
            index += 1
        if cont_flag:
            continue
        function_string = '%s: self.oid_.setDestField(%s)' % (fndef, fnargs)
        # print 'Setting %s.%s - Function string: %s' % (class_obj.__name__, finfoName, function_string)
        setattr(class_obj, finfoName, eval(function_string))
    # Add this class to globals dict
    globals()[class_name] = class_obj
    array_class_name = class_name + 'Array'
    if base != 'none':
        base_class = globals()[base + 'Array']
    else:
        base_class = object
    array_class_obj = type(array_class_name, (base_class,), {}) # We need to tell the 
    globals()[array_class_name] = array_class_obj
        
            

classes_Id = Id('/classes')
class_obj_list = classes_Id[0].getField('children')    
for child in class_obj_list:
    define_class(child)


# 
# moose.py ends here
