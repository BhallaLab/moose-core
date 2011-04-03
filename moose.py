# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Sun Apr  3 14:05:20 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 407
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

getFieldType(field) -- data type of field as a human readable string.

getField(field) -- value of field

setField(field, value) -- assign value to field

getValue() -- unsigned integer representation of id

getPath() -- string representing the path corresponding this id

getShape -- tuple containing the dimensions of this id

Id implements part of the sequence protocol:

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

getFieldType(field) -- human readable datatype information of field

getField(field) -- get value of field

setField(field, value) -- assign value to field

getId -- Id object containing this ObjId.

getFieldNames(fieldType) -- tuple containing names of all the fields
of type fieldType. fieldType can be valueFinfo, lookupFinfo, srcFinfo,
destFinfo and sharedFinfo. If nothing is passed, a union of all of the
above is used and all the fields are returned.

connect(srcField, destObj, destField, msgType) -- connect srcField of
this element to destField of destObj.

getDataIndex() -- unsigned integer representing the index of this
element in containing MOOSE object.

getFieldIndex() -- unsigned integer representing the index of this
element as a field in the containing Element.


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
_oid) - a triplet of id, dataIndex and fieldIndex. Here id is the Id
of the NeutralArray object containing this element. dataIndex is the index
of this element in the container. FieldIndex is a tertiary index and
used only when this element acts as a field of another
element. Otherwise fieldIndex is 0.

Indexing a NeutralArray object returns a Neutral.

i_f = intfire[0] will return a reference to the first element in the
IntFire object we created earlier. All field-wise operations are done
on Neutrals.

Methods:


module functions:

copy(src=<src>, dest=<dest>, name=<name_of_the_copy>, n=<num_copies>, copyMsg=<whether_to_copy_messages) -- make a copy of source object as a child of the destination object.


move(src, dest) -- move src object under dest object.

useClock(tick, path, field) -- schedule <field> of every object that
matches <path> on clock no. <tick>.

setClock(tick, dt) -- set dt of clock no <tick>.

start(runtime) -- start simulation of <runtime> time.

reinit() -- reinitialize simulation.

stop() -- stop simulation

isRunning() -- true if simulation is in progress, false otherwise.

loadModel(filepath, modelpath) -- load file in <filepath> into node
<modelpath> of the moose model-tree.

setCwe(obj) -- set the current working element to <obj> - which can be either a string representing the path of the object in the moose model-tree, or an Id.

getCwe() -- returns Id of the current working element.


"""

__version__ = "$Revision$"
# $Source$

import _moose
from _moose import useClock, setClock, start, reinit, stop, isRunning, loadModel, getFieldDict, Id, ObjId


class _MooseDescriptor(object):
    """Descriptor to give access to MOOSE class' ValueFinfo attributes"""
    def __init__(self, name):
        print 'Creating attribute', name
        self.name = name

    def __get__(self, obj, objtype=None):
        print 'Getting field', self.name
        return obj._oid.getField(self.name)

    def __set__(self, obj, value):
        print 'Setting field', self.name
        obj._oid.setField(self.name, value)

    def __delete__(self, obj):
        raise AttributeError('ValueFinfos cannot be deleted.')


    
    
class _MooseMeta(type):
    def __init__(cls, name, bases, classdict):        
        print "Creating class %s using NeutralMeta" % (name)
        super(_MooseMeta, cls).__init__(name, bases, classdict)        
        field_dict = getFieldDict('/classes/%s' % (name), 'valueFinfo')
        for field in field_dict.keys():
            setattr(cls, field, _MooseDescriptor(field))
        



class NeutralArray(object):
    def __init__(self, *args, **kwargs):
        path ='/'
        dims = [1]
        className = 'Neutral'
        self._id = None
        try:
            className = kwargs['type']
        except KeyError:
            kwargs['type'] = 'Neutral'
        try:
            dims = kwargs['dims']
        except KeyError:
            dims = [1]
        try:
            path = kwargs['path']
        except KeyError:
            pass
        if len(args) > 0:
            if isinstance(args[0], str):
                path = args[0]
            elif isinstance(args[0], Id):
                self._id = args[0]
            elif isinstance(args[0], int):
                self._id = Id(args[0])
        if len(args) > 1:
            dims = args[1]
        if len(args) > 2:
            className = args[2]
        if self._id is None:
            print path, dims, className
            self._id = _moose.Id(path=path, dims=dims, type=className)

    def getFieldNames(self, ftype=''):
        return self._id[0].getFieldNames(ftype)

    def getFieldType(self, field):
        return self._id[0].getFieldType(field)

    def __getitem__(self, index):
        objid = self._id[index]
        retclass = eval(self._id[0].getField('class'))
        ret = retclass(objid)
        return ret

    def __len__(self):
        return len(self._id)    
    
    path = property(lambda self: self._id.getPath())
    id = property(lambda self: self._id)
    fieldNames = property(lambda self: self._id[0].getFieldNames('valueFinfo'))
    className = property(lambda self: self._id[0].getField('class'))
    name = property(lambda self: self._id[0].getField('name'))
    shape = property(lambda self: self._id.getShape())
    

class IntFireArray(NeutralArray):
    def __init__(self, *args, **kwargs):
        try:
            className = kwargs['type']
        except KeyError:
            kwargs['type'] = 'IntFire'
        NeutralArray.__init__(self, *args, **kwargs)

        
class Neutral(object):
    __metaclass__ = _MooseMeta
    def __init__(self, *args, **kwargs):
        id_ = 0
        dindex = 0
        findex = 0
        if len(args) >= 1:
            if isinstance(args[0], ObjId):
                (id_, dindex, findex) = (args[0].getId().getValue(), args[0].getDataIndex(), args[0].getFieldIndex())
            elif isinstance(args[0], Id):
                id_ = args[0].getValue()
        if len(args) >= 2:
            dindex = args[1]
        if len(args) >= 3:
            findex = args[2]
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
        print id_, dindex, findex
        self._oid = _moose.ObjId(id_, dindex, findex)

    def getField(self, field):
        return self._oid.getField(field)
    
    className = property(lambda self: self._oid.getField('class'))
    fieldNames = property(lambda self: self._oid.getFieldNames())
    name = property(lambda self: self._oid.getField('name'))
    path = property(lambda self: self._oid.getField('path'))
    
class IntFire(Neutral):
    def __init__(self, *args, **kwargs):
	Neutral.__init__(self, *args, **kwargs)
 

def copy(src, dest, name, n=1, copyMsg=True):
    if isinstance(src, NeutralArray):
        src = src._id
    if isinstance(dest, NeutralArray):
        dest = dest._id
    new_id = _moose.copy(src=src, dest=dest, name=name, n=n, copyMsg=copyMsg)
    ret = NeutralArray(new_id)
    return ret

def move(src, dest):
    if isinstance(src, NeutralArray):
        src = src._id
    if isinstance(dest, NeutralArray):
        dest = dest._id
    _moose.move(src, dest)

def setCwe(element):
    if isinstance(element, NeutralArray):
        _moose.setCwe(element._id)
    elif isinstance(element, str):
        _moose.setCwe(element)
    else:
        raise TypeError('Argument must be an element or the path of a valid element.')

def getCwe():
    _id = _moose.getCwe()
    obj = NeutralArray(_id)
    return obj

# 
# moose.py ends here
