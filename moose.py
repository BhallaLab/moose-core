# moose.py --- 
# 
# Filename: moose.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sat Mar 12 14:02:40 2011 (+0530)
# Version: 
# Last-Updated: Thu Aug 25 18:35:42 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 831
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
_oid) - a triplet of id, dataIndex and fieldIndex. Here id is the Id
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

Methods:

Neutral and all its derivatives will have a bunch of methods that are
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

getCwe() -- returns Id of the current working element.


"""

__version__ = "$Revision$"
# $Source$

import _moose
from _moose import useClock, setClock, start, reinit, stop, isRunning, loadModel, getFieldDict, Id, ObjId, exists


class _MooseDescriptor(object):
    """Descriptor to give access to MOOSE class' ValueFinfo attributes"""
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj._oid.getField(self.name)

    def __set__(self, obj, value):
        obj._oid.setField(self.name, value)

    def __delete__(self, obj):
        raise AttributeError('ValueFinfos cannot be deleted.')


    
    
class _MooseMeta(type):
    def __init__(cls, name, bases, classdict):        
        super(_MooseMeta, cls).__init__(name, bases, classdict)
        field_dict = getFieldDict('/classes/%s' % (name), 'valueFinfo')
        for field in field_dict.keys():
            setattr(cls, field, _MooseDescriptor(field))
        

class NeutralArray(object):
    def __init__(self, *args, **kwargs):
        path ='/'
        dims = [1]
        self._id = None
        try:
            className = kwargs['type']
            self.className = className
        except KeyError:
            # This code is messy and depends on the class name. I could not find a way to pass the element data type to the class definition dynamically
            if not hasattr(self, 'className'):
                self.className = 'Neutral'
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
            self.className = args[2]
        if self._id is None:
            self._id = _moose.Id(path=path, dims=dims, type=self.className)
        orig_classname = self._id[0].getField('class')
        if self.__class__.__name__ != orig_classname+'Array':
            orig_class = eval('%sArray' % (orig_classname))
            if self.__class__ not in orig_class.mro():        
                self._id = None
                raise TypeError('Cannot convert %s to %s' % (orig_class, self.__class__))

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

    def __contains__(self, other):
        if isinstance(other, Neutral):
            return other._oid in self._id
        elif isinstance(other, ObjId):
            return other in self._id
        else:
            return False

    def __repr__(self):
        return self._id.getPath()

    path = property(lambda self: self._id.getPath())
    id_ = property(lambda self: self._id)
    fieldNames = property(lambda self: self._id[0].getFieldNames('valueFinfo'))
    name = property(lambda self: self._id[0].getField('name'))
    shape = property(lambda self: self._id.getShape())
    

        
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
            elif isinstance(args[0], Neutral):
                (id_, dindex, findex) = (args[0]._oid.getId().getValue(),args[0]._oid.getDataIndex(), args[0]._oid.getFieldIndex())
            elif isinstance(args[0], NeutralArray):
                id_ = args[0]._id
            elif isinstance(args[0], str):
                path = args[0].replace('[', ' ').replace(']', ' ')
                components = path.split('/')
                current_path = ''
                for component in components:
                    if not component:
                        continue
                    tokens = component.split()
                    current_path = current_path + '/' + tokens[0]
                    if not _moose.exists(current_path):
                        if component == components[-1]: # this is the last entry and does not exist, so create a new one
                            class_obj = eval(self.__class__.__name__ + 'Array')
                            array_obj = class_obj(current_path)
                            id_ = array_obj.id_
                        else:
                            raise NameError('%s does not exist.' % (current_path))
                    else:
                        array_obj = Id(current_path)
                        print 'Current path', current_path
                        for index in tokens[1:]:
                            print 'Index:', index
                            array_obj = array_obj[int(index)]
                            print array_obj
                        if isinstance(array_obj, Id):
                            id_ = array_obj
                        elif isinstance(array_obj, ObjId):
                            id_ = array_obj.getId()
                            dindex = array_obj.getDataIndex()
                            findex = array_obj.getFieldIndex()
                        
                    
            else:
                raise TypeError('First non-keyword argument must be a number or an existing Id/ObjId/Neutral/NeutralArray object.')
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
        self._oid = _moose.ObjId(id_, dindex, findex)
        orig_classname = self._oid.getField('class')
        if self.__class__.__name__ != orig_classname:
            orig_class = eval(orig_classname)
            if self.__class__ not in orig_class.mro():
                self._oid = None
                raise TypeError('Cannot convert %s to %s' % (orig_class, self.__class__))

    def getField(self, field):
        return self._oid.getField(field)

    def getFieldType(self, field):
        return self._oid.getFieldType(field)

    def getFieldNames(self, fieldType=None):
        if fieldType is None:
            return self._oid.getFieldNames()
        else:
            return self._oid.getFieldNames(fieldType)

    def getMsgSrc(self, fieldName):
        return self._oid.getMsgSrc(fieldName)
    
    def getMsgDest(self, fieldName):
        return self._oid.getMsgDest(fieldName)

    def connect(self, srcField, dest, destField, msgType='Single'):
        return self._oid.connect(srcField, dest._oid, destField, msgType)

    
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
    
    
    className = property(lambda self: self._oid.getField('class'))
    fieldNames = property(lambda self: self._oid.getFieldNames())
    name = property(lambda self: self._oid.getField('name'))
    path = property(lambda self: '%s[%d]' % (self._oid.getField('path'), self._oid.getDataIndex()))
    id_ = property(lambda self: self._oid.getId())
    fieldIndex = property(lambda self: self._oid.getFieldIndex())
    dataIndex = property(lambda self: self._oid.getDataIndex())


################################################################
# Wrappers for global functions
################################################################ 

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

def delete(target):
    if isinstance(target, NeutralArray):
        target = target._id
    if not isinstance(target, Id):
        raise TypeError('Only Id or Array objects can be deleted: received %s' % (target.__class__.__name__))
    _moose.delete(target)
    
def setCwe(element):
    if isinstance(element, NeutralArray):
        _moose.setCwe(element._id)
    elif isinstance(element, Neutral):
        _moose.setCwe(element._oid)
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
        src = src._oid
    if isinstance(dest, Neutral):
        dest = dest._oid
    return src.connect(srcMsg, dest, destMsg, msgType)

def le(element=None):
    if element is None:
        element = getCwe()[0]
    elif isinstance(element, str):
        element = Neutral(element)
    print 'Elements under', element.path
    for ch in element.children:
        print ch
    

#######################################################
# This is to generate class definitions automatically
#######################################################
def define_class(classId):
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
    class_obj = type(class_name, (base_class,), {'className': class_name})
    field_dict = getFieldDict('/classes/%s' % (class_name), 'valueFinfo')
    for field in field_dict.keys():
        setattr(class_obj, field, _MooseDescriptor(field))
    # Go through the destFinfos and make them look like methods
    known_types = ['char', 'short', 'int', 'unsigned int', 'double', 'float', 'long', 'unsigned long', 'string']
    # print 'Processing:', class_name
    for finfoName, finfoArgs in getFieldDict(class_name, 'destFinfo').items():
        # print finfoName, finfoArgs
        cont_flag = False
        # 'get_...' and 'set_...' are internal destFinfos
        if finfoName.startswith('get_') or finfoName.startswith('set_'):
            continue
        destfnArgs = finfoArgs.split(',')
        fndef = 'lambda self'
        fnargs = "'%s'" % (finfoName)
        index = 0
        for arg in destfnArgs:
            # print arg
            if arg not in known_types:
                # print 'Cannot handle argument of type:', arg                # for debug
                cont_flag = True
                break
            fndef += ', arg_%d_%s' % (index, arg.replace(' ', '_'))
            fnargs += ', arg_%d_%s' % (index, arg.replace(' ', '_'))
            index += 1
        if cont_flag:
            continue
        function_string = '%s: self._oid.setDestField(%s)' % (fndef, fnargs)
        # print 'Setting %s.%s - Function string: %s' % (class_obj.__name__, finfoName, function_string)
        setattr(class_obj, finfoName, eval(function_string))

    globals()[class_name] = class_obj
    array_class_name = class_name + 'Array'
    if base != 'none':
        base_class = globals()[base + 'Array']
    else:
        base_class = object
    array_class_obj = type(array_class_name, (base_class,), {'className':class_name})
    globals()[array_class_name] = array_class_obj
        
            

classes_Id = Id('/classes')
class_obj_list = classes_Id[0].getField('children')    
for child in class_obj_list:
    define_class(child)


# 
# moose.py ends here
