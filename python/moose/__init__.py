"""
MOOSE = Multiscale Object Oriented Simulation Environment.
=========================================================

How to use the documentation
----------------------------

MOOSE documentation is split into Python documentation and builtin
documentation. The functions and classes that are only part of the
Python interface can be viewed via Python's builtin ``help``
function::

>>> help(moose.connect)
...

The documentation built into main C++ code of MOOSE can be accessed
via the module function ``doc``:

>>> moose.doc('Neutral')
...

To get documentation about a particular field:

>>> moose.doc('Neutral.childMsg')


Brief overview of PyMOOSE
=========================

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

getSources(fieldName) - return a list of (source_element, source_field) for all 
messages coming in to fieldName of this object.

getDestinations(fieldName) - return a list of (destination_elemnt, destination_field)
for all messages going out of fieldName.


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

useClock(tick, path, update_function) -- schedule <update_function> of
every object that matches <path> on clock no. <tick>. Most commonly
the function is 'process'.  NOTE: unlike earlier versions, now
autoschedule is not available. You have to call useClock for every
element that should be updated during the simulation. 

Example: 
moose.useClock(0, '/model/compartment_1', 'init')
moose.useClock(1, '/model/compartment_1', 'process')

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

from .moose import *
