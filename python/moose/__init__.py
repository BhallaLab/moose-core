"""MOOSE = Multiscale Object Oriented Simulation Environment.
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

ematrix
----------------

this is the unique identifier of a MOOSE object. Note that you can
create multiple references to the same MOOSE object in Python, but as
long as they have the same path/id value, they all point to the same
entity in MOOSE.

Constructor:

You can create a new ematrix using the constructor:

ematrix(path, dimension, classname)

Fields:

value -- unsigned integer representation of id of this ematrix

path -- string representing the path corresponding this ematrix

shape -- tuple containing the dimensions of this ematrix


Apart from these, every ematrix exposes the fields of all its elements
in a vectorized form. For example:

>>> iaf = moose.ematrix('/iaf', (10), 'IntFire')
>>> iaf.Vm = range(10) 
>>> print iaf[5].Vm 
5.0
>>> print iaf.Vm
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])


Methods:

ematrix implements part of the sequence protocol:

len(em) -- the first dimension of em.

em[n] -- the n-th element in em.

em[n1:n2] -- a tuple containing n1 to n2-th (exclusive) element in em.

elem in em -- True if elem is contained in em.



melement
----------------

Single moose object. It has three numbers to uniquely identify it:

id - id of the ematrix containing this element

dataIndex - index of this element in the container ematrix

fieldIndex - if this is a tertiary object, i.e. acts
as a field in another element (like synapse[0] in IntFire[1]), then
the index of this field in the containing element.

Methods:

getId -- ematrix object containing this element.
ematrix() -- ematrix object containing this element.

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

melement is something like an abstract base class in C++. The concrete
base class is Neutral. However you do not need to cast objects down to
access their fields. The PyMOOSE interface will automatically do the
check for you and raise an exception if the specified field does not
exist for the current element.

Creating melements
------------------

To create the objects of concrete subclasses of melement, the class
can be called as follows:

melement(path, dims, dtype, parent)

path: This is like unix filesystem path and is the concatenation of
name of the element to be created and that of all its ancestors
spearated by `/`. For example, path=`/a/b` will create the element
named `b` under element `a`. Note that if `a` does not exist, this
will raise an error. However, if `parent` is specified, `path` should
contain only the name of the element.

dims: (optional) tuple specifying the dimension of the containing melement to be
created. It is (1,) by default.

dtype: string specifying the class name of the element to be created.

parent: (optional) string specifying the path of the parent element or
the Id or the ObjId of the parent element or a reference to the parent
element. If this is specified, the first argument `path` is treated as
the name of the element to be created.

All arguments can be passed as keyword arguments.

For concrete subclasses of melement, you do not need to pass the class
argument because the class name is passed automatically to `melement`
__init__ method.

a = Neutral('alpha') # Creates element named `alpha` under current working element
b = Neutral('alpha/beta') # Creates the element named `beta` under `alpha`
c = Cell('charlie', parent=a) # creates element `charlie` under `alpha`
d = DiffAmp('delta', parent='alpha/beta') # creates element `delta` under `beta`


module functions
----------------

element(path) - returns a reference to an existing object converted to
the right class. Raises ValueError if path does not exist.

copy(src=<src>, dest=<dest>, name=<name_of_the_copy>, n=<num_copies>,
copyMsg=<whether_to_copy_messages) -- make a copy of source object as
a child of the destination object.


move(src, dest) -- move src object under dest object.

useClock(tick, path, update_function) -- schedule <update_function> of
every object that matches <path> on clock no. <tick>. Most commonly
the function is 'process'.  NOTE: unlike earlier versions, now
autoschedule is not available. You have to call useClock for every
element that should be updated during the simulation. 

The sequence of clockticks with the same dt is according to their
number. This is utilized for controlling the order of updates in
various objects where it matters.

The following convention should be observed when assigning clockticks
to various components of a model:

Clock ticks 0-3 are for electrical (biophysical) components, 4 and 5
are for chemical kinetics, 6 and 7 are for lookup tables and stimulus,
8 and 9 are for recording tables.

Generally, 'process' is the method to be assigned a clock
tick. Notable exception is 'init' method of Compartment class which is
assigned tick 0.

0 : Compartment: 'init'
1 : Compartment: 'process'
2 : HHChannel and other channels: 'process'
3 : CaConc : 'process'
4,5 : Elements for chemical kinetics : 'process'
6,7 : Lookup (tables), stimulus : 'process'
8,9 : Tables for plotting : process

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
model-tree, or an ematrix.
ce(obj) -- an alias for setCwe.

getCwe() -- returns ematrix containing the current working element.
pwe() -- an alias for getCwe.

showfields(obj) -- print the fields in object in human readable format

le(obj) -- list element under object, if no parameter specified, list
elements under current working element

"""

__version__ = '$Revision: 4454$'
# $Source$

from .moose import *
