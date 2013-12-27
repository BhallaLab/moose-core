"""This code is under testing: checking the protocol for
FieldElements. Once that is cleaned up, this can be reused as a
demo.

>>> import moose
on node 0, numNodes = 1, numCores = 8
Info: Time to define moose classes:0
Info: Time to initialize module:0.05
>>> a = moose.IntFire('a')
Created 123 path=a numData=1 isGlobal=0 baseType=IntFire
>>> b = moose.Synapse('a/b')
Created 125 path=a/b numData=1 isGlobal=0 baseType=Synapse
>>> b.numSynapse = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'moose.Synapse' object has no attribute 'numSynapse'
>>> a.numSynapse = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'moose.IntFire' object has no attribute 'numSynapse'
>>> dir(a)
['__class__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'connect', 'ematrix', 'getDataIndex', 'getField', 'getFieldIndex', 'getFieldNames', 'getFieldType', 'getId', 'getLookupField', 'getNeighbors', 'neighbours', 'parentMsg', 'process', 'reinit', 'setDestField', 'setField', 'setLookupField', 'synapse']
>>> a.synapse
/a.synapse
>>> a.synapse[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: moose.ElementField.getItem: index out of bounds.
>>> a.synapse.num = 10
>>> a.synapse[0]
<moose.Synapse: id=124, dataId=0, path=/a/synapse[0]>
>>> a.synapse[1]
<moose.Synapse: id=124, dataId=1, path=/a/synapse[0]>
>>> 


"""
import moose

a = moose.IntFire('alpha', 10)
a.synapse.num = 3
print a.synapse[0], a.synapse[1] # The fieldIndex should change, not dataId
b = moose.element('alpha[0]/synapse[1]')
c = moose.element('alpha[1]/synapse[2]')
print b, c
# The fieldIndex should change, not dataId
x = moose.element(a.id_, 0, 1)
y = moose.element(a.id_, 1, 2)
print x, y
