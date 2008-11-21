#!/usr/bin/env python
from moose import *

def traverseTree(node,prefix):	
	print prefix,node.path
	# BUG HERE - instead of /comp_0 .. /comp_3 it prints /sched/cj/t2 .. t5
	for i in node.children:	
		traverseTree(Neutral(i), prefix+' ')

def traverseHierarchy(node, prefix):
	hierarchy = []
	hierarchy.append(node)
	while(len(hierarchy)>0):
		child = hierarchy.pop()
		print child.path
		for c in child.children:
			hierarchy.append(Neutral(c))
		
			
if __name__ == '__main__':		
	r = Neutral('/')
	print 'Creating 10 compartments' 
	for i in range(10):
		pth='comp_'+str(i)
		Compartment(pth)
		
	print 'Now printing the list of elements'
	for i in r.children:
		print i
		n = Neutral(i)
		print n
		print n.path
	print 'Doing ce comp_1:'
	PyMooseBase.ce('comp_1')
	print 'Current working element- pwe:'
	PyMooseBase.pwe()
	newComp = Compartment('test')
	print 'Doing an le:'
	for i in PyMooseBase.le():
		print Neutral(i).path

       #	traverseTree(r,' ')
	print '==='
	traverseHierarchy(r,'')
