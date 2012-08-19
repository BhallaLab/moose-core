import os
import subprocess
import sys
import unittest
import uuid

sys.path.append('../python')
try:
    import moose
except ImportError:    
    print 'Please include the directory containing moose.py and _moose.so in your PYTHONPATH environmental variable.'
    sys.exit(1)

class TestObjId(unittest.TestCase):
    def setUp(self):
        self.a_path = 'neutral%d' % (uuid.uuid4().int)
        self.b_path = self.a_path + '/b'
        self.c_path = '/neutral%d' % (uuid.uuid4().int)
        self.d_path = self.c_path + '/d'
        self.c_len = 3
        self.d_dim = (4, 3)
        self.a = moose.Neutral(self.a_path)
        self.b = moose.Neutral(self.b_path)
        self.c = moose.Neutral(self.c_path, self.c_len)
        self.d = moose.Neutral(self.d_path, self.d_dim)
                
    def testNew(self):
        self.assertTrue(moose.exists(self.a_path))

    def testNewChild(self):
        self.assertTrue(moose.exists(self.b_path))

    def testNewChildWithSingleDim(self):
        self.assertTrue(moose.exists(self.c_path))    

    def testNewChildWithTupleDim(self):
        self.assertTrue(moose.exists(self.d_path))

    def testDimension(self):
        self.assertEqual(self.d.getId().shape[0], self.c_len)
        self.assertEqual(self.d.getId().shape[1], self.d_dim[0])
        self.assertEqual(self.d.getId().shape[2], self.d_dim[1])

    def testPath(self):
        self.assertEqual(self.b.path, '/' + self.b_path)
        self.assertEqual(self.c.path, self.c_path + '[0]')
        self.assertEqual(self.d.path, self.c_path + '[0]/d[0][0]')

    def testName(self):
        self.assertEqual(self.b.name, 'b')
        self.assertEqual(self.d.name, 'd')

    def testPathEndingWithSlash(self):
        self.assertRaises(ValueError, moose.Neutral, 'test/')

    def testNonExistentPath(self):
        self.assertRaises(ValueError, moose.Neutral, '/nonexistent_parent/invalid_child')

    def testDeletedCopyException(self):
        moose.delete(self.d.getId())
        self.assertRaises(ValueError, moose.Neutral, self.d)

    def testDeletedGetFieldException(self):
        moose.delete(self.d.getId())
        with self.assertRaises(ValueError):
            s = self.d.name

    def testDeletedParentException(self):
        moose.delete(self.a.getId())
        with self.assertRaises(ValueError):
            s = self.b.name
        
# class TestPyMooseGlobals(unittest.TestCase):
#     def setUp(self):
#         path1 = 'neutral%d' % (uuid.uuid4().int)
#         path2 = 'neutral%d' % (uuid.uuid4().int)
#         self.src1 = moose.Id(path1, 1, 'Neutral')
#         self.dest1 = moose.Id(path2, 1, 'Neutral')

#     def testCopy(self):
#         print 'Testing copy ...',
#         newname = 'neutral%d' % (uuid.uuid4().int)
#         new_id = moose.copy(self.src1, self.dest1, newname, 3, toGlobal=False)
#         self.assertEqual(len(new_id), 3)
#         self.assertEqual(new_obj.path, self.dest1.path + "/" + newname + '[0]')
#         print 'OK'

#     def testElement(self):
#         print 'Testing element() ...'
#         x = moose.element(self.src1.path)
#         self.assertTrue(isinstance(x, moose.Neutral))
#         self.assertEqual(x.path, self.src1.path)
#         x = moose.element(self.src1.id_)
#         self.assertTrue(isinstance(x, moose.Neutral))
#         self.assertEqual(x.path, self.src1.path)
#         x = moose.element(self.src1[0].oid_)
#         self.assertTrue(isinstance(x, moose.Neutral))
#         self.assertEqual(x.path, self.src1.path)
#         self.assertRaises(NameError, moose.element, 'abracadabra')
        

# class TestMessages(unittest.TestCase):
#     def setUp(self):
#         path1 = '/comp%d' % (uuid.uuid4().int)
#         path2 = '/comp%d' % (uuid.uuid4().int)
#         self.src1 = moose.Compartment(path1)
#         self.dest1 = moose.Compartment(path2)

#     def testConnect(self):
#         print 'Testing connect ...',
#         self.assertTrue(self.src1.connect('raxial', self.dest1, 'axial'))
#         outmsgs_src = self.src1.msgOut
#         outmsgs_dest = self.dest1.msgOut
#         self.assertEqual(len(outmsgs_dest), len(outmsgs_src))
#         for ii in range(len(outmsgs_src)):
#             self.assertEqual(outmsgs_src[ii], outmsgs_dest[ii])
#             srcFieldsOnE1 = outmsgs_src[ii].getField('srcFieldsOnE1')
#             self.assertEqual(srcFieldsOnE1[0], 'raxialOut')
#             destFieldsOnE2 = outmsgs_src[ii].getField('destFieldsOnE2')
#             self.assertEqual(destFieldsOnE2[0], 'handleRaxial')
#         print 'OK'

#     def test_getInMessageDict(self):
#         print 'Testing getInMessageDict ...',
#         indict = self.src1.getInMessageDict()
#         self.assertTrue('parentMsg' in indict)
        

# class TestNeighbors(unittest.TestCase):
#     def setUp(self):
#         self.pulsegen = moose.PulseGen('pulsegen')
#         self.compartment = moose.Compartment('compartment')
#         self.table = moose.Table('table')
#         moose.connect(self.table, 'requestData', self.compartment, 'get_Im')
#         moose.connect(self.pulsegen, 'outputOut', self.compartment, 'injectMsg')
        
#     def testNeighborDict(self):
#         print 'Testing neighbour dict ...'
#         neighbors = self.compartment.neighborDict
#         self.assertTrue(self.pulsegen.oid_ in [ n.oid_ for n in neighbors['injectMsg']])
#         self.assertTrue(self.table.oid_ in [n.oid_ for n in neighbors['get_Im']])
#         self.assertTrue(self.compartment.oid_ in [n.oid_ for n in self.pulsegen.neighborDict['outputOut']])
#         self.assertTrue(self.compartment.oid_ in [n.oid_ for n in self.table.neighborDict['requestData']])
#         print 'OK'
                      
class TestDelete(unittest.TestCase):
    def setUp(self):
        self.oid = moose.Neutral('a')
        moose.delete(self.oid.getId())

    def testRepr(self):
        with self.assertRaises(ValueError):
            print(self.oid)

    def testGetField(self):
        with self.assertRaises(ValueError):
            print(self.oid.name)
        
if __name__ == '__main__':
    print 'PyMOOSE Regression Tests:'
    unittest.main()
