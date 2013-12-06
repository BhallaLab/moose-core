import os
import subprocess
import sys
import unittest
import uuid

sys.path.append('../../python')
try:
    import moose
except ImportError:    
    print 'Please include the directory containing moose.py and _moose.so in your PYTHONPATH environmental variable.'
    sys.exit(1)

class TestEmatrix(unittest.TestCase):
    """Test pymoose basics"""
    def testCreate(self):
        em = moose.ematrix('test', 10, 0, 'Neutral')
        self.assertEqual(em.path, 'test')

    def testCreateKW(self):
        em = moose.ematrix(path='/testCreateKW', n=10, g=1, dtype='Neutral')
        self.assertEqual(em.path, '/testCreateKW')

    def testGetItem(self):
        em = moose.ematrix('testGetItem', n=10, g=1, dtype='Neutral')
        el = em[5]
        self.assertEqual(el.path,'/%s[5]' % (em.name))

    def testIndexError(self):
        em = moose.ematrix('testIndexError', n=3, g=1, dtype='Neutral')
        with self.assertRaises(IndexError):
            el = em[5]
        
    def testSlice(self):
        em = moose.ematrix('/testSlice', n=10, g=1, dtype='Neutral')
        sl = em[5:8]
        for ii, el in enumerate(sl):
            self.assertEqual(el.path,  '/testSlice[%d]' % (ii+5))

class TestNeutral(unittest.TestCase):
    def testPath(self):
        a = moose.Neutral('a')
        self.assertEqual(a.path, '/a[0]')

class TestNeutral1(unittest.TestCase):
    def setUp(self):
        self.a_path = 'neutral%d' % (uuid.uuid4().int)
        self.b_path = self.a_path + '/b'
        self.c_path = '/neutral%d' % (uuid.uuid4().int)
        self.d_path = self.c_path + '/d'
        self.c_len = 3
        self.a = moose.Neutral(self.a_path)
        self.b = moose.Neutral(self.b_path)
        self.c = moose.Neutral(self.c_path, self.c_len)
        print self.a_path, self.b_path
        print self.a.path, self.b.path
        print len(self.c.id_), self.c_len
                
    def testNew(self):
        self.assertTrue(moose.exists(self.a_path))

    def testNewChild(self):
        self.assertTrue(moose.exists(self.b_path))

    def testNewChildWithSingleDim(self):
        self.assertTrue(moose.exists(self.c_path))    

    def testDimension(self):
        self.assertEqual(self.c.id_.shape[0], self.c_len)

    def testLen(self):
        self.assertEqual(len(self.c.id_), self.c_len)

    def testPath(self):
        # Unfortunately the indexing in path seems unstable - in
        # async13 it is switched to have [0] for the first element,
        # breaking old code which was supposed to skip the [0] and
        # include the index only for second entry onwards.
        self.assertEqual(self.b.path, '/%s[0]/%s[0]' % (self.a_path, 'b'))
        em = moose.ematrix(self.c)
        self.assertEqual(em[1].path, self.c_path + '[1]')

    def testName(self):
        self.assertEqual(self.b.name, 'b')

    def testPathEndingWithSlash(self):
        self.assertRaises(ValueError, moose.Neutral, 'test/')

    def testNonExistentPath(self):
        self.assertRaises(ValueError, moose.Neutral, '/nonexistent_parent/invalid_child')

    def testDeletedCopyException(self):
        moose.delete(self.c.id_)
        self.assertRaises(ValueError, moose.Neutral, self.c)

    def testDeletedGetFieldException(self):
        moose.delete(self.c.id_)
        with self.assertRaises(ValueError):
            s = self.c.name

    def testDeletedParentException(self):
        moose.delete(self.a.id_)
        with self.assertRaises(ValueError):
            s = self.b.name

    def testIdObjId(self):
        id_ = moose.ematrix(self.a)
        self.assertEqual(id_, self.a.id_)

    def testCompareId(self):
        """Test the rich comparison between ids"""
        id1 = moose.ematrix('A', n=2, dtype='Neutral')
        id2 = moose.ematrix('B', n=4, dtype='Neutral')
        id3 = moose.ematrix('A')
        self.assertTrue(id1 < id2)
        self.assertEqual(id1, id3)
        self.assertTrue(id2 > id1)
        self.assertTrue(id2 >= id1)
        self.assertTrue(id1 <= id2)
    
    def testRename(self):
        """Rename an element in a Id and check if that was effective. This
        tests for setting values also."""
        id1 = moose.ematrix(path='/alpha', n=1, dtype='Neutral')
        id2 = moose.ematrix('alpha')
        id1[0].name = 'bravo'
        self.assertEqual(id1.path, '/bravo')
        self.assertEqual(id2.path, '/bravo')
        
        
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
        moose.delete(self.oid.id_)

    def testRepr(self):
        with self.assertRaises(ValueError):
            print(self.oid)

    def testGetField(self):
        with self.assertRaises(ValueError):
            print(self.oid.name)

# class TestValueFieldTypes(unittest.TestCase):
#     def setUp(self):
#         self.id_ = uuid.uuid4().int
#         self.container = moose.Neutral('/test%d' % (self.id_))
#         cwe = moose.getCwe()
#         self.model = moose.loadModel('../Demos/Genesis_files/Kholodenko.g', '%s/kholodenko' % (self.container.path))
#         moose.setCwe(cwe)
    
#     def testVecUnsigned(self):
#         x = moose.element('%s/kinetics' % (self.model.path))
#         self.assertTrue(len(x.meshToSpace) > 0)
        
if __name__ == '__main__':
    print 'PyMOOSE Regression Tests:'
    unittest.main()
