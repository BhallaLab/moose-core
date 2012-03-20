import os
import subprocess
import sys
import unittest
import uuid

try:
    import moose
except ImportError:
    print 'Please include the directory containing moose.py and _moose.so in your PYTHONPATH environmental variable.'
    sys.exit(1)

class TestNeutralArray(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.valueFinfos = ['name',
                            'me',
                            'parent',
                            'children',
                            'path',
                            'class',
                            'linearSize',
                            'dimensions',
                            'lastDimension',
                            'localNumField',
                            'msgIn',
                            'msgOut',
                            # 'msgSrc',
                            # 'msgDest',
                            'this']
        self.lookupFinfos = [] #['msgSrc', 'msgDest'] # not clear what category these finfos belong to!
        self.srcFinfos = ['childMsg']
        self.destFinfos = ['parentMsg',
                           'set_this',
                           'get_this',
                           'set_name',
                           'get_name',
                           'get_me',
                           'get_parent',
                           'get_children',
                           'get_path',
                           'get_class',
                           'get_linearSize',
                           'get_dimensions',
                           'set_lastDimension',
                           'get_lastDimension',
                           'get_localNumField',
                           'get_msgOut',
                           'get_msgIn',                           
                           'get_msgDest',
                           'get_msgSrc',
                           ]
        self.sharedFinfos = []
    
    def setUp(self):
        path = 'neutral%d' % (uuid.uuid4().int)
        self.testObj = moose.NeutralArray(path)
        self.valueFinfos.sort()
        self.lookupFinfos.sort()
        self.srcFinfos.sort()
        self.destFinfos.sort()
        self.sharedFinfos.sort()
        
    def testFields(self):
        """This test has become pointless as the fields in the MOOSE
        Neutral class keep changing. No use manually listing them and
        verifying."""
        pass
        # print 'Testing fields ...',
        # srcFields = sorted(list(self.testObj.getFieldNames('srcFinfo')))
        # self.assertEqual(len(self.srcFinfos), len(srcFields))        
        # for ii in range(len(self.srcFinfos)):
        #     self.assertEqual(self.srcFinfos[ii], srcFields[ii])
            
        # destFields = sorted(list(self.testObj.getFieldNames('destFinfo')))
        # self.assertEqual(len(self.destFinfos), len(destFields))        
        # for ii in range(len(self.destFinfos)):
        #     self.assertEqual(self.destFinfos[ii], destFields[ii])

        # valueFields = sorted(list(self.testObj.getFieldNames('valueFinfo')))
        # self.assertEqual(len(self.valueFinfos), len(valueFields))        
        # for ii in range(len(self.valueFinfos)):
        #     self.assertEqual(self.valueFinfos[ii], valueFields[ii])
        
        # lookupFields = sorted(list(self.testObj.getFieldNames('lookupFinfo')))
        # self.assertEqual(len(self.lookupFinfos), len(lookupFields))        
        # for ii in range(len(self.lookupFinfos)):
        #     self.assertEqual(self.lookupFinfos[ii], lookupFields[ii])
    
        # sharedFields = sorted(list(self.testObj.getFieldNames('sharedFinfo')))
        # self.assertEqual(len(self.sharedFinfos), len(sharedFields))        
        # for ii in range(len(self.sharedFinfos)):
        #     self.assertEqual(self.sharedFinfos[ii], sharedFields[ii])
        # print 'OK'


    def testNew(self):
        print 'Testing array object creation ...',
        a_path = 'neutral%d' % (uuid.uuid4().int)
        b_path = a_path + '/b'
        c_path = '/neutral%d' % (uuid.uuid4().int)
        d_path = c_path + '/d'
        c_len = 3
        d_len = 4
        a = moose.NeutralArray(a_path)
        b = moose.NeutralArray(b_path)
        c = moose.NeutralArray(c_path, c_len)
        d = moose.NeutralArray(d_path, (d_len))
        self.assertEqual(a.path, '/' + a_path)
        self.assertEqual(b.path, '/' + b_path)
        self.assertEqual(c.path, c_path + '[0]')
        self.assertEqual(d.path, c_path + '[0]/d[0]')
        self.assertEqual(b.name, 'b')
        self.assertEqual(d.name, 'd')
        self.assertEqual(len(c), c_len)
        self.assertEqual(d.shape, (c_len, d_len))
        self.assertRaises(ValueError, moose.NeutralArray, 'test/')
        self.assertRaises(ValueError, moose.NeutralArray, '/nonexistent_parent/invalid_child')
        print 'OK'
        
class TestPyMooseGlobals(unittest.TestCase):
    def setUp(self):
        path1 = 'neutral%d' % (uuid.uuid4().int)
        path2 = 'neutral%d' % (uuid.uuid4().int)
        self.src1 = moose.NeutralArray(path1)
        self.dest1 = moose.NeutralArray(path2)

    def testCopy(self):
        print 'Testing copy ...',
        newname = 'neutral%d' % (uuid.uuid4().int)
        new_id = moose.copy(self.src1, self.dest1, newname, 3, toGlobal=False)
        new_obj = moose.NeutralArray(new_id)
        self.assertEqual(len(new_obj), 3)
        self.assertEqual(new_obj.path, self.dest1.path + "/" + newname + '[0]')
        print 'OK'

    def testElement(self):
        print 'Testing element() ...'
        x = moose.element(self.src1.path)
        self.assertTrue(isinstance(x, moose.Neutral))
        self.assertEqual(x.path, self.src1.path)
        x = moose.element(self.src1.id_)
        self.assertTrue(isinstance(x, moose.Neutral))
        self.assertEqual(x.path, self.src1.path)
        x = moose.element(self.src1[0].oid_)
        self.assertTrue(isinstance(x, moose.Neutral))
        self.assertEqual(x.path, self.src1.path)
        self.assertRaises(NameError, moose.element, 'abracadabra')
        

class TestMessages(unittest.TestCase):
    def setUp(self):
        path1 = '/comp%d' % (uuid.uuid4().int)
        path2 = '/comp%d' % (uuid.uuid4().int)
        self.src1 = moose.CompartmentArray(path1)
        self.dest1 = moose.CompartmentArray(path2)

    def testConnect(self):
        print 'Testing connect ...',
        self.assertTrue(self.src1[0].connect('raxial', self.dest1[0], 'axial'))
        outmsgs_src = self.src1[0].msgOut
        outmsgs_dest = self.dest1[0].msgOut
        self.assertEqual(len(outmsgs_dest), len(outmsgs_src))
        for ii in range(len(outmsgs_src)):
            self.assertEqual(outmsgs_src[ii], outmsgs_dest[ii])
            srcFieldsOnE1 = outmsgs_src[ii].getField('srcFieldsOnE1')
            self.assertEqual(srcFieldsOnE1[0], 'raxialOut')
            destFieldsOnE2 = outmsgs_src[ii].getField('destFieldsOnE2')
            self.assertEqual(destFieldsOnE2[0], 'handleRaxial')
        print 'OK'

    def test_getInMessageDict(self):
        print 'Testing getInMessageDict ...',
        indict = self.src1[0].getInMessageDict()
        self.assertTrue('parentMsg' in indict)
        

class TestNeighbors(unittest.TestCase):
    def setUp(self):
        self.pulsegen = moose.PulseGen('pulsegen')
        self.compartment = moose.Compartment('compartment')
        self.table = moose.Table('table')
        moose.connect(self.table, 'requestData', self.compartment, 'get_Im')
        moose.connect(self.pulsegen, 'outputOut', self.compartment, 'injectMsg')
        
    def testNeighborDict(self):
        print 'Testing neighbour dict ...'
        neighbors = self.compartment.neighborDict
        self.assertTrue(self.pulsegen.oid_ in [ n.oid_ for n in neighbors['injectMsg']])
        self.assertTrue(self.table.oid_ in [n.oid_ for n in neighbors['get_Im']])
        self.assertTrue(self.compartment.oid_ in [n.oid_ for n in self.pulsegen.neighborDict['outputOut']])
        self.assertTrue(self.compartment.oid_ in [n.oid_ for n in self.table.neighborDict['requestData']])
        print 'OK'
                      
            
        
if __name__ == '__main__':
    print 'PyMOOSE Regression Tests:'
    unittest.main()
