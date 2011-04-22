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
                            'fieldDimension',
                            'msgIn',
                            'msgOut',
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
                           'set_fieldDimension',
                           'get_fieldDimension',
                           'get_msgOut',
                           'get_msgIn',
                           'get_msgSrc',
                           'get_msgDest']
        self.sharedFinfos = []
    
    def setUp(self):
        self.testObj = moose.NeutralArray('neutral%d' % (uuid.uuid4().int))
        self.valueFinfos.sort()
        self.lookupFinfos.sort()
        self.srcFinfos.sort()
        self.destFinfos.sort()
        self.sharedFinfos.sort()
        
    def testFields(self):
        srcFields = sorted(list(self.testObj.getFieldNames('srcFinfo')))
        self.assertEqual(len(self.srcFinfos), len(srcFields))        
        for ii in range(len(self.srcFinfos)):
            self.assertEqual(self.srcFinfos[ii], srcFields[ii])
            
        destFields = sorted(list(self.testObj.getFieldNames('destFinfo')))
        self.assertEqual(len(self.destFinfos), len(destFields))        
        for ii in range(len(self.destFinfos)):
            self.assertEqual(self.destFinfos[ii], destFields[ii])

        valueFields = sorted(list(self.testObj.getFieldNames('valueFinfo')))
        self.assertEqual(len(self.valueFinfos), len(valueFields))        
        for ii in range(len(self.valueFinfos)):
            self.assertEqual(self.valueFinfos[ii], valueFields[ii])
        
        lookupFields = sorted(list(self.testObj.getFieldNames('lookupFinfo')))
        self.assertEqual(len(self.lookupFinfos), len(lookupFields))        
        for ii in range(len(self.lookupFinfos)):
            self.assertEqual(self.lookupFinfos[ii], lookupFields[ii])
    
        sharedFields = sorted(list(self.testObj.getFieldNames('sharedFinfo')))
        self.assertEqual(len(self.sharedFinfos), len(sharedFields))        
        for ii in range(len(self.sharedFinfos)):
            self.assertEqual(self.sharedFinfos[ii], sharedFields[ii])


    def testNew(self):
        a_path = 'neutral%d' % (uuid.uuid4().int)
        b_path = a_path + '/b'
        c_path = '/neutral%d' % (uuid.uuid4().int)
        d_path = c_path + '/d'
        a = moose.NeutralArray(a_path)
        b = moose.NeutralArray(b_path)
        c = moose.NeutralArray(c_path)
        d = moose.NeutralArray(d_path)
        self.assertEqual(a.path, '/' + a_path)
        self.assertEqual(b.path, '/' + b_path)
        self.assertEqual(c.path, c_path)
        self.assertEqual(d.path, d_path)
        self.assertEqual(b.name, 'b')
        self.assertEqual(d.name, 'd')
        self.assertRaises(ValueError, moose.NeutralArray, 'test/')
        
class TestPyMooseGlobals(unittest.TestCase):
    def setUp(self):
        self.src1 = moose.NeutralArray('/neutral%d' % (uuid.uuid4().int))
        self.dest1 = moose.NeutralArray('/neutral%d' % (uuid.uuid4().int))

    def testCopy(self):
        newname = 'neutral%d' % (uuid.uuid4().int)
        newobj = moose.copy(self.src1, self.dest1, newname, 3, True)
        self.assertEqual(newobj.path, self.dest1.path + "/" + newname)
        self.assertEqual(len(newobj), 3)

class TestMessages(unittest.TestCase):
    def setUp(self):
        self.src1 = moose.MolArray('/molecule%d' % (uuid.uuid4().int))
        self.dest1 = moose.ReacArray('/reaction%d' % (uuid.uuid4().int))

    def testConnect(self):
        self.src1[0].connect('reac', self.dest1[0], 'sub')
        print 1
        outmsgs_src = self.src1[0].msgOut
        print 2
        outmsgs_dest = self.dest1[0].msgOut
        print 3
        self.assertEqual(len(outmsgs_dest), len(outmsgs_src))
        print 4
        for ii in range(len(outmsgs_src)):
            self.assertEqual(outmsgs_src[ii], outmsgs_dest[ii])
            srcFieldsOnE1 = outmsgs_src[ii].getField('srcFieldsOnE1')
            self.assertEqual(srcFieldsOnE1[0], 'nOut')
            destFieldsOnE2 = outmsgs_src[ii].getField('destFieldsOnE2')
            self.assertEqual(destFieldsOnE2[0], 'subDest')
        # inmsg_list = self.dest1[0].msgIn
        # outmsg_list = self.src1[0].msgOut
        # print inmsg_list, outmsg_list
        # self.assertEqual(len(inmsg_list), 2)
        # self.assertEqual(len(outmsg_list), 1)
        # srcfield1 = inmsg_list[0].getField('srcFieldsOnE2')
        # destfield1 = inmsg_list[0].getField('destFieldsOnE1')
        # srcField2 = inmsg_list[0].getField('srcFieldsOnE1')
        # destField2 = inmsg_list[0].getField('destFieldsOnE2')
        # self.assertEqual(destfield1, 'reacDest')
        # self.assertEqual(srcfield1, 'nOut')
        # self.assertEqual(destField2, 'subDest')
        # self.assertEqual(srcField2, 'toSub')
        
if __name__ == '__main__':
    unittest.main()
