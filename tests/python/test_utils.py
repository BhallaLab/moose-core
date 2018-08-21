# -*- coding: utf-8 -*-
import moose
import moose.utils as mu
import uuid
import unittest
import sys
from io import StringIO as _sio

class _TestMooseUtils(unittest.TestCase):
    def test_printtree(self):
        s = moose.Neutral('/cell')
        soma = moose.Neutral('%s/soma'% (s.path))
        d1 = moose.Neutral('%s/d1'% (soma.path))
        d2 = moose.Neutral('%s/d2'% (soma.path))
        d3 = moose.Neutral('%s/d3'% (d1.path))
        d4 = moose.Neutral('%s/d4'% (d1.path))
        d5 = moose.Neutral('%s/d5'% (s.path))
        orig_stdout = sys.stdout
        sys.stdout = _sio()
        mu.printtree(s)
        expected = """
cell
|
|__ soma
|  |
|  |__ d1
|  |  |
|  |  |__ d3
|  |  |
|  |  |__ d4
|  |
|  |__ d2
|
|__ d5
"""
        self.assertEqual(sys.stdout.getvalue(), expected)
        sys.stdout = _sio()
        s1 = moose.Neutral('cell1')
        c1 = moose.Neutral('%s/c1' % (s1.path))
        c2 = moose.Neutral('%s/c2' % (c1.path))
        c3 = moose.Neutral('%s/c3' % (c1.path))
        c4 = moose.Neutral('%s/c4' % (c2.path))
        c5 = moose.Neutral('%s/c5' % (c3.path))
        c6 = moose.Neutral('%s/c6' % (c3.path))
        c7 = moose.Neutral('%s/c7' % (c4.path))
        c8 = moose.Neutral('%s/c8' % (c5.path))
        mu.printtree(s1)
        expected1 = """
cell1
|
|__ c1
   |
   |__ c2
   |  |
   |  |__ c4
   |     |
   |     |__ c7
   |
   |__ c3
      |
      |__ c5
      |  |
      |  |__ c8
      |
      |__ c6
"""
        self.assertEqual(sys.stdout.getvalue(), expected1)

    def test_autoposition(self):
        """Simple check for automatic generation of positions.

        A spherical soma is created with 20 um diameter. A 100
        compartment cable is created attached to it with each
        compartment of length 100 um.

        """
        testid = 'test%s' % (uuid.uuid4())
        container = moose.Neutral('/test')
        model = moose.Neuron('/test/%s' % (testid))
        soma = moose.Compartment('%s/soma' % (model.path))
        soma.diameter = 20e-6
        soma.length = 0.0
        parent = soma
        comps = []
        for ii in range(100):
            comp = moose.Compartment('%s/comp_%d' % (model.path, ii))
            comp.diameter = 10e-6
            comp.length = 100e-6
            moose.connect(parent, 'raxial', comp, 'axial')
            comps.append(comp)
            parent = comp
        soma = mu.autoposition(model)
        sigfig = 8
        self.assertAlmostEqual(soma.x0, 0.0, sigfig)
        self.assertAlmostEqual(soma.y0, 0.0, sigfig)
        self.assertAlmostEqual(soma.z0, 0.0, sigfig)
        self.assertAlmostEqual(soma.x, 0.0, sigfig)
        self.assertAlmostEqual(soma.y, 0.0, sigfig)
        self.assertAlmostEqual(soma.z, soma.diameter/2.0, sigfig)
        for ii, comp in enumerate(comps):
            print(comp.path, ii)
            self.assertAlmostEqual(comp.x0, 0, sigfig)
            self.assertAlmostEqual(comp.y0, 0.0, sigfig)
            self.assertAlmostEqual(comp.z0, soma.diameter/2.0 + ii * 100e-6, sigfig)
            self.assertAlmostEqual(comp.x, 0.0, sigfig)
            self.assertAlmostEqual(comp.y, 0.0, sigfig)
            self.assertAlmostEqual(comp.z, soma.diameter/2.0 + (ii + 1) * 100e-6, sigfig)

if __name__ == "__main__": # test printtree
    unittest.main()
