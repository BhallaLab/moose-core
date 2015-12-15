# Issue 34 on github.
import sys
sys.path.append('../../python')
import moose
import moose.utils as mu

compt = moose.CubeMesh('/compt')
compt.volume = 1e-20

molecules = [ "ca", "S", "S1", "Sp" ]
pools = {}
tables = {}
for m in molecules:
    if m == 'S1':
        pools[m] = moose.BufPool('/compt/%s' % m)
    else:
        pools[m] = moose.Pool('/compt/%s' % m)
    t = moose.Table2("/table%s" % m)
    tables[m] = t
    moose.connect(t, 'requestOut', pools[m], 'getConc')

pools['ca'].nInit = 20

r_p0_p1 = moose.Reac('/compt/reacA')
funA = moose.Function('/compt/S1/func')
funA.expr = "{0}*(y0/{1})^6/(1+(y0/{1})^3)^2".format("1.5", "0.7e-3")
print funA.expr
moose.connect(funA, 'requestOut',  pools['ca'], 'getConc')
moose.connect(funA, 'valueOut', pools['S1'], 'setConc')
moose.connect(r_p0_p1, 'sub', pools['S'], 'reac')
moose.connect(r_p0_p1, 'prd', pools['S1'], 'reac')

r_p1_up = moose.Reac('/compt/reacB')
moose.connect(r_p1_up, 'sub', pools['S1'], 'reac')
moose.connect(r_p1_up, 'prd', pools['Sp'], 'reac')

# Change GSolve to Ksolve and bugs go away.
k = moose.Gsolve('/compt/ksolve')
s = moose.Stoich('/compt/stoich')
s.compartment = compt
s.ksolve = k
s.path = '/compt/##'

moose.reinit()
moose.start(10)
