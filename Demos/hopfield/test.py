import moose
from pylab import *
from moose.utils import *

inputGiven = 1
moose.Neutral('/elec')
pg = moose.PulseGen('/elec/inPulGen')
pgTable = moose.Table('/elec/inPulGen/pgTable')
moose.connect(pgTable, 'requestData', pg, 'get_output')

pg.count = 1
pg.level[0] = 3
pg.width[0] = 10e-03
pg.delay[0] = 5e-02 #every 50ms

cellPath = '/cell'
cell = moose.IntFire(cellPath) #non programmer friendly numbering
cell.setField('tau',10e-3)
cell.setField('Vm', -0.07)
#cell.setField('refractoryPeriod', 0.1)
#cell.setField('thresh', 0.0)
cell.synapse.num = 1
cell.synapse[0].weight = 4
cell.synapse[0].delay = 1e-3

VmVal = moose.Table(cellPath+'/Vm_cell')
print 'table>cell:',moose.connect(VmVal, 'requestData', cell, 'get_Vm')

inSpkGen = moose.SpikeGen(cellPath+'/inSpkGen')
inSpkGen.setField('threshold', 2.0)
inSpkGen.setField('edgeTriggered', True)

if inputGiven == 1:
    print 'pulse>spike:', moose.connect(pg, 'outputOut', moose.element(cellPath+'/inSpkGen'), 'Vm')
    inTable = moose.Table(cellPath+'/inSpkGen/inTable')
    print 'table>spike:',moose.connect(inTable, 'requestData', inSpkGen, 'get_hasFired')

print 'spike>cell:', moose.connect(inSpkGen, 'event', cell.synapse[0] ,'addSpike')

moose.setClock(0, 1e-4)
moose.useClock(0, '/cell,/cell/##,/elec/##','process')
moose.reinit()
moose.start(0.2)

subplot(311)
plot(pgTable.vec[1:])
subplot(312)
plot(inTable.vec[1:])
subplot(313)
plot(VmVal.vec[1:])
show()
