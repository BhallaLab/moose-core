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
moose.connect(VmVal, 'requestData', cell, 'get_Vm')

inSpkGen = moose.SpikeGen(cellPath+'/inSpkGen')
inSpkGen.setField('threshold', 2.0)
inSpkGen.setField('edgeTriggered', 1)

if inputGiven == 1:
    moose.connect(pg, 'outputOut', moose.element(cellPath+'/inSpkGen'), 'Vm')
    inTable = moose.Table(cellPath+'/inSpkGen/inTable')
    moose.connect(inTable, 'requestData', inSpkGen, 'get_hasFired')

moose.connect(inSpkGen, 'event', cell.synapse[0] ,'addSpike') #self connection is the input 

resetSim([cellPath,'/elec'],1e-4,1e-3)
moose.start(0.2)

plot(pgTable.vec[1:])
plot(float(1)+inTable.vec[1:])
plot(float(2)+VmVal.vec[1:])
show()
