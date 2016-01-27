import moose
import pylab
import rdesigneur as rd

library = moose.Neutral( '/library' )
compt = rd.make_Chem_Oscillator( 'osc' )
moose.copy( compt, '/library/osc', 'dend' )
moose.copy( compt, '/library/osc', 'psd' )

rdes = rd.rdesigneur(
    turnOffElec = True,
    cellProto = [[ './cells/h10.CNG.swc', 'elec']],
    spineProto = [[ 'make_passive_spine()', 'spine' ] ],
    spineDistrib = [ ["spine", '#apical#,#dend#', '10e-6', '1e-6' ]],
    #chemProto = [['./chem/psd.sbml', 'spiny']],
    chemProto = [['/library/osc', 'osc']],
    chemDistrib =[[ 'osc', '#apical#,#dend#', 'install', 'H(p - 5e-4)' ]],
    plotList = [['#', '1', 'psd/a', 'conc', 'conc of a in PSD']]
)

rdes.buildModel()
print 'Done -1'

moose.reinit()
print 'Done 0'
moose.start( 0.05 )
print 'Done'

for i in moose.wildcardFind( '/model/chem/psd/s[]' ):
    print i.conc


rdes.display()
