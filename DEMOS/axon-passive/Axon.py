# This code is meant to be run after loading the model in 'Axon.g'.
# 'Axon.g' loads 2 identical copies of an linear, passive neuron.
#
# These are helper functions to scale passive parameters Cm, Rm, Ra,
# diameter and length. By default the second copy ('/axon1') is
# modified.
#
# After scaling a passive parameter, run simulation again, and compare
# plots for '/axon' and '/axon1'.

from pymoose import tweak_field

path = '/axon1'
wildcard = path + '/#[TYPE=Compartment]'

def scale_cm( scale ):
	tweak_field( wildcard, 'Cm', '{0} * Cm'.format( scale ) )

def scale_ra( scale ):
	tweak_field( wildcard, 'Ra', '{0} * Ra'.format( scale ) )

def scale_rm( scale ):
	tweak_field( wildcard, 'Rm', '{0} * Rm'.format( scale ) )

def scale_diameter( scale ):
	tweak_field( wildcard, 'diameter', '{0} * diameter'.format( scale ) )
	
	cm_scale = scale
	tweak_field( wildcard, 'Cm', '{0} * Cm'.format( cm_scale ) )
	
	rm_scale = 1.0 / scale
	tweak_field( wildcard, 'Rm', '{0} * Rm'.format( rm_scale ) )
	
	ra_scale = 1.0 / ( scale * scale )
	tweak_field( wildcard, 'Ra', '{0} * Ra'.format( ra_scale ) )

def scale_length( scale ):
	tweak_field( wildcard, 'length', '{0} * length'.format( scale ) )
	
	cm_scale = scale
	tweak_field( wildcard, 'Cm', '{0} * Cm'.format( cm_scale ) )
	
	rm_scale = 1.0 / scale
	tweak_field( wildcard, 'Rm', '{0} * Rm'.format( rm_scale ) )
	
	ra_scale = scale
	tweak_field( wildcard, 'Ra', '{0} * Ra'.format( ra_scale ) )
