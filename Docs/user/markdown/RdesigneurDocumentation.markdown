-----

# **Rdesigneur: Building multiscale models**

Upi Bhalla

Dec 28 2015.

-----

## Contents
	

## Introduction

**Rdesigneur** (Reaction Diffusion and Electrical SIGnaling in NEURons) is an
interface to the multiscale modeling capabilities in MOOSE. It is designed
to build models incorporating biochemical signaling pathways in 
dendrites and spines, coupled to electrical events in neurons. Rdesigneur
assembles models from predefined parts: it delegates the details to 
specialized model definition formats. Rdesigneur combines one or more of
the following cell parts to build models:

*	Neuronal morphology
*	Dendritic spines
*	Ion channels
*	Reaction systems

Rdesigneur's main role is to specify how these are put together, including 
assigning parameters to do so. Rdesigneur also helps with setting up the
simulation input and output.

## Quick Start
Here we provide a few use cases, building up from a minimal model to a 
reasonably complete multiscale model spanning chemical and electrical signaling.

### Bare Rdesigneur: single passive compartment
If we don't provide any arguments at all to the Rdesigneur, it makes a model
with a single passive electrical compartment in the MOOSE path 
`/model/elec/soma`. Here is how to do this:

	import moose
	import rdesigneur as rd
	rdes = rd.rdesigneur()
	rdes.buildModel()

To confirm that it has made a compartment with some default values we can add 
a line:
	
	moose.showfields( rdes.soma )

This should produce the output:

	[ /model[0]/elec[0]/soma[0] ]
	diameter         = 0.0005
	fieldIndex       = 0
	Ra               = 7639437.26841
	y0               = 0.0
	Rm               = 424413.177334
	index            = 0
	numData          = 1
	inject           = 0.0
	initVm           = -0.065
	Em               = -0.0544
	y                = 0.0
	numField         = 1
	path             = /model[0]/elec[0]/soma[0]
	dt               = 0.0
	tick             = -2
	z0               = 0.0
	name             = soma
	Cm               = 7.85398163398e-09
	x0               = 0.0
	Vm               = -0.06
	className        = ZombieCompartment
	idValue          = 465
	length           = 0.0005
	Im               = 1.3194689277e-08
	x                = 0.0005
	z                = 0.0


### Simulate and display current pulse to soma
A more useful script would run and display the model. Rdesigneur can help with
the stimulus and the plotting. This simulation has the same passive 
compartment, and current is injected as the simulation runs.
This script displays the membrane potential of the soma as it charges and 
discharges.

	import moose
	import rdesigneur as rd
	rdes = rd.rdesigneur(
		stimList = [['soma', '1', 'inject', 'sign((t - 0.1) * (0.2 - t) )*1e-8' ]],
		plotList = [['soma', '1', 'Vm', 'Soma membrane potential']],
	)
	rdes.buildModel()
	moose.reinit()
	moose.start( 0.3 )
	rdes.display()

The *stimList* defines a stimulus. Each entry has four arguments:

	`[region_in_cell, region_expression, parameter, expression_string]`

+	`region_in_cell` specifies the objects to stimulate. Here it is just the
	soma.
+	`region_expression` specifies a geometry based calculation to decide
	whether to apply the stimulus. The value must be >0 for the stimulus
	to be present. Here it is just 1.
+	`parameter` specifies the simulation parameter to assign. Here it is
	the injection current to the compartment.
+	`expression_string` calculates the value of the parameter, typically
	as a function of time. Here we use the function sign(x),
	where sign(x) == +1 for x > 0, 0 for x = 0 and -1 for x < 0. 

To summarise this, the *stimList* here means *inject a current of -10nA to the
soma up to 0.1 s, then inject +10nA up to 0.2 s, then inject -10 nA till the
end of the simulation*.

The *plotList* defines what to plot. It has a similar set of arguments:

	`[region_in_cell, region_expression, parameter, title_of_plot]`
These mean the same thing as for the stimList except for the title of the plot.

The *rdes.display()* function causes the plots to be displayed.

![Plot for current input to passive compartment](/home/bhalla/moose/master/moose-core/Docs/user/markdown/images/test2.png)

### HH Squid model in a single compartment
Here we put the Hodgkin-Huxley squid model channels into a passive compartment.
The HH channels are predefined as prototype channels for Rdesigneur,

	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.rdesigneur(
    	chanProto = [['make_HH_Na()', 'Na'], ['make_HH_K()', 'K']],
    	chanDistrib = [
        	['Na', 'soma', 'Gbar', '1200' ],
        	['K', 'soma', 'Gbar', '360' ]],
    	stimList = [['soma', '1', 'inject', '(t>0.1 && t<0.2) * 1e-8' ]],
    	plotList = [['soma', '1', 'Vm', 'Membrane potential']]
	)
	
	rdes.buildModel()
	moose.reinit()
	moose.start( 0.3 )
	rdes.display()


Here we introduce two new model specification lines:

+	chanProto: This specifies which ion channels will be used in the model.
	Each entry here has two fields: the source of the channel definition,
	and (optionally) the name of the channel.
	In this example we specify two channels, an Na and a K channel using
	the original Hodgkin-Huxley parameters. As the source of the channel
	definition we use the name of the  Python function that builds the 
	channel. The *make_HH_Na()* and *make_HH_K()* functions are predefined 
	but we can also specify our own functions for making prototypes.
	We could also have specified the channel prototype using the name
	of a channel definition file in ChannelML (a subset of NeuroML) format.
+	chanDistrib: This specifies  *where* the channels should be placed
	over the geometry of the cell. Each entry in the chanDistrib list 
	specifies the distribution of parameters for one channel using four 
	entries: 

	`[object_name, region_in_cell, parameter, expression_string]`

	In this case the job is almost trivial, since we just have a single 
	compartment named *soma*. So the line

	`['Na', 'soma', 'Gbar', '1200' ]`

	means *Put the Na channel in the soma, and set its maximal 
	conductance density (Gbar) to 1200 Siemens/m^2*. 

As before we apply a somatic current pulse. Since we now have HH channels in
the model, this generates action potentials.

### Reaction system in a single compartment
Here we use the compartment as a place in which to embed a chemical model.
The chemical oscillator model is predefined in the rdesigneur prototypes.

	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.rdesigneur(
    	turnOffElec = True,
    	diffusionLength = 1e-3, # The default diffusion length is 2 microns
    	chemProto = [['make_Chem_Oscillator()', 'osc']],
    	chemDistrib = [['osc', 'soma', 'install', '1' ]], 
    	plotList = [['soma', '1', 'dend/a', 'conc', 'a Conc'],
        	['soma', '1', 'dend/b', 'conc', 'b Conc']]
	)

	rdes.buildModel()
	bv = moose.vec( '/model/chem/dend/b' )
	bv[0].concInit *= 2
	moose.reinit()
	moose.start( 200 )

	rdes.display()


In this special case we set the turnOffElec flag to True, so that Rdesigneur 
only sets up chemical and not electrical calculations.  This makes the 
calculations much faster, since we disable electrical calculations and delink
chemical calculations from them.

We also have a line which sets the `diffusionLength` to 1 mm, so that it is 
bigger than the 0.5 mm squid axon segment in the default compartment. If you 
don't do this the system will subdivide the compartment into 2 micron voxels for
the purposes of putting in a reaction-diffusion system, which we discuss below.

### Reaction-diffusion system

In order to see what a reaction-diffusion system looks like, delete the
`diffusionLength` expression in the previous example: 

    	`diffusionLength = 1e-3,`

This tells the system to use the default 2 micron diffusion length. 
The 500-micron axon segment is now subdivided into 250 voxels, each of 
which has a reaction system and diffusing molecules. To make it more 
picturesque, we can add a line after the plotList, to display the outcome 
in 3-D:

    	`moogliList = [['soma', '1', 'dend/a', 'conc', 'b Conc']]`

### Make a toy multiscale model with electrical and chemical signaling.
Now we put together the previous two models. In this toy model we have a
HH-squid type single compartment electrical model, cohabiting with a chemical
oscillator. The chemical oscillator regulates K+ channel amounts, and the
average membrane potential regulates the amounts of a reactant in the 
chemical oscillator. This is a recipe for some strange firing patterns.


	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.buildRdesigneur(
		chanProto = [['makeHHNa()', 'Na'], ['makeHHK()', 'K']],
		chanDistrib = [ 
			['Na', 'soma', 'Gbar', '1250' ], 
			['K', 'soma', 'Gbar', '1000' ]],
		chemProto = [['./chem/osc.sbml', 'osc']],
		chemDistrib = [[ 'osc', 'soma', 'install', '1' ]],
		adaptorList = [
			[ 'dend/K', 'n', 'K', 'modulation', 0.5, 0.002 ],
			[ '.', 'Vm', 'osc_input', 'concInit', 0.1, 0.001 ]
		],
		plot = [['chem/soma', '1', 'A'],['soma', '1', 'Vm']],
	)
	moose.reinit()
	moose.start( 10 )


We've already modeled the HH squid model and the oscillator individually.
The new section that makes this work the *adaptorList* which specifies how 
the electrical and chemical parts talk to each other.
(stuff here)


### Morphology: Load .swc morphology file and view it
Here we build a passive model using a morphology file in the .swc file format
(as used by NeuroMorpho.org). The morphology file is predefined for Rdesigneur
and resides in the 
directory `./cells`. We apply a somatic current pulse, and view
the somatic membrane potential in a plot, as before. 
To make things interesting we display the morphology in 3-D upon which we
represent the membrane potential as colors.

	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.buildRdesigneur(
		cellProto = [[ './cells/h10.swc', 'elec']],
		stim = [['elec/soma', 'H((t - 0.1) * (0.2 - t) )*1e-9', 'inject']],
		plot = [['elec/soma', '1', 'Vm']],
		moogli = [['elec/#', '1', 'Vm']]
	)
	moose.reinit()
	moose.start( 0.3 )

### Build a spiny neuron from a morphology file and put active channels in it.
This is where we begin to use some of the power of Rdesigneur.
We decorate a bare neuronal morphology file with dendritic spines and
distribute voltage-gated ion channels over the neuron. This time the voltage-
gated channels are obtained from a number of channelML files, located in the
`./channels` subdirectory. Since we have a spatially extended neuron, 
we need to specify the spatial distribution of channel densities too. 


	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.buildRdesigneur(
		cellProto = [[ './cells/h10.swc', 'elec']],
		chanProto = [
			['./channels/hd.xml'],
			['./channels/kap.xml'],
			['./channels/kad.xml'],
			['./channels/kdr.xml'],
			['./channels/na3.xml'],
			['./channels/nax.xml'],
			['./channels/CaConc.xml'],
			['./channels/Ca.xml']
		],
		spineProto = [[ 'makePassiveSpineProto()', 'spine' ] ],
		chanDistrib = [ 
			['hd', '#dend#,#apical#,' 'Gbar', '5e-2*(1+(p*3e4))' ],
			['kdr', '#', 'Gbar', '100' ],
			['na3', '#', 'Gbar', '250' ],
			['nax', '#axon#', 'Gbar', '1250' ],
			['nax', '#soma#', 'Gbar', '100' ],
			['kap', '#axon#,#soma#', 'Gbar', '300' ],
			['kap', '#dend#,#apical#,#user#', 'Gbar',
				'300*(H(100-p*1e6)) * (1+(p*1e4))' ],
			['Ca_conc', '#', 'tau', '0.0133' ],
			['kad', '#dend#,#apical#', 'Gbar',
				'300*H(p*1e6-100)*(1+p*1e4)' ]
		],
		spineDistrib = [
			["spine", '#apical#,#dend#', "spineSpacing", "5e-6",
			"spineSpacingDistrib", "1e-6",
			"angle", "0",
			"angleDistrib", str( 2*PI ),
			"size", "1",
			"sizeDistrib", "0.5" ] 
		],
		stim = [['elec/soma', 'H((t - 0.02) * (0.12 - t) )*1e-9', 'inject']],
		plot = [['elec/soma', '1', 'Vm']],
		moogli = [['elec/#', '1', 'Vm']]
	)
	moose.reinit()
	moose.start( 0.15 )


As before, the channel distributions are specified by a list of entries each
containing:

`[name, region_in_cell, parameter, expression_string]`

- The *name* is the name of the prototype. This is usually an ion channel, 
but in the example above you can also see a calcium concentration pool
defined.
- The *region_in_cell* is typically defined using wildcards, so that it
generalizes to any cell morphology.
For example, the plain wildcard `#` means to consider 
all cell compartments. The wildcard `#dend#` means to consider all compartments with the string `dend`
somewhere in the name. Wildcards can be comma-separated, so 
`#soma#,#dend#` means consider all compartments with either soma or dend in
their name. The naming in MOOSE is defined by the model file. Importantly,
in **.swc** files MOOSE generates names that respect the classification of 
compartments into axon, soma, dendrite, and apical dendrite compartments 
respectively.
- The *parameter* is usually Gbar, the channel conductance density in *S/m^2*.
If *Gbar* is zero or less, then the system economizes by not incorporating any
calculations for this channel. Similarly, for calcium pools, if the *tau* is
below zero then the calcium pool object is simply not inserted into this part 
of the cell.
- The *expression_string* defines the value of the parameter, such as Gbar.
This is typically a function of position in the cell. The expression evaluator 
knows about several parameters of cell geometry. All units are in metres: 
	+ *x*, *y* and *z* coordinates.
	+ *g*, the geometrical distance from the soma
	+ *p*, the path length from the soma, measured along the dendrites. 
	+ *dia*, the diameter of the dendrite.
	+ *L*, The electrotonic length from the soma (no units).

Along with these geometrical arguments, we make liberal use of the Heaviside 
function H(x) to set up the channel distributions. The expression evaluator
also knows about pretty much all common algebraic, trignometric, and logarithmic
functions, should you wish to use these.

The spine distributions are specified in a similar way, but here we get to
see the full parameter definition string where we assign multiple parameters
for the spine distribution. We start out as before:

- *spine*: The prototype name
- *#apical#,#dend#'*: Put the spines on the any of ithe apical and basal dendrites.
- *'spineSpacing' '5e-6'*: Put the spines in 5 microns apart. Here the spacing
expression could have been any function of cell geometry, as above. Also, if
the spacing is zero or less, no spines are inserted.
- *'spineSpacingDistrib' '1e-6'*: Granularity for recomputing whether to put
in a new spine. In other words, every 1 micron we recompute whether to put in
a new spine. Given that the spacing is 5e-6, the likelihood of a spine coming
in to any given 1-micron segment is 0.2.
- *angle*: This specifies the initial angle at which the spine sticks out of
the dendrite. If all angles were zero, they would all point away from the soma.
- *angleDistrib*: Specifies a random number to add to the initial angle. In 
this example we have *2 * PI* as the range of the number, so the spines stick
out at any angle.
- *size*: Linear scale factor for size of spine. The default spine head here is 
0.5 microns in diameter and length. If the scale factor were to be 2, the
volume would be 8 times as large.
- *sizeDistrib*: Range for size of spine. A random number R is computed in the
range 0 to 1, and the final size used is `size + (R - 0.5) * sizeDistrib`.

### Build a spiny neuron from a morphology file and put a reaction-diffusion system in it.
Rdesigneur is specially designed to take reaction systems with a dendrite,
a spine head, and a spine PSD compartment, and embed these systems into 
neuronal morphologies. This example shows how this is done.

The dendritic molecules diffuse along the dendrite
in the region specified by the *chemDistrib* keyword. In this case they are
placed on all apical and basal dendrites, but only at distances over 
500 microns from the soma. The spine head and PSD 
reaction systems are inserted only into spines within this same *chemDistrib*
zone. Diffusion coupling between dendrite, and each spine head and PSD is also 
set up.
It takes a predefined chemical model file for Rdesigneur, which resides 
in the `./chem` subdirectory. As in an earlier example, we turn off the 
electrical calculations here as they are not needed. 
Here we plot out the number of receptors on every single spine as a function
of time.

	import moose
	import pylab
	import rdesigneur as rd
	rdes = rd.buildRdesigneur(
		turnOffElec = True,
		cellProto = [[ './cells/h10.swc', 'elec']],
		spineProto = [[ 'makePassiveSpineProto()', 'spine' ] ],
		spineDistrib = [
			["spine", '#apical#,#dend#', "spineSpacing", "5e-6",
			"spineSpacingDistrib", "1e-6",
			"angle", "0",
			"angleDistrib", str( 2*PI ),
			"size", "1",
			"sizeDistrib", "0.5" ] 
		],
		chemProto = [['./chem/psd.sbml', 'spiny']]
		chemDistrib = [[ 'spiny', '#apical#,#dend#', 'install', 'H(p - 5e-4)' ]],
		plot = [['chem/#/PSDR', '1', 'n']],
	)
	moose.reinit()
	moose.start( 0.15 )

### Make a full multiscale model with complex spiny morphology and electrical and chemical signaling.

(stuff here)
