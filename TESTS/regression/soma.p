// coarse asymmetrical mitral cell (olf bulb) model : camit
// Control lines start with '*'. Valid control options are 
// *relative 			- relative coords.
// *absolute			- absolute coords.
// *asymmetric			- use asymmetric compartments
// *symmetric			- use symmetric compartments

// #	name	parent		x	y	z	d	ch	dens	ch	dens...
// *hsolve

*asymmetric
*relative

*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01

soma		none		0	0	28	19	Na_mit_usb	2e3	K_mit_usb	4e3
