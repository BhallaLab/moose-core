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

soma		none		0	0	28	19
h1		.		0	0	10	2
c1		.		0	0	10	1
c2		.		0	0	10	1
c3		.		0	0	10	1
c4		.		0	0	10	1
c5		.		0	0	10	1
c6		.		0	0	10	1
c7		.		0	0	10	1
c8		.		0	0	10	1
c9		.		0	0	10	1
c10		.		0	0	10	1
