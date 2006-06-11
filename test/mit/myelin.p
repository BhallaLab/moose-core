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

*start_cell /library/node
node		none		0	0	20	1	Na_mit_usb	8e3	K_mit_usb	16e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.001
i1		.		0	0	10	1
i2		.		0	0	10	1
i3		.		0	0	10	1
i4		.		0	0	10	1
i5		.		0	0	10	1
i6		.		0	0	10	1
i7		.		0	0	10	1
i8		.		0	0	10	1
i9		.		0	0	10	1
i10		.		0	0	10	1
i11		.		0	0	10	1
i12		.		0	0	10	1
i13		.		0	0	10	1
i14		.		0	0	10	1
i15		.		0	0	10	1
i16		.		0	0	10	1
i17		.		0	0	10	1
i18		.		0	0	10	1
i19		.		0	0	10	1
i20		.		0	0	10	1

*makeproto /library/node

*start_cell /library/rnode
rnode		none		0	0	10	1	Na_mit_usb	8e3	K_mit_usb	16e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.001
i1		.		0	0	-10	1
i2		.		0	0	-10	1
i3		.		0	0	-10	1
i4		.		0	0	-10	1
i5		.		0	0	-10	1
i6		.		0	0	-10	1
i7		.		0	0	-10	1
i8		.		0	0	-10	1
i9		.		0	0	-10	1
i10		.		0	0	-10	1
i11		.		0	0	-10	1
i12		.		0	0	-10	1
i13		.		0	0	-10	1
i14		.		0	0	-10	1
i15		.		0	0	-10	1
i16		.		0	0	-10	1
i17		.		0	0	-10	1
i18		.		0	0	-10	1
i19		.		0	0	-10	1
i20		.		0	0	-10	1

*makeproto /library/rnode

*start_cell /library/snode
snode		none		0	0	10	1	Na_mit_usb	8e3	K_mit_usb	16e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.001
i1		.		0	10	0	1
i2		.		0	10	0	1
i3		.		0	10	0	1
i4		.		0	10	0	1
i5		.		0	10	0	1
i6		.		0	10	0	1
i7		.		0	10	0	1
i8		.		0	10	0	1
i9		.		0	10	0	1
i10		.		0	10	0	1
i11		.		0	10	0	1
i12		.		0	10	0	1
i13		.		0	10	0	1
i14		.		0	10	0	1
i15		.		0	10	0	1
i16		.		0	10	0	1
i17		.		0	10	0	1
i18		.		0	10	0	1
i19		.		0	10	0	1
i20		.		0	10	0	1

*makeproto /library/snode

*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01

*start_cell
soma		none		0	0	28	19	Na_mit_usb	2e3	K_mit_usb	4e3
*compt /library/node
n1		.		0	0	10	1
n2		n1/i20		0	0	10	1
n3		n2/i20		0	0	10	1
n4		n3/i20		0	0	10	1
n5		n4/i20		0	0	10	1
n6		n5/i20		0	0	10	1
n7		n6/i20		0	0	10	1
n8		n7/i20		0	0	10	1
n9		n8/i20		0	0	10	1
*compt /library/snode
n10		n9/i20		0	10	0	1
*compt /library/rnode
n11		n10/i20		0	0	-10	1
n12		n11/i20		0	0	-10	1
n13		n12/i20		0	0	-10	1
n14		n13/i20		0	0	-10	1
n15		n14/i20		0	0	-10	1
n16		n15/i20		0	0	-10	1
n17		n16/i20		0	0	-10	1
n18		n17/i20		0	0	-10	1
n19		n18/i20		0	0	-10	1
*compt /library/snode
n20		n19/i20		0	10	0	1
*compt /library/node
n21		n20/i20		0	0	10	1
n22		n21/i20		0	0	10	1
n23		n22/i20		0	0	10	1
n24		n23/i20		0	0	10	1
n25		n24/i20		0	0	10	1
n26		n25/i20		0	0	10	1
n27		n26/i20		0	0	10	1
n28		n27/i20		0	0	10	1
n29		n28/i20		0	0	10	1
*compt /library/snode
n30		n29/i20		0	10	0	1
*compt /library/rnode
n31		n30/i20		0	0	-10	1
n32		n31/i20		0	0	-10	1
n33		n32/i20		0	0	-10	1
n34		n33/i20		0	0	-10	1
n35		n34/i20		0	0	-10	1
n36		n35/i20		0	0	-10	1
n37		n36/i20		0	0	-10	1
n38		n37/i20		0	0	-10	1
n39		n38/i20		0	0	-10	1
*compt /library/snode
n40		n39/i20		0	10	0	1
*compt /library/node
n41		n40/i20		0	0	10	1
n42		n41/i20		0	0	10	1
n43		n42/i20		0	0	10	1
n44		n43/i20		0	0	10	1
n45		n44/i20		0	0	10	1
n46		n45/i20		0	0	10	1
n47		n46/i20		0	0	10	1
n48		n47/i20		0	0	10	1
n49		n48/i20		0	0	10	1
*compt /library/snode
n50		n49/i20		0	10	0	1
