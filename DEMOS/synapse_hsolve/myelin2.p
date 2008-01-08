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
node		none		0	0	10	0.5	Na_mit_usb	15e3	K_mit_usb	25e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.0001
i1		.		0	0	10	0.5
i2		.		0	0	10	0.5
i3		.		0	0	10	0.5
i4		.		0	0	10	0.5
i5		.		0	0	10	0.5
i6		.		0	0	10	0.5
i7		.		0	0	10	0.5
i8		.		0	0	10	0.5
i9		.		0	0	10	0.5
i10		.		0	0	10	0.5
i11		.		0	0	10	0.5
i12		.		0	0	10	0.5
i13		.		0	0	10	0.5
i14		.		0	0	10	0.5
i15		.		0	0	10	0.5
i16		.		0	0	10	0.5
i17		.		0	0	10	0.5
i18		.		0	0	10	0.5
i19		.		0	0	10	0.5
i20		.		0	0	10	0.5

*makeproto /library/node

*start_cell /library/rnode
*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01
rnode		none		0	0	10	0.5	Na_mit_usb	15e3	K_mit_usb	25e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.0001
i1		.		0	0	-10	0.5
i2		.		0	0	-10	0.5
i3		.		0	0	-10	0.5
i4		.		0	0	-10	0.5
i5		.		0	0	-10	0.5
i6		.		0	0	-10	0.5
i7		.		0	0	-10	0.5
i8		.		0	0	-10	0.5
i9		.		0	0	-10	0.5
i10		.		0	0	-10	0.5
i11		.		0	0	-10	0.5
i12		.		0	0	-10	0.5
i13		.		0	0	-10	0.5
i14		.		0	0	-10	0.5
i15		.		0	0	-10	0.5
i16		.		0	0	-10	0.5
i17		.		0	0	-10	0.5
i18		.		0	0	-10	0.5
i19		.		0	0	-10	0.5
i20		.		0	0	-10	0.5

*makeproto /library/rnode

*start_cell /library/snode
*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01
snode		none		0	0	10	0.5	Na_mit_usb	15e3	K_mit_usb	25e3
*set_global	RM	200.0
*set_global	RA	0.5
*set_global	CM	0.0001
i1		.		0	10	0	0.5
i2		.		0	10	0	0.5
i3		.		0	10	0	0.5
i4		.		0	10	0	0.5
i5		.		0	10	0	0.5
i6		.		0	10	0	0.5
i7		.		0	10	0	0.5
i8		.		0	10	0	0.5
i9		.		0	10	0	0.5
i10		.		0	10	0	0.5
i11		.		0	10	0	0.5
i12		.		0	10	0	0.5
i13		.		0	10	0	0.5
i14		.		0	10	0	0.5
i15		.		0	10	0	0.5
i16		.		0	10	0	0.5
i17		.		0	10	0	0.5
i18		.		0	10	0	0.5
i19		.		0	10	0	0.5
i20		.		0	10	0	0.5

*makeproto /library/snode

*set_global	RM	2.0
*set_global	RA	0.5
*set_global	CM	0.01

*start_cell
soma		none		0	0	28	19	Na_mit_usb	2e3	K_mit_usb	4e3
*compt /library/node
n1		.		0	0	10	0.5
n2		n1/i20		0	0	10	0.5
n3		n2/i20		0	0	10	0.5
n4		n3/i20		0	0	10	0.5
n5		n4/i20		0	0	10	0.5
n6		n5/i20		0	0	10	0.5
n7		n6/i20		0	0	10	0.5
n8		n7/i20		0	0	10	0.5
n9		n8/i20		0	0	10	0.5
*compt /library/snode
n10		n9/i20		0	10	0	0.5
*compt /library/rnode
n11		n10/i20		0	0	-10	0.5
n12		n11/i20		0	0	-10	0.5
n13		n12/i20		0	0	-10	0.5
n14		n13/i20		0	0	-10	0.5
n15		n14/i20		0	0	-10	0.5
n16		n15/i20		0	0	-10	0.5
n17		n16/i20		0	0	-10	0.5
n18		n17/i20		0	0	-10	0.5
n19		n18/i20		0	0	-10	0.5
*compt /library/snode
n20		n19/i20		0	10	0	0.5
*compt /library/node
n21		n20/i20		0	0	10	0.5
n22		n21/i20		0	0	10	0.5
n23		n22/i20		0	0	10	0.5
n24		n23/i20		0	0	10	0.5
n25		n24/i20		0	0	10	0.5
n26		n25/i20		0	0	10	0.5
n27		n26/i20		0	0	10	0.5
n28		n27/i20		0	0	10	0.5
n29		n28/i20		0	0	10	0.5
*compt /library/snode
n30		n29/i20		0	10	0	0.5
*compt /library/rnode
n31		n30/i20		0	0	-10	0.5
n32		n31/i20		0	0	-10	0.5
n33		n32/i20		0	0	-10	0.5
n34		n33/i20		0	0	-10	0.5
n35		n34/i20		0	0	-10	0.5
n36		n35/i20		0	0	-10	0.5
n37		n36/i20		0	0	-10	0.5
n38		n37/i20		0	0	-10	0.5
n39		n38/i20		0	0	-10	0.5
*compt /library/snode
n40		n39/i20		0	10	0	0.5
*compt /library/node
n41		n40/i20		0	0	10	0.5
n42		n41/i20		0	0	10	0.5
n43		n42/i20		0	0	10	0.5
n44		n43/i20		0	0	10	0.5
n45		n44/i20		0	0	10	0.5
n46		n45/i20		0	0	10	0.5
n47		n46/i20		0	0	10	0.5
n48		n47/i20		0	0	10	0.5
n49		n48/i20		0	0	10	0.5
*compt /library/snode
n50		n49/i20		0	10	0	0.5
*compt /library/rnode
n51		n50/i20		0	0	-10	0.5
n52		n51/i20		0	0	-10	0.5
n53		n52/i20		0	0	-10	0.5
n54		n53/i20		0	0	-10	0.5
n55		n54/i20		0	0	-10	0.5
n56		n55/i20		0	0	-10	0.5
n57		n56/i20		0	0	-10	0.5
n58		n57/i20		0	0	-10	0.5
n59		n58/i20		0	0	-10	0.5
*compt /library/snode
n60		n59/i20		0	10	0	0.5
*compt /library/node
n61		n60/i20		0	0	10	0.5
n62		n61/i20		0	0	10	0.5
n63		n62/i20		0	0	10	0.5
n64		n63/i20		0	0	10	0.5
n65		n64/i20		0	0	10	0.5
n66		n65/i20		0	0	10	0.5
n67		n66/i20		0	0	10	0.5
n68		n67/i20		0	0	10	0.5
n69		n68/i20		0	0	10	0.5
*compt /library/snode
n70		n69/i20		0	10	0	0.5
*compt /library/rnode
n71		n70/i20		0	0	-10	0.5
n72		n71/i20		0	0	-10	0.5
n73		n72/i20		0	0	-10	0.5
n74		n73/i20		0	0	-10	0.5
n75		n74/i20		0	0	-10	0.5
n76		n75/i20		0	0	-10	0.5
n77		n76/i20		0	0	-10	0.5
n78		n77/i20		0	0	-10	0.5
n79		n78/i20		0	0	-10	0.5
*compt /library/snode
n80		n79/i20		0	10	0	0.5
*compt /library/node
n81		n80/i20		0	0	10	0.5
n82		n81/i20		0	0	10	0.5
n83		n82/i20		0	0	10	0.5
n84		n83/i20		0	0	10	0.5
n85		n84/i20		0	0	10	0.5
n86		n85/i20		0	0	10	0.5
n87		n86/i20		0	0	10	0.5
n88		n87/i20		0	0	10	0.5
n89		n88/i20		0	0	10	0.5
*compt /library/snode
n90		n89/i20		0	10	0	0.5
*compt /library/rnode
n91		n90/i20		0	0	-10	0.5
n92		n91/i20		0	0	-10	0.5
n93		n92/i20		0	0	-10	0.5
n94		n93/i20		0	0	-10	0.5
n95		n94/i20		0	0	-10	0.5
n96		n95/i20		0	0	-10	0.5
n97		n96/i20		0	0	-10	0.5
n98		n97/i20		0	0	-10	0.5
n99		n98/i20		0	0	-10	0.5