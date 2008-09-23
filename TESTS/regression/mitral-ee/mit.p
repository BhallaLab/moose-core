//	PARAMETER FILE FOR NEURON 'block' : mitral cell model under
//		conditions of TEA and TTX block
//	Author : Upi Bhalla 
//	Mar 29 1991
//	Highly detailed model of mit cell with experimental averages for
//		cell geometry.


//	Format of file :
// x,y,z,dia are in microns, all other units are SI (Meter Kilogram Second Amp)
// In polar mode 'r' is in microns, theta and phi in degrees 
// Control line options start with a '*'
// The format for each compartment parameter line is :
//name	parent	r	theta	phi	d	ch	dens ...
//in polar mode, and in cartesian mode :
//name	parent	x	y	z	d	ch	dens ...

//		Coordinate mode
*cartesian
*relative

//		Specifying constants
*set_global	RM	10
*set_global	RA	2.0
*set_global	CM	0.01
*set_global	EREST_ACT	-0.065

soma	none	32	0	0	32	LCa3_mit_usb	40	K_mit_usb	28	KA_bsg_yka	58.7	Ca_mit_conc	5.2e-6	Kca_mit_usb	142	Na_mit_usb	1532	K2_mit_usb	1956	Rm	120e6


axon	soma	0	0	-10	10	LCa3_mit_usb	20	K_mit_usb	15.5	KA_bsg_yka	51.5	Ca_mit_conc	5.2e-6	Kca_mit_usb	88.7	K2_mit_usb	1541	Na_mit_usb	4681
axon[1]	.	0	0	-10	7	LCa3_mit_usb	20	K_mit_usb	15.5	KA_bsg_yka	51.5	Ca_mit_conc	5.2e-6	Kca_mit_usb	88.7	K2_mit_usb	1541	Na_mit_usb	4681
axon[2]	.	0	0	-10	5	LCa3_mit_usb	20	K_mit_usb	15.5	KA_bsg_yka	51.5	Ca_mit_conc	5.2e-6	Kca_mit_usb	88.7	K2_mit_usb	1156	Na_mit_usb	4681
axon[3]	.	0	0	-10	3	LCa3_mit_usb	20	K_mit_usb	15.5	KA_bsg_yka	51.5	Ca_mit_conc	5.2e-6	Kca_mit_usb	88.7	K2_mit_usb	1156	Na_mit_usb	4681
axon[4]	.	0	0	-10	2	LCa3_mit_usb	20	K_mit_usb	15.5	KA_bsg_yka	51.5	Ca_mit_conc	5.2e-6	Kca_mit_usb	88.7	K2_mit_usb	1156	Na_mit_usb	4681
axon[5]	.	0	0	-10	1.5	K2_mit_usb	771	Na_mit_usb	3511
axon[6]	.	0	0	-20	1.5	K2_mit_usb	771	Na_mit_usb	3511
axon[7]	.	0	0	-50	1.5	K2_mit_usb	771	Na_mit_usb	2340
axon[8]	.	0	0	-50	1.0	K2_mit_usb	771	Na_mit_usb	1170
axon[9]	.	0	0	-90	0.9	K2_mit_usb	385	Na_mit_usb	586
axon[10] .	0	0	-90	0.9	K2_mit_usb	77	Na_mit_usb	234
axon[11] .	0	0	-90	0.9	K2_mit_usb	77	Na_mit_usb	117
axon[12] .	0	0	-90	0.9	K2_mit_usb	77	Na_mit_usb	117

primary_dend	soma	0	0	95	7.9	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4

primary_dend[1]	.	0	0	95	7.3	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4
primary_dend[2]	.	0	0	95	6.6	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4
primary_dend[3]	.	0	0	95	6	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4
primary_dend[4]	.	0	0	95	5.3	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4
primary_dend[5]	.	0	0	95	4.7	LCa3_mit_usb	22	K_mit_usb	17.4	K2_mit_usb	12.3	Na_mit_usb	13.4

*polar
glom1		primary_dend[5]	20	0	30	3.5	LCa3_mit_usb	95	K_mit_usb	28

glom11		glom1		20	60	15	2.8	LCa3_mit_usb	95	K_mit_usb	28

glom111		glom11		30	90	45	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom1111	glom111		25	120	30	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom11111	glom1111	25	150	45	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom111111	glom11111	30	210	90	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom111112	glom11111	25	90	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom1111111	glom111112	20	45	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom1111112	glom111112	20	120	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom11112	glom1111	30	60	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom1112	glom111		30	30	120	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom112		glom11		30	30	30	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom1121	glom112		25	45	15	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom11211	glom1121	25	60	75	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom112111	glom11211	30	90	150	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom112112	glom11211	25	30	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom1121121	glom112112	20	75	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom11211211	glom1121121	20	120	15	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom11211212	glom1121121	20	210	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom112112121	glom11211212	10	255	60	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom112112122	glom11211212	10	135	45	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom1121122	glom112112	20	-45	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom11212	glom1121	25	-30	90	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom1122	glom112		30	-60	120	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom12		glom1		20	-60	15	2.8	LCa3_mit_usb	95	K_mit_usb	28

glom121		glom12		30	0	15	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom1211	glom121		25	30	0	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom12111	glom1211	30	60	15	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom121111	glom12111	30	120	40	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom121112	glom12111	30	-180	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom1211111	glom121112	20	-45	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom1211112	glom121112	25	30	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom12112	glom1211	25	-30	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom1212	glom121		30	-60	75	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom122		glom12		30	-60	30	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom1221	glom122		25	-45	15	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom12211	glom1221	25	-30	75	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom122111	glom12211	25	0	150	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom122112	glom12211	25	-60	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom1221121	glom122112	20	-15	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom12211211	glom1221121	20	30	15	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom12211212	glom1221121	20	120	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom122112121	glom12211212	10	165	60	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom122112122	glom12211212	10	45	45	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom1221122	glom122112	20	-135	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom12212	glom1221	30	-120	90	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom1222	glom122		30	-150	120	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom2		primary_dend[5]	20	180	30	3.5	LCa3_mit_usb	95	K_mit_usb	28

glom21		glom2		20	240	15	2.8	LCa3_mit_usb	95	K_mit_usb	28

glom211		glom21		30	-90	45	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom2111	glom211		25	300	30	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom21111	glom2111	25	330	45	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom211111	glom21111	30	30	90	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom211112	glom21111	25	-90	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom2111111	glom211112	20	215	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom2111112	glom211112	20	-60	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom21112	glom2111	30	240	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom2112	glom211		30	210	120	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom212		glom21		30	210	30	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom2121	glom212		25	215	15	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom21211	glom2121	25	240	75	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom212111	glom21211	30	-90	150	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom212112	glom21211	25	210	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom2121121	glom212112	20	255	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom21211211	glom2121121	20	-60	15	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom21211212	glom2121121	20	30	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom212112121	glom21211212	10	75	60	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom212112122	glom21211212	10	-45	45	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom2121122	glom212112	20	135	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom21212	glom2121	25	150	90	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom2122	glom212		30	120	120	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom22		glom2		20	120	15	2.8	LCa3_mit_usb	95	K_mit_usb	28

glom221		glom22		30	180	15	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom2211	glom221		25	210	0	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom22111	glom2211	30	240	15	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom221111	glom22111	30	-60	40	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom221112	glom22111	30	0	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom2211111	glom221112	20	135	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom2211112	glom221112	25	210	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom22112	glom2211	25	150	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom2212	glom221		30	120	75	0.7	LCa3_mit_usb	95	K_mit_usb	28

glom222		glom22		30	120	30	2.0	LCa3_mit_usb	95	K_mit_usb	28
glom2221	glom222		25	135	15	1.5	LCa3_mit_usb	95	K_mit_usb	28
glom22211	glom2221	30	150	75	1.0	LCa3_mit_usb	95	K_mit_usb	28
glom222111	glom22211	25	180	150	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom222112	glom22211	25	120	30	0.9	LCa3_mit_usb	95	K_mit_usb	28
glom2221121	glom222112	20	165	30	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom22211211	glom2221121	20	210	15	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom22211212	glom2221121	20	-60	30	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom222112121	glom22211212	10	-15	60	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom222112122	glom22211212	10	225	45	0.5	LCa3_mit_usb	95	K_mit_usb	28
glom2221122	glom222112	20	45	150	0.8	LCa3_mit_usb	95	K_mit_usb	28
glom22212	glom2221	30	60	90	0.7	LCa3_mit_usb	95	K_mit_usb	28
glom2222	glom222		30	30	120	0.7	LCa3_mit_usb	95	K_mit_usb	28


sec_dend1	soma		30	0	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend1[1]	.		40	0	60	5.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend1[2]	.		12	0	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend11	sec_dend1[2]	80	30	75	3.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend11[1]	.		80	30	75	2.9	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend11[2]	.		80	30	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend11[3]	.		80	30	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend11[4]	.		50	30	75	3.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend111	sec_dend11[4]	100	60	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend111[1]	.		100	75	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend111[2]	.		100	60	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend111[3]	.		100	45	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend111[4]	.		60	60	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend112	sec_dend11[4]	100	0	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend112[1]	.		100	-15	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend112[2]	.		100	-30	75	1.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend112[3]	.		100	-30	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend112[4]	.		60	-30	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend1121	sec_dend112[4]	80	-15	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend1121[1]	.		80	-15	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend1121[2]	.		80	0	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend1121[3]	.		80	0	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend1121[4]	.		72	0	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend1122	sec_dend112[4]	90	-45	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend1122[1]	.		90	-45	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend1122[2]	.		90	-45	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend1122[3]	.		90	-45	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend1122[4]	.		32	-45	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend11221	sec_dend1122[4]	90	15	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend11221[1] .		90	15	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend11221[2] .		90	15	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend11221[3] .		90	15	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend11221[4] .		40	15	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend11222	sec_dend1122[4]	90	-45	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend11222[1] .		90	-45	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend11222[2] .		90	-75	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend11222[3] .		90	-75	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend11222[4] .		40	-75	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend12	sec_dend1[2]	80	-30	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend12[1]	.		80	-30	75	2.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend12[2]	.		80	-30	75	1.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend12[3]	.		80	-30	75	1.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend12[4]	.		50	-30	75	1.5	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330

sec_dend2	soma		30	90	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend2[1]	.		40	90	60	5.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend2[2]	.		12	90	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend21	sec_dend2[2]	80	120	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend21[1]	.		80	120	75	2.9	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend21[2]	.		80	120	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend21[3]	.		80	105	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend21[4]	.		50	105	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330

sec_dend211	sec_dend21[4]	100	150	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend211[1]	.		100	165	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend211[2]	.		100	180	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend211[3]	.		100	180	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend211[4]	.		60	180	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend212	sec_dend21[4]	100	105	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend212[1]	.		100	105	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend212[2]	.		100	90	75	1.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend212[3]	.		100	90	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend212[4]	.		60	90	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend2121	sec_dend212[4]	80	105	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend2121[1]	.		80	135	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend2121[2]	.		80	135	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend2121[3]	.		80	135	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend2121[4]	.		72	135	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend2122	sec_dend212[4]	90	90	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend2122[1]	.		90	60	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend2122[2]	.		90	30	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend2122[3]	.		90	30	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend2122[4]	.		32	30	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend21221	sec_dend2122[4]	90	60	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend21221[1] .		90	90	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend21221[2] .		90	120	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend21221[3] .		90	120	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend21221[4] .		40	120	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend21222	sec_dend2122[4]	90	30	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend21222[1] .		90	30	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend21222[2] .		90	-15	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend21222[3] .		90	-15	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend21222[4] .		40	-15	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend22	sec_dend2[2]	80	90	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend22[1]	.		80	75	75	2.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend22[2]	.		80	75	75	1.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend22[3]	.		80	75	75	1.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend22[4]	.		50	75	75	1.5	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330

sec_dend3	soma		30	180	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend3[1]	.		40	180	60	5.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend3[2]	.		12	180	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend31	sec_dend3[2]	80	180	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend31[1]	.		80	210	75	2.9	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend31[2]	.		80	210	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend31[3]	.		80	210	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend31[4]	.		50	210	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend311	sec_dend31[4]	100	230	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend311[1]	.		100	230	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend311[2]	.		100	270	75	1.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311[3]	.		100	270	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend311[4]	.		60	270	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend3111	sec_dend311[4]	80	285	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend3111[1]	.		80	300	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend3111[2]	.		80	300	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend3111[3]	.		80	300	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend3111[4]	.		72	300	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend3112	sec_dend311[4]	90	240	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend3112[1]	.		90	210	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend3112[2]	.		90	210	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend3112[3]	.		90	210	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend3112[4]	.		32	210	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend31121	sec_dend3112[4]	90	210	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend31121[1] .		90	240	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend31121[2] .		90	240	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend31121[3] .		90	240	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend31121[4] .		40	240	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend311211	sec_dend31121[4] 74	240	75	1.1	K2_mit_usb	128	Na_mit_usb	122
sec_dend311211[1] .		70	240	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend311211[2] .		70	265	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311211[3] .		70	265	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311211[4] .		70	265	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311212	sec_dend31121[4] 74	210	75	1.1	K2_mit_usb	128	Na_mit_usb	122
sec_dend311212[1] .		70	165	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend311212[2] .		70	165	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311212[3] .		70	165	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend311212[4] .		70	165	75	0.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend31122	sec_dend3112[4]	90	150	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend31122[1] .		90	135	75	1.3	K2_mit_usb	128	Na_mit_usb	122
sec_dend31122[2] .		90	135	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend31122[3] .		90	135	75	1.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend31122[4] .		40	135	75	1.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend312	sec_dend31[4]	100	195	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend312[1]	.		100	150	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend312[2]	.		100	150	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend312[3]	.		100	150	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend312[4]	.		60	150	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend32	sec_dend3[2]	80	150	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend32[1]	.		80	150	75	2.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend32[2]	.		80	150	75	1.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend32[3]	.		80	150	75	1.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend32[4]	.		50	150	75	1.5	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330

sec_dend4	soma		30	270	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend4[1]	.		40	270	60	5.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend4[2]	.		12	270	60	5.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend41	sec_dend4[2]	80	-45	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend41[1]	.		80	-45	75	2.9	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend41[2]	.		80	-45	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend41[3]	.		80	-45	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend41[4]	.		50	-45	75	3.1	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend411	sec_dend41[4]	100	-45	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend411[1]	.		100	-45	75	2.0	K2_mit_usb	128	Na_mit_usb	122
sec_dend411[2]	.		100	-30	75	1.9	K2_mit_usb	128	Na_mit_usb	122
sec_dend411[3]	.		100	-30	75	1.8	K2_mit_usb	128	Na_mit_usb	122
sec_dend411[4]	.		60	-30	75	2.2	K2_mit_usb	128	Na_mit_usb	122
sec_dend4111	sec_dend411[4]	80	0	75	1.6	K2_mit_usb	128	Na_mit_usb	122
sec_dend4111[1]	.		80	15	75	1.5	K2_mit_usb	128	Na_mit_usb	122
sec_dend4111[2]	.		80	30	75	1.4	K2_mit_usb	128	Na_mit_usb	122
sec_dend4111[3]	.		80	30	75	1.3	K2_mit_usb	122	Na_mit_usb	122
sec_dend4111[4]	.		72	30	75	1.0	K2_mit_usb	122	Na_mit_usb	122
sec_dend4112	sec_dend411[4]	80	-45	75	1.6	K2_mit_usb	122	Na_mit_usb	122
sec_dend4112[1]	.		80	-45	75	1.5	K2_mit_usb	122	Na_mit_usb	122
sec_dend4112[2]	.		80	-45	75	1.4	K2_mit_usb	122	Na_mit_usb	122
sec_dend4112[3]	.		80	-45	75	1.3	K2_mit_usb	122	Na_mit_usb	122
sec_dend4112[4]	.		72	-45	75	1.0	K2_mit_usb	122	Na_mit_usb	122
sec_dend412	sec_dend41[4]	100	300	75	2.0	K2_mit_usb	122	Na_mit_usb	122
sec_dend412[1]	.		100	270	75	1.8	K2_mit_usb	122	Na_mit_usb	122
sec_dend412[2]	.		100	270	75	1.6	K2_mit_usb	122	Na_mit_usb	122
sec_dend412[3]	.		100	270	75	1.5	K2_mit_usb	122	Na_mit_usb	122
sec_dend412[4]	.		60	270	75	1.4	K2_mit_usb	122	Na_mit_usb	122
sec_dend42	sec_dend4[2]	80	-90	75	2.7	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend42[1]	.		80	-75	75	2.2	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend42[2]	.		80	-75	75	1.8	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend42[3]	.		80	-75	75	1.6	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
sec_dend42[4]	.		50	-75	75	1.5	LCa3_mit_usb	4	K_mit_usb	8.5	K2_mit_usb	226	Na_mit_usb	330
