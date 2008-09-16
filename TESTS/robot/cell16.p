// genesis
// cell parameter file for the 1991 Traub CA1 hippocampal cell
// "phi" parameter reduced by e-3
*cartesian
*relative
*symmetric

*set_global RM 1.0	//ohm*m^2
*set_global RA 1.0	//ohm*m
*set_global CM 0.03     //F/m^2
*set_global EREST_ACT	-0.06	// volts

// The format for each compartment parameter line is :
// name  parent  x       y       z       d       ch      dens ...
// For channels, "dens" =  maximum conductance per unit area of compartment

/* NOTE: The format of the cell descriptor files does not allow for
   continuation to another line.  The following long line lists the
   soma channels with their "density" parameters.

   Ca_conc	-17.402e12
	Not really a channel, but a "Ca_concen" object.  Normally, the B 
	field is set to "dens"/compt_volume (m^3), but the negative sign
	causes the absolute value to be used with no scaling by volume.
   Na		300  
   Ca		 40
   K_DR		150
   K_AHP	  8
   K_C		100
   K_A		 50
*/

// The compartment numbering corresponds to that in the paper, with soma = 9

apical_19 none	     -120  0  0  4.0

apical_18_12	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_18_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
spine_neck_18_12	apical_18_12	0   0.5  0  0.1
spine_head_18_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_11	apical_18_11	0   0.5  0  0.1
spine_head_18_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_10	apical_18_10	0   0.5  0  0.1
spine_head_18_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_9	apical_18_9	0   0.5  0  0.1
spine_head_18_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_8	apical_18_8	0   0.5  0  0.1
spine_head_18_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_7	apical_18_7	0   0.5  0  0.1
spine_head_18_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_6	apical_18_6	0   0.5  0  0.1
spine_head_18_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_5	apical_18_5	0   0.5  0  0.1
spine_head_18_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_4	apical_18_4	0   0.5  0  0.1
spine_head_18_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_3	apical_18_3	0   0.5  0  0.1
spine_head_18_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_2	apical_18_2	0   0.5  0  0.1
spine_head_18_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_18_1	apical_18_1	0   0.5  0  0.1
spine_head_18_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_17_12	apical_18_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
apical_17_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 15 K_DR 50 K_A 100
spine_neck_17_12	apical_17_12	0   0.5  0  0.1
spine_head_17_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_11	apical_17_11	0   0.5  0  0.1
spine_head_17_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_10	apical_17_10	0   0.5  0  0.1
spine_head_17_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_9	apical_17_9	0   0.5  0  0.1
spine_head_17_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_8	apical_17_8	0   0.5  0  0.1
spine_head_17_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_7	apical_17_7	0   0.5  0  0.1
spine_head_17_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_6	apical_17_6	0   0.5  0  0.1
spine_head_17_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_5	apical_17_5	0   0.5  0  0.1
spine_head_17_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_4	apical_17_4	0   0.5  0  0.1
spine_head_17_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_3	apical_17_3	0   0.5  0  0.1
spine_head_17_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_2	apical_17_2	0   0.5  0  0.1
spine_head_17_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_17_1	apical_17_1	0   0.5  0  0.1
spine_head_17_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60

apical_16_12	apical_17_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_16_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
spine_neck_16_12	apical_16_12	0   0.5  0  0.1
spine_head_16_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_11	apical_16_11	0   0.5  0  0.1
spine_head_16_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_10	apical_16_10	0   0.5  0  0.1
spine_head_16_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_9	apical_16_9	0   0.5  0  0.1
spine_head_16_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_8	apical_16_8	0   0.5  0  0.1
spine_head_16_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_7	apical_16_7	0   0.5  0  0.1
spine_head_16_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_6	apical_16_6	0   0.5  0  0.1
spine_head_16_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_5	apical_16_5	0   0.5  0  0.1
spine_head_16_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_4	apical_16_4	0   0.5  0  0.1
spine_head_16_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_3	apical_16_3	0   0.5  0  0.1
spine_head_16_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_2	apical_16_2	0   0.5  0  0.1
spine_head_16_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_16_1	apical_16_1	0   0.5  0  0.1
spine_head_16_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_15_12	apical_16_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_15_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
spine_neck_15_12	apical_15_12	0   0.5  0  0.1
spine_head_15_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_11	apical_15_11	0   0.5  0  0.1
spine_head_15_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_10	apical_15_10	0   0.5  0  0.1
spine_head_15_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_9	apical_15_9	0   0.5  0  0.1
spine_head_15_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_8	apical_15_8	0   0.5  0  0.1
spine_head_15_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_7	apical_15_7	0   0.5  0  0.1
spine_head_15_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_6	apical_15_6	0   0.5  0  0.1
spine_head_15_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_5	apical_15_5	0   0.5  0  0.1
spine_head_15_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_4	apical_15_4	0   0.5  0  0.1
spine_head_15_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_3	apical_15_3	0   0.5  0  0.1
spine_head_15_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_2	apical_15_2	0   0.5  0  0.1
spine_head_15_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_15_1	apical_15_1	0   0.5  0  0.1
spine_head_15_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_14_12	apical_15_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
apical_14_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 20 K_DR 50 K_A 100
spine_neck_14_12	apical_14_12	0   0.5  0  0.1
spine_head_14_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_11	apical_14_11	0   0.5  0  0.1
spine_head_14_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_10	apical_14_10	0   0.5  0  0.1
spine_head_14_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_9	apical_14_9	0   0.5  0  0.1
spine_head_14_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_8	apical_14_8	0   0.5  0  0.1
spine_head_14_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_7	apical_14_7	0   0.5  0  0.1
spine_head_14_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_6	apical_14_6	0   0.5  0  0.1
spine_head_14_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_5	apical_14_5	0   0.5  0  0.1
spine_head_14_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_4	apical_14_4	0   0.5  0  0.1
spine_head_14_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_3	apical_14_3	0   0.5  0  0.1
spine_head_14_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_2	apical_14_2	0   0.5  0  0.1
spine_head_14_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_1	apical_14_1	0   0.5  0  0.1
spine_head_14_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_13_12	apical_14_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
apical_13_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 25 K_DR 50 K_A 100
spine_neck_13_12	apical_13_12	0   0.5  0  0.1
spine_head_13_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_11	apical_13_11	0   0.5  0  0.1
spine_head_13_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_10	apical_13_10	0   0.5  0  0.1
spine_head_13_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_9	apical_13_9	0   0.5  0  0.1
spine_head_13_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_8	apical_13_8	0   0.5  0  0.1
spine_head_13_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_7	apical_13_7	0   0.5  0  0.1
spine_head_13_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_6	apical_13_6	0   0.5  0  0.1
spine_head_13_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_5	apical_13_5	0   0.5  0  0.1
spine_head_13_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_4	apical_13_4	0   0.5  0  0.1
spine_head_13_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_3	apical_13_3	0   0.5  0  0.1
spine_head_13_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_2	apical_13_2	0   0.5  0  0.1
spine_head_13_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_13_1	apical_13_1	0   0.5  0  0.1
spine_head_13_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_12_12	apical_13_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
apical_12_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 30 K_DR 50 K_A 100
spine_neck_12_12	apical_12_12	0   0.5  0  0.1
spine_head_12_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_11	apical_12_11	0   0.5  0  0.1
spine_head_12_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_10	apical_12_10	0   0.5  0  0.1
spine_head_12_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_9	apical_12_9	0   0.5  0  0.1
spine_head_12_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_8	apical_12_8	0   0.5  0  0.1
spine_head_12_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_7	apical_12_7	0   0.5  0  0.1
spine_head_12_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_6	apical_12_6	0   0.5  0  0.1
spine_head_12_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_5	apical_12_5	0   0.5  0  0.1
spine_head_12_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_4	apical_12_4	0   0.5  0  0.1
spine_head_12_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_3	apical_12_3	0   0.5  0  0.1
spine_head_12_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_2	apical_12_2	0   0.5  0  0.1
spine_head_12_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_12_1	apical_12_1	0   0.5  0  0.1
spine_head_12_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60


apical_11_12	apical_12_1  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_11  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_10  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_9  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_8  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_7  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_6  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_5  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_4  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_3  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_2  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
apical_11_1  	.  -10   0  0  4.0 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50 Na 40 K_DR 50 K_A 100
spine_neck_11_12	apical_11_12	0   0.5  0  0.1
spine_head_11_12	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_11	apical_11_11	0   0.5  0  0.1
spine_head_11_11	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_10	apical_11_10	0   0.5  0  0.1
spine_head_11_10	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_9	apical_11_9	0   0.5  0  0.1
spine_head_11_9	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_8	apical_11_8	0   0.5  0  0.1
spine_head_11_8	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_7	apical_11_7	0   0.5  0  0.1
spine_head_11_7	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_6	apical_11_6	0   0.5  0  0.1
spine_head_11_6	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_5	apical_11_5	0   0.5  0  0.1
spine_head_11_5	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_4	apical_11_4	0   0.5  0  0.1
spine_head_11_4	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_3	apical_11_3	0   0.5  0  0.1
spine_head_11_3	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_2	apical_11_2	0   0.5  0  0.1
spine_head_11_2	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_11_1	apical_11_1	0   0.5  0  0.1
spine_head_11_1	.  		0   0.5  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60

apical_10 	apical_11_1  -120  0  0  4.0 Ca_conc -26.404e12 Na 150 Ca 80 K_DR 100 K_AHP 8 K_C 200

soma	apical_10    -125  0  0  8.46 Ca_conc -17.402e12 Na 300 Ca 40 K_DR 250 K_AHP 8 K_C 100 K_A 50
basal_8  soma	    -110  0  0  3.84 Ca_conc -34.53e12 Na 150 Ca 80 K_DR 100 K_AHP 8 K_C 200
basal_7  basal_8    -110  0  0  3.84 Ca_conc 8.96e-3 Ca 50 K_DR 50 K_AHP 8 K_C 50
basal_6  basal_7    -110  0  0  3.84 Ca_conc 8.96e-3 Na 200 Ca 120 K_DR 200 K_AHP 8 K_C 100
basal_5  basal_6    -110  0  0  3.84 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50
basal_4  basal_5    -110  0  0  3.84 Ca_conc 8.96e-3 Ca 70 K_AHP 8 K_C 50
basal_3  basal_4    -110  0  0  3.84 Ca_conc 8.96e-3 Ca 50 K_AHP 8 K_C 50
basal_2  basal_3    -110  0  0  3.84 Ca_conc 8.96e-3 Ca 50 K_AHP 8 K_C 50
basal_1  basal_2    -110  0  0  3.84
