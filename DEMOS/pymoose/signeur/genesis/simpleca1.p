// genesis
// cell parameter file for the 1991 Traub CA1 hippocampal cell
// "phi" parameter reduced by e-3
*cartesian
*relative
*asymmetric

*set_compt_param RM 1.0	//ohm*m^2
*set_compt_param RA 1.0	//ohm*m
*set_compt_param CM 0.03     //F/m^2
//*set_compt_param EREST_ACT	-0.06	// volts
//*set_global     ELEAK	-0.06
*set_compt_param ELEAK -0.06

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

soma    	none	0  	125  0  8.46	Ca_conc -17.402e12 Ca 40 K_AHP 8 K_C 100 	Na 300  K_DR 250  K_A 5
apical_1 	soma	0  120  0  4.0 		Ca_conc -26.404e12 	Ca 80 K_AHP 8 K_C 200 	Na 150 K_DR 100 
apical_2	.	0   120  0  3 		Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 40 K_DR 50 K_A 100
apical_3	.	0   120  0  3 		Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 30 K_DR 50 K_A 100
apical_4  	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 25 K_DR 50 K_A 100
apical_5  	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 20 K_DR 50 K_A 100
apical_6	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 20 K_DR 50 K_A 100
apical_7	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 20 K_DR 50 K_A 100
apical_8	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 15 K_DR 50 K_A 100
apical_9	.	0   120  0  2.6 	Ca_conc 8.96e-3 	Ca 70 K_AHP 8 K_C 50 	Na 15 K_DR 50 K_A 100
apical_10 	.	0  120   0  2.6

spine_neck_14_1	apical_6	0.5  0  0  0.1
//spine_head_14_1	.  		0.5  0  0  0.5	Ca 60	Ca_conc 8.96e-3
spine_head_14_1	.  		0.5  0  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
spine_neck_14_2	apical_9	0.5  0  0  0.1
//spine_head_14_2	.  		0.5  0  0  0.5	Ca 60	Ca_conc 8.96e-3
spine_head_14_2	.  		0.5  0  0  0.5 NMDA_Ca_conc 8.96e-3 glu 700 Ca_NMDA 20 NMDA 200 Ca 60
