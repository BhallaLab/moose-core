// genesis
// cell parameter file for the 1991 Traub CA3 hippocampal cell
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

/*
compt      parent       x   y  z    d   Ca  x0.25  Ca  x0.75   Co   x1.00        Co   x1.00        K_AHP    K_AHP    Na      K_DR
*/
apical_19  none       -120  0  0  5.78
apical_18  apical_19  -120  0  0  5.78  Ca_1 12.5  Ca_2 37.5   Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_17  apical_18  -120  0  0  5.78  Ca_1 12.5  Ca_2 37.5   Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_16  apical_17  -120  0  0  5.78  Ca_1 25.0  Ca_2 75.0   Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_15  apical_16  -120  0  0  5.78  Ca_1 25.0  Ca_2 75.0   Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_14  apical_15  -120  0  0  5.78  Ca_1 42.5  Ca_2 127.5  Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_13  apical_14  -120  0  0  5.78  Ca_1 42.5  Ca_2 127.5  Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_12  apical_13  -120  0  0  5.78  Ca_1 42.5  Ca_2 127.5  Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0  Na 200  K_DR 200
apical_11  apical_12  -120  0  0  5.78  Ca_1 12.5  Ca_2 37.5   Co_1 -5.941e12    Co_4 -5.941e12    K_1_1 0  K_4_1 0
apical_10  apical_11  -120  0  0  5.78  Ca_1 20.0  Ca_2 60.0   Co_1 -26.404e12   Co_4 -26.404e12   K_1_1 0  K_4_1 0  Na 150  K_DR 50

/*
compt      parent       x   y  z    d   Ca  x1.00  Ca  xxxxx   Co   x1.00        Co   xxxxx        K_AHP    K_AHP    Na      K_DR      K_A
*/
soma       apical_10  -125  0  0  8.46  Ca_1 40                Co_1 -17.402e12                     K_1_1 0  K_1_2 0  Na 300  K_DR 150  K_A 50
basal_8    soma       -110  0  0  4.84  Ca_1 80                Co_1 -34.53e12                      K_1_1 0  K_1_2 0  Na 150  K_DR 50
basal_7    basal_8    -110  0  0  4.84  Ca_1 50                Co_1 -7.769e12                      K_1_1 0  K_1_2 0
basal_6    basal_7    -110  0  0  4.84  Ca_1 120               Co_1 -7.769e12                      K_1_1 0  K_1_2 0  Na 200  K_DR 200
basal_5    basal_6    -110  0  0  4.84  Ca_1 120               Co_1 -7.769e12                      K_1_1 0  K_1_2 0
basal_4    basal_5    -110  0  0  4.84  Ca_1 120               Co_1 -7.769e12                      K_1_1 0  K_1_2 0
basal_3    basal_4    -110  0  0  4.84  Ca_1 50                Co_1 -7.769e12                      K_1_1 0  K_1_2 0
basal_2    basal_3    -110  0  0  4.84  Ca_1 50                Co_1 -7.769e12                      K_1_1 0  K_1_2 0
basal_1    basal_2    -110  0  0  4.84
