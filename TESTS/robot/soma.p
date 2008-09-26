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


soma	none    -125  0  0  8.46 Ca_conc -17.402e12 Na 10000 Ca 40 K_DR 2500 K_AHP 20 K_C 200 K_A 50
