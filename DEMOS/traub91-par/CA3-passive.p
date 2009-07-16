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
			continuation	to	
			soma	channels	

			Ca_conc	-17.402e12	
	Not	really	
	field	is	
	causes	the	
			Na		300	
			Ca			40	
			K_DR		150	
			K_AHP			8	
			K_C		100	
			K_A			50	
*/

// The compartment numbering corresponds to that in the paper, with soma = 9

apical_19	none						-120		0		0		5.78	
apical_18	apical_19		-120		0		0		5.78	
apical_17	apical_18		-120		0		0		5.78	
apical_16	apical_17		-120		0		0		5.78	
apical_15	apical_16		-120		0		0		5.78	
apical_14	apical_15		-120		0		0		5.78	
apical_13	apical_14		-120		0		0		5.78	
apical_12	apical_13		-120		0		0		5.78	
apical_11	apical_12		-120		0		0		5.78	
apical_10	apical_11		-120		0		0		5.78	

soma	apical_10				-125		0		0		8.46

basal_8		soma					-110		0		0		4.84	
basal_7		basal_8				-110		0		0		4.84	
basal_6		basal_7				-110		0		0		4.84	
basal_5		basal_6				-110		0		0		4.84	
basal_4		basal_5				-110		0		0		4.84	
basal_3		basal_4				-110		0		0		4.84	
basal_2		basal_3				-110		0		0		4.84	
basal_1		basal_2				-110		0		0		4.84	
