// genesis
*cartesian
*relative
*asymmetric

*set_global RM 0.33333	//ohm*m^2
*set_global RA 3000.0	//ohm*m
*set_global CM 0.01     //F/m^2
*set_global EREST_ACT	-0.059387	// volts

// The format for each compartment parameter line is :
// name  parent  x       y       z       d       ch      dens ...
// For channels, "dens" =  maximum conductance per unit area of compartment

squid none	     500  0  0  500	Na 1200 K 360
