// genesis

/* FILE INFORMATION
** Some Ca channels for thje purkinje cell
** L Channel data from :
**	T. Hirano and S. Hagiwara Pflugers A 413(5) pp463-469, 1989
** 
** T Channel data from :
** \
    	Kaneda, Wakamori, Ito and Akaike J Neuroph 63(5), pp1046-1051 1990
** 
** Implemented by Eric De Schutter - January 1991
** Converted to NEUROKIT format by Upinder S. Bhalla. Feb 1991
** This file depends on functions and constants defined in defaults.g
*/

// CONSTANTS
// (I-current)
float ECa = 0.07
// (I-current)
float ENa = 0.045
// sq m
float SOMA_A = 1e-9


/* FILE INFORMATION
** Rat Na channel, cloned, in oocyte expression system.
** Data from :
** Stuhmer, Methfessel, Sakmann, Noda an Numa, Eur Biophys J 1987
**	14:131-138
**
** Expts carred out at 16 deg Celsius.
** 
** Implemented in tabchan format by Upinder S. Bhalla March 1991
** This file depends on functions and constants defined in defaults.g
*/

// CONSTANTS
// (I-current)
float ENa = 0.045
// sq m
float SOMA_A = 1e-9

//========================================================================
//                        Adjusted LCa channel
//========================================================================
function make_LCa3_mit_usb
	if (({exists LCa3_mit_usb}))
		return
	end
// (I-current)
float ECa = 0.07

    create tabchannel LCa3_mit_usb
    setfield LCa3_mit_usb Ek {ECa} Gbar {1200.0*SOMA_A} Ik 0 Gk 0  \
        Xpower 1 Ypower 1 Zpower 0

	setup_tabchan LCa3_mit_usb X 7500.0 0.0 1.0 -0.013 -0.007 1650.0 \
	     0.0 1.0 -0.014 0.004

	setup_tabchan LCa3_mit_usb Y 6.8 0.0 1.0 0.030 0.012 60.0 0.0  \
	    1.0 0.0 -0.011
end



/*********************************************************************
**                          I-Current (Na)
*********************************************************************/

function make_Na_rat_smsnn// Na current

	// (I-current)
	float ENa = 0.045
	float x, y, dx
	int i
    if (({exists Na_rat_smsnn}))
        return
    end

    create tabchannel Na_rat_smsnn
    setfield Na_rat_smsnn Ek {ENa} Gbar {1200.0*SOMA_A} Ik 0 Gk 0  \
        Xpower 3 Ypower 1 Zpower 0

	call Na_rat_smsnn TABCREATE X 30 -0.1 0.05

    // -0.100 Volts
    // -0.095 Volts
    // -0.090 Volts
    // -0.085 Volts
    // -0.080 Volts
    // -0.075 Volts
    // -0.070 Volts
    // -0.065 Volts
    // -0.060 Volts
    // -0.055 Volts
    // -0.050 Volts
    // -0.045 Volts
    // -0.040 Volts
    // -0.030
    // -0.020
    // -0.010
    // 0.0
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield Na_rat_smsnn X_A->table[0] 1.0e-4 X_A->table[1] 1.0e-4  \
        X_A->table[2] 1.2e-4 X_A->table[3] 1.45e-4 X_A->table[4] 1.67e-4 \
         X_A->table[5] 2.03e-4 X_A->table[6] 2.47e-4  \
        X_A->table[7] 3.20e-4 X_A->table[8] 3.63e-4  \
        X_A->table[9] 4.94e-4 X_A->table[10] 4.07e-4  \
        X_A->table[11] 4.00e-4 X_A->table[12] 3.56e-4  \
        X_A->table[13] 3.49e-4 X_A->table[14] 3.12e-4  \
        X_A->table[15] 2.83e-4 X_A->table[16] 2.62e-4  \
        X_A->table[17] 2.25e-4 X_A->table[18] 2.03e-4  \
        X_A->table[19] 1.74e-4 X_A->table[20] 1.67e-4  \
        X_A->table[21] 1.31e-4 X_A->table[22] 1.23e-4  \
        X_A->table[23] 1.16e-4 X_A->table[24] 1.02e-4  \
        X_A->table[25] 0.87e-4 X_A->table[26] 0.73e-4  \
        X_A->table[27] 0.80e-4 X_A->table[28] 0.80e-4  \
        X_A->table[29] 0.80e-4 X_A->table[30] 0.80e-4

	x = -0.1
	dx = 0.15/30.0

	for (i = 0; i <= 30; i = i + 1)
		y = 1.0/(1.0 + {exp {-(x + 0.041)/0.0086}})
		setfield Na_rat_smsnn X_B->table[{i}] {y}
		x = x + dx
	end
	tau_tweak_tabchan Na_rat_smsnn X
	setfield Na_rat_smsnn X_A->calc_mode 0 X_B->calc_mode 0
	call Na_rat_smsnn TABFILL X 3000 0


	call Na_rat_smsnn TABCREATE Y 30 -0.1 0.05
   // settab2const(Na_rat_smsnn,Y_A,0,10,6.4e-3)
	//-0.1 thru -0.05=>0.0

    // -0.100 Volts
    // -0.095 Volts
    // -0.090 Volts
    // -0.085 Volts
    // -0.080 Volts
    // -0.075 Volts
    // -0.070 Volts
    // -0.065 Volts
    // -0.060 Volts
    // -0.055 Volts
    // -0.050 Volts
    // -0.045 Volts
    // -0.040 Volts
    // -0.030
    // -0.020
    // -0.010
    // 0.0
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield Na_rat_smsnn Y_A->table[0] 0.9e-3 Y_A->table[1] 1.0e-3  \
        Y_A->table[2] 1.2e-3 Y_A->table[3] 1.45e-3 Y_A->table[4] 1.7e-3  \
        Y_A->table[5] 2.05e-3 Y_A->table[6] 2.55e-3 Y_A->table[7] 3.2e-3 \
         Y_A->table[8] 4.0e-3 Y_A->table[9] 5.0e-3  \
        Y_A->table[10] 6.49e-3 Y_A->table[11] 6.88e-3  \
        Y_A->table[12] 4.07e-3 Y_A->table[13] 2.71e-3  \
        Y_A->table[14] 2.03e-3 Y_A->table[15] 1.55e-3  \
        Y_A->table[16] 1.26e-3 Y_A->table[17] 1.07e-3  \
        Y_A->table[18] 0.87e-3 Y_A->table[19] 0.78e-3  \
        Y_A->table[20] 0.68e-3 Y_A->table[21] 0.63e-3  \
        Y_A->table[22] 0.58e-3 Y_A->table[23] 0.53e-3  \
        Y_A->table[24] 0.48e-3 Y_A->table[25] 0.48e-3  \
        Y_A->table[26] 0.48e-3 Y_A->table[27] 0.48e-3  \
        Y_A->table[28] 0.48e-3 Y_A->table[29] 0.43e-3  \
        Y_A->table[30] 0.39e-3

	x = -0.1
	dx = 0.15/30.0
	for (i = 0; i <= 30; i = i + 1)
		y = 1.0/(1.0 + {exp {(x + 0.064)/0.0102}})
		setfield Na_rat_smsnn Y_B->table[{i}] {y}
		x = x + dx
	end
	tau_tweak_tabchan Na_rat_smsnn Y
	setfield Na_rat_smsnn Y_A->calc_mode 0 Y_B->calc_mode 0
	call Na_rat_smsnn TABFILL Y 3000 0
end

function make_Na2_rat_smsnn
	if (({exists Na2_rat_smsnn}))
		return
	end
	float EK = -0.07

	if (({exists Na_rat_smsnn}))
		move Na_rat_smsnn Na2_rat_smsnn
		make_Na_rat_smsnn
	else
		make_Na_rat_smsnn
		move Na_rat_smsnn Na2_rat_smsnn
	end
	setfield Na2_rat_smsnn X_A->ox 0.01 X_B->ox 0.01 Y_A->ox 0.01  \
	    Y_B->ox 0.01
end

/********************************************************************
**            Transient outward K current
********************************************************************/

// CONSTANTS

float V_OFFSET = 0.0
float VKTAU_OFFSET = 0.0
float VKMINF_OFFSET = 0.02
float EK = -0.07

function make_KA_bsg_yka
	if (({exists KA_bsg_yka}))
		return
	end

    create tabchannel KA_bsg_yka
    setfield KA_bsg_yka Ek {EK} Gbar {1200.0*SOMA_A} Ik 0 Gk 0 Xpower 1  \
        Ypower 1 Zpower 0

	setup_tabchan_tau KA_bsg_yka X 1.38e-3 0.0 1.0 -1.0e3 1.0 1.0  \
	    0.0 1.0 {0.042 - V_OFFSET} -0.013

	setup_tabchan_tau KA_bsg_yka Y 0.150 0.0 1.0 -1.0e3 1.0 1.0 0.0  \
	    1.0 {0.110 - V_OFFSET} 0.018
end

/********************************************************************
**            Non-inactivating Muscarinic K current
********************************************************************/
function make_KM_bsg_yka
	if (({exists KM_bsg_yka}))
		return
	end

	int i
	float x, dx, y, b

    create tabchannel KM_bsg_yka
    setfield KM_bsg_yka Ek {EK} Gbar {1200.0*SOMA_A} Ik 0 Gk 0 Xpower 1  \
        Ypower 0 Zpower 0

	call KM_bsg_yka TABCREATE X 49 -0.1 0.1
	x = -0.1
	dx = 0.2/49.0

	for (i = 0; i <= 49; i = i + 1)
		y = 1.0/(3.3*({exp {(x + 0.035 - V_OFFSET)/0.04}}) + {exp {-(x + 0.035 - V_OFFSET)/0.02}})
		setfield KM_bsg_yka X_A->table[{i}] {y}

		y = 1.0/(1.0 + {exp {-(x + 0.035 - V_OFFSET)/0.01}})
		setfield KM_bsg_yka X_B->table[{i}] {y}
		x = x + dx
	end
	tau_tweak_tabchan KM_bsg_yka X
	/*
	for (i = 0 ; i <= 49 ; i = i + 1)
		y = get(KM_bsg_yka,X_A->table[{i}])
		b = get(KM_bsg_yka,X_B->table[{i}])
		set KM_bsg_yka X_A->table[{i}] {y+y}
		set KM_bsg_yka X_B->table[{i}] {b+y}
	end
	*/
	setfield KM_bsg_yka X_A->calc_mode 0 X_B->calc_mode 0
	call KM_bsg_yka TABFILL X 3000 0
end

/**********************************************************************
**                      Mitral K current
**  Heavily adapted from :
**	K current activation from Thompson, J. Physiol 265, 465 (1977)
**	(Tritonia (LPl	2 and LPl 3 cells)
** Inactivation from RW Aldrich, PA Getting, and SH Thompson, 
** J. Physiol, 291, 507 (1979)
**
**********************************************************************/
function make_K_mit_usb// K-current     

    if ({exists K_mit_usb})
        return
    end
	float EK = -0.07

    create tabchannel K_mit_usb
    setfield K_mit_usb Ek {EK} Gbar {1200*SOMA_A} Ik 0 Gk 0 Xpower 2  \
        Ypower 1 Zpower 0

    call K_mit_usb TABCREATE X 30 -0.100 0.050
    settab2const K_mit_usb X_A 0 12 0.0    //-0.1 thru -0.045=>0.0

    // -0.030
    // -0.020
    // -0.010
    // 0.0
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield K_mit_usb X_A->table[13] 0.00 X_A->table[14] 2.87  \
        X_A->table[15] 4.68 X_A->table[16] 7.46 X_A->table[17] 10.07  \
        X_A->table[18] 14.27 X_A->table[19] 17.87 X_A->table[20] 22.9  \
        X_A->table[21] 33.6 X_A->table[22] 49.3 X_A->table[23] 65.6  \
        X_A->table[24] 82.0 X_A->table[25] 110.0 X_A->table[26] 147.1  \
        X_A->table[27] 147.1 X_A->table[28] 147.1 X_A->table[29] 147.1  \
        X_A->table[30] 147.1

    // -0.100 Volts
    // -0.095 Volts
    // -0.090 Volts
    // -0.085 Volts
    // -0.080 Volts
    // -0.075 Volts
    // -0.070 Volts
    // -0.065 Volts
    // -0.060 Volts
    // -0.055 Volts
    // -0.050 Volts
    // -0.045 Volts
    // -0.040 Volts
    // -0.030
    // -0.020
    // -0.010
    // 0.00
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield K_mit_usb X_B->table[0] 36.0 X_B->table[1] 34.4  \
        X_B->table[2] 32.8 X_B->table[3] 31.2 X_B->table[4] 29.6  \
        X_B->table[5] 28.0 X_B->table[6] 26.3 X_B->table[7] 24.7  \
        X_B->table[8] 23.1 X_B->table[9] 21.5 X_B->table[10] 19.9  \
        X_B->table[11] 18.3 X_B->table[12] 16.6 X_B->table[13] 15.4  \
        X_B->table[14] 13.5 X_B->table[15] 13.2 X_B->table[16] 11.9  \
        X_B->table[17] 11.5 X_B->table[18] 10.75 X_B->table[19] 9.30  \
        X_B->table[20] 8.30 X_B->table[21] 6.00 X_B->table[22] 5.10  \
        X_B->table[23] 4.80 X_B->table[24] 3.20 X_B->table[25] 1.60  \
        X_B->table[26] 0.00 X_B->table[27] 0.00 X_B->table[28] 0.00  \
        X_B->table[29] 0.00 X_B->table[30] 0.00

		/* Setting the calc_mode to NO_INTERP for speed */
		setfield K_mit_usb X_A->calc_mode 0 X_B->calc_mode 0

		/* tweaking the tables for the tabchan calculation */
		tweak_tabchan K_mit_usb X

		/* Filling the tables using B-SPLINE interpolation */
		call K_mit_usb TABFILL X 3000 0


    call K_mit_usb TABCREATE Y 30 -0.100 0.050
    settab2const K_mit_usb Y_A 0 11 1.0    //-0.1 thru -0.035 => 1.0

    // -0.040	Volts
    // 
    // -0.030	Volts
    // -0.020
    // -0.010
    // 0.00
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield K_mit_usb Y_A->table[12] 1.00 Y_A->table[13] 0.97  \
        Y_A->table[14] 0.94 Y_A->table[15] 0.88 Y_A->table[16] 0.75  \
        Y_A->table[17] 0.61 Y_A->table[18] 0.43 Y_A->table[19] 0.305  \
        Y_A->table[20] 0.220 Y_A->table[21] 0.175 Y_A->table[22] 0.155  \
        Y_A->table[23] 0.143 Y_A->table[24] 0.138 Y_A->table[25] 0.137  \
        Y_A->table[26] 0.136 Y_A->table[27] 0.135 Y_A->table[28] 0.135  \
        Y_A->table[29] 0.135 Y_A->table[30] 0.135

    settab2const K_mit_usb Y_B 0 11 0.0    //-0.1 thru -0.045 => 0.0

    // -0.040	Volts
    //
    // -0.030	Volts
    // -0.020
    // -0.010
    // 0.00
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield K_mit_usb Y_B->table[12] 0.0 Y_B->table[13] 0.03  \
        Y_B->table[14] 0.06 Y_B->table[15] 0.12 Y_B->table[16] 0.25  \
        Y_B->table[17] 0.39 Y_B->table[18] 0.57 Y_B->table[19] 0.695  \
        Y_B->table[20] 0.78 Y_B->table[21] 0.825 Y_B->table[22] 0.845  \
        Y_B->table[23] 0.857 Y_B->table[24] 0.862 Y_B->table[25] 0.863  \
        Y_B->table[26] 0.864 Y_B->table[27] 0.865 Y_B->table[28] 0.865  \
        Y_B->table[29] 0.865 Y_B->table[30] 0.865

		/* Setting the calc_mode to NO_INTERP for speed */
		setfield K_mit_usb Y_A->calc_mode 0 Y_B->calc_mode 0

		/* tweaking the tables for the tabchan calculation */
		tweak_tabchan K_mit_usb Y

		/* Filling the tables using B-SPLINE interpolation */
		call K_mit_usb TABFILL Y 3000 0

		setfield K_mit_usb X_A->sy 5.0 X_B->sy 5.0 Y_A->sy 5.0  \
		    Y_B->sy 5.0 Ek {EK}

end

function make_K2_mit_usb
	if (({exists K2_mit_usb}))
		return
	end
float EK = -0.07

	if (({exists K_mit_usb}))
		move K_mit_usb K2_mit_usb
		make_K_mit_usb
	else
		make_K_mit_usb
		move K_mit_usb K2_mit_usb
	end

	setfield K2_mit_usb X_A->sy 20.0 X_B->sy 20.0 Y_A->sy 20.0  \
	    Y_B->sy 20.0 Ek {EK}
end

function make_K_slow_usb
	if (({exists K_slow_usb}))
		return
	end
	float EK = -0.07

	if (({exists K_mit_usb}))
		move K_mit_usb K_slow_usb
		make_K_mit_usb
	else
		make_K_mit_usb
		move K_mit_usb K_slow_usb
	end
	setfield K_slow_usb X_A->sy 1.0 X_B->sy 1.0 Y_A->sy 1.0  \
	    Y_B->sy 1.0
end

//========================================================================
//			Tabchan Na Mitral cell channel 
//========================================================================

function make_Na_mit_usb
	if (({exists Na_mit_usb}))
		return
	end

	/* offset both for erest and for thresh */
	float THRESH = -0.055
	/* Sodium reversal potl */
	float ENA = 0.045

	create tabchannel Na_mit_usb
		//	V
		//	S
		//	A
		//	S
		setfield ^ Ek {ENA} Gbar {1.2e3*SOMA_A} Ik 0 Gk 0  \
		    Xpower 3 Ypower 1 Zpower 0

	setup_tabchan Na_mit_usb X {320e3*(0.013 + THRESH)} -320e3 -1.0  \
	    {-1.0*(0.013 + THRESH)} -0.004 {-280e3*(0.040 + THRESH)}  \
	    280e3 -1.0 {-1.0*(0.040 + THRESH)} 5.0e-3

	setup_tabchan Na_mit_usb Y 128.0 0.0 0.0 {-1.0*(0.017 + THRESH)} \
	     0.018 4.0e3 0.0 1.0 {-1.0*(0.040 + THRESH)} -5.0e-3
end

//========================================================================

function make_Na2_mit_usb
	if (({exists Na2_mit_usb}))
		return
	end
	/* offset both for erest and for thresh */
	float THRESH = -0.060
	/* Sodium reversal potl */
	float ENA = 0.045

	create tabchannel Na2_mit_usb
		//	V
		//	S
		//	A
		//	S
		setfield ^ Ek {ENA} Gbar {1.2e3*SOMA_A} Ik 0 Gk 0  \
		    Xpower 3 Ypower 1 Zpower 0

	setup_tabchan Na2_mit_usb X {320e3*(0.013 + THRESH)} -320e3 -1.0 \
	     {-1.0*(0.013 + THRESH)} -0.004 {-280e3*(0.040 + THRESH)}  \
	    280e3 -1.0 {-1.0*(0.040 + THRESH)} 5.0e-3

	setup_tabchan Na2_mit_usb Y 128.0 0.0 0.0  \
	    {-1.0*(0.017 + THRESH)} 0.018 4.0e3 0.0 1.0  \
	    {-1.0*(0.040 + THRESH)} -5.0e-3
end

//========================================================================
// CONSTANTS
float EGlu = 0.045
float EGABA_1 = -0.080
float EGABA_2 = -0.080
float SOMA_A = 1e-9
float GGlu = SOMA_A*50
float GGABA_1 = SOMA_A*50
float GGABA_2 = SOMA_A*50

//===================================================================
//                     SYNAPTIC CHANNELS   (Values guessed at)
//===================================================================


function make_glu_mit_usb
	if (({exists glu_mit_usb}))
		return
	end

	// for receptor input only
	create channelC2 glu_mit_usb
    	// sec
    	// sec
    	// Siemens
    	setfield glu_mit_usb Ek {EGlu} tau1 {4.0e-3} tau2 {4.0e-3}  \
    	    gmax {GGlu}
end

function make_GABA_1_mit_usb
	if (({exists GABA_1_mit_usb}))
		return
	end

	// for both dd and ax inputs
	create ddsyn GABA_1_mit_usb
	call GABA_1_mit_usb TABCREATE 10 -0.065 0.05
   	// sec
   	// sec
   	// Siemens
   	// Setting up the table for 
   	// transforming from presyn Vm to
   	// activation.
   	setfield GABA_1_mit_usb Ek {EGABA_1} tau1 {10.0e-3}  \
   	    tau2 {10.0e-3} gmax {GGABA_1} transf->table[0] 0  \
   	    transf->table[1] 0.02 transf->table[2] 0.05  \
   	    transf->table[3] 0.1 transf->table[4] 0.2  \
   	    transf->table[5] 0.5 transf->table[6] 0.8  \
   	    transf->table[7] 0.9 transf->table[8] 0.95  \
   	    transf->table[9] 0.98 transf->table[10] 1
    call GABA_1_mit_usb TABFILL 1000 0
end

function make_GABA_2_mit_usb
	if (({exists GABA_2_mit_usb}))
		return
	end

	// for both dd and ax inputs
	create ddsyn GABA_2_mit_usb
	call GABA_2_mit_usb TABCREATE 10 -0.065 0.05
   	// sec
   	// sec
   	// Siemens
   	// Setting up the table for 
   	// transforming from presyn Vm to
   	// activation.
   	setfield GABA_2_mit_usb Ek {EGABA_2} tau1 {10.0e-3}  \
   	    tau2 {10.0e-3} gmax {GGABA_2} transf->table[0] 0  \
   	    transf->table[1] 0.05 transf->table[2] 0.1  \
   	    transf->table[3] 0.2 transf->table[4] 0.5  \
   	    transf->table[5] 0.7 transf->table[6] 0.8  \
   	    transf->table[7] 0.9 transf->table[8] 0.95  \
   	    transf->table[9] 0.98 transf->table[10] 1
    call GABA_2_mit_usb TABFILL 1000 0
end

function make_glu_gran_usb
	if (({exists glu_gran_usb}))
		return
	end

	// for dd, ax, and centrif inputs
	create ddsyn glu_gran_usb
	call glu_gran_usb TABCREATE 10 -0.070 0.05
    	// sec
    	// sec
    	// Siemens
    	// Setting up the table for 
    	// transforming from presyn Vm to
    	// activation.
    	setfield glu_gran_usb Ek {EGlu} tau1 {4.0e-3} tau2 {6.0e-3}  \
    	    gmax {GGlu} transf->table[0] 0 transf->table[1] 0.05  \
    	    transf->table[2] 0.1 transf->table[3] 0.2  \
    	    transf->table[4] 0.5 transf->table[5] 0.7  \
    	    transf->table[6] 0.8 transf->table[7] 0.9  \
    	    transf->table[8] 0.95 transf->table[9] 0.98  \
    	    transf->table[10] 1
    call glu_gran_usb TABFILL 1000 0
end

function make_glu_pg_usb
	if (({exists glu_pg_usb}))
		return
	end

	// for dd, ax, and centrif inputs
	create ddsyn glu_pg_usb
	call glu_pg_usb TABCREATE 10 -0.070 0.05
    	// sec
    	// sec
    	// Siemens
    	// Setting up the table for 
    	// transforming from presyn Vm to
    	// activation.
    	setfield glu_pg_usb Ek {EGlu} tau1 {4.0e-3} tau2 {6.0e-3}  \
    	    gmax {GGlu} transf->table[0] 0 transf->table[1] 0.02  \
    	    transf->table[2] 0.05 transf->table[3] 0.1  \
    	    transf->table[4] 0.2 transf->table[5] 0.5  \
    	    transf->table[6] 0.8 transf->table[7] 0.9  \
    	    transf->table[8] 0.95 transf->table[9] 0.98  \
    	    transf->table[10] 1
    call glu_pg_usb TABFILL 1000 0
end

function make_olf_receptor
	if (({exists olf_receptor}))
		return
	end
// Volts
float ENa = 0.045

	create receptor2 olf_receptor
		//sec
		//sec
		//Siemens
		// unitless
		setfield ^ Ek {ENa} tau1 0.05 tau2 0.1 gmax 5e-8  \
		    modulation 1
end

function make_spike
	if (({exists spike}))
		return
	end

	create spike spike
	// V
	// sec
	setfield spike thresh -0.00 abs_refract 10e-3 output_amp 1
	create axon spike/axon
	addmsg spike spike/axon BUFFER name
end

//========================================================================

function make_Kca_mit_usb
	if (({exists Kca_mit_usb}))
		return
	end
	float EK = -0.08

	create vdep_channel Kca_mit_usb
		//	V
		//	S
		//	A
		//	S
		setfield ^ Ek {EK} gbar {360.0*SOMA_A} Ik 0 Gk 0

	create table Kca_mit_usb/qv
	call Kca_mit_usb/qv TABCREATE 100 -0.1 0.1
	int i
	float x, dx, y
	x = -0.1
	dx = 0.2/100.0
	for (i = 0; i <= 100; i = i + 1)
		y = {exp {(x - {EREST_ACT})/0.027}}
		setfield Kca_mit_usb/qv table->table[{i}] {y}
		x = x + dx
	end

	create tabgate Kca_mit_usb/qca

	setupgate Kca_mit_usb/qca alpha  {5.0e5*0.015}  \
	    -5.0e5 -1.0 -0.015.0 -0.0013 -size 1000 -range 0.0 0.01

	call Kca_mit_usb/qca TABCREATE beta 1 -1 100
	setfield Kca_mit_usb/qca beta->table[0] 50
	setfield Kca_mit_usb/qca beta->table[1] 50

	addmsg Kca_mit_usb/qv Kca_mit_usb/qca PRD_ALPHA output
	addmsg Kca_mit_usb/qca Kca_mit_usb MULTGATE m 1
	addfield Kca_mit_usb addmsg1
	addfield Kca_mit_usb addmsg2
	setfield  Kca_mit_usb  \
	    addmsg1 "../Ca_mit_conc		qca		VOLTAGE		Ca" \
	      \
	    addmsg2 "..					qv		INPUT		Vm"
end

//========================================================================
//			Ca conc - mitral cell
//========================================================================

function make_Ca_mit_conc
	if (({exists Ca_mit_conc}))
		return
	end
	create Ca_concen Ca_mit_conc
	// sec
	// Curr to conc
	setfield Ca_mit_conc tau 0.01 B 5.2e-6 Ca_base 0.00001
	addfield Ca_mit_conc addmsg1
	setfield  Ca_mit_conc  \
	    addmsg1 "../LCa3_mit_usb	.		INCREASE	Ik"
end
//genesis

/* FILE INFORMATION
** 
** Tabchannel implementation of Hodgkin-Huxley squid Na and K channels
** This version uses SI (MKS) units and is equivalent to (but faster than)
** hh_chan.g, except for the function and channel names.
*/

// CONSTANTS
float EREST_ACT = -0.060
float ENA = 0.045
float EK = -0.090
/* Square meters */
float SOMA_A = 1e-9

/*************************************************************************

The function setupalpha uses the same form for the gate variables
X and Y as the vdep_gate, namely: (A+B*V)/(C+exp((V+D)/F))
The constants above are related to the hh_channel constants by:

EXPONENTIAL :
gate variables          in terms of hh-channel variables
A			A
B			0
C			0
D			-V0
F			-B

SIGMOID :
Gate			in terms of hh-ch variables
A			A
B			0
C			1
D			-V0
F			B

LINOID :
A			-A * V0
B			A
C			-1
D			-V0
F			B

*************************************************************************/
//========================================================================
//			Tabchan Na channel 
//========================================================================

function make_Na_hh_tchan
	str chanpath = "Na_hh_tchan"
//	str chanpath = "Na_squid_hh"
	if ({exists {chanpath}})
		return
	end

	create tabchannel {chanpath}
		//	V
		//	S
		//	A
		//	S
		setfield ^ Ek {ENA} Gbar {1.2e3*SOMA_A} Ik 0 Gk 0  \
		    Xpower 3 Ypower 1 Zpower 0

	setupalpha {chanpath} X {0.1e6*(0.025 + EREST_ACT)} -0.1e6  \
	    -1.0 {-1.0*(0.025 + EREST_ACT)} -0.01  \
	    4e3 0.0 0.0 {-1.0*EREST_ACT} 18e-3

	setupalpha {chanpath} Y 70.0 0.0 0.0  \
	    {-1.0*EREST_ACT} 0.02 1.0e3 0.0 1.0  \
	    {-1.0*(0.030 + EREST_ACT)} -10.0e-3
end

//========================================================================
//			Tabchan version of K channel
//========================================================================
function make_K_hh_tchan
	str chanpath = "K_hh_tchan"
//	str chanpath = "K_squid_hh"
	if (({exists {chanpath}}))
		return
	end

	create tabchannel {chanpath}
		//	V
		//	S
		//	A
		//	S
		setfield ^ Ek {EK} Gbar {360.0*SOMA_A} Ik 0 Gk 0  \
		    Xpower 4 Ypower 0 Zpower 0

	setupalpha {chanpath} X {10e3*(0.01 + EREST_ACT)} -10.0e3  \
	    -1.0 {-1.0*(0.01 + EREST_ACT)} -0.01 125.0 0.0 0.0  \
	    {-1.0*EREST_ACT} 80.0e-3
end
// genesis

/* FILE INFORMATION
** Rat Na channel, cloned, in oocyte expression system.
** Data from :
** Stuhmer, Methfessel, Sakmann, Noda an Numa, Eur Biophys J 1987
**	14:131-138
**
** Expts carred out at 16 deg Celsius.
** 
** Implemented in tabchan format by Upinder S. Bhalla March 1991
** This file depends on functions and constants defined in defaults.g
*/

// CONSTANTS
// (I-current)
float ENa = 0.045
// sq m
float SOMA_A = 1e-9


/*********************************************************************
**                          I-Current (Na)
*********************************************************************/

function make_Na_rat_smsnn// Na current

	// (I-current)
	float ENa = 0.045
	float x, y, dx
	int i
    if (({exists Na_rat_smsnn}))
        return
    end

    create tabchannel Na_rat_smsnn
    setfield Na_rat_smsnn Ek {ENa} Gbar {1200.0*SOMA_A} Ik 0 Gk 0  \
        Xpower 3 Ypower 1 Zpower 0

	call Na_rat_smsnn TABCREATE X 30 -0.1 0.05

    // -0.100 Volts
    // -0.095 Volts
    // -0.090 Volts
    // -0.085 Volts
    // -0.080 Volts
    // -0.075 Volts
    // -0.070 Volts
    // -0.065 Volts
    // -0.060 Volts
    // -0.055 Volts
    // -0.050 Volts
    // -0.045 Volts
    // -0.040 Volts
    // -0.030
    // -0.020
    // -0.010
    // 0.0
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield Na_rat_smsnn X_A->table[0] 1.0e-4 X_A->table[1] 1.0e-4  \
        X_A->table[2] 1.2e-4 X_A->table[3] 1.45e-4 X_A->table[4] 1.67e-4 \
         X_A->table[5] 2.03e-4 X_A->table[6] 2.47e-4  \
        X_A->table[7] 3.20e-4 X_A->table[8] 3.63e-4  \
        X_A->table[9] 4.94e-4 X_A->table[10] 4.07e-4  \
        X_A->table[11] 4.00e-4 X_A->table[12] 3.56e-4  \
        X_A->table[13] 3.49e-4 X_A->table[14] 3.12e-4  \
        X_A->table[15] 2.83e-4 X_A->table[16] 2.62e-4  \
        X_A->table[17] 2.25e-4 X_A->table[18] 2.03e-4  \
        X_A->table[19] 1.74e-4 X_A->table[20] 1.67e-4  \
        X_A->table[21] 1.31e-4 X_A->table[22] 1.23e-4  \
        X_A->table[23] 1.16e-4 X_A->table[24] 1.02e-4  \
        X_A->table[25] 0.87e-4 X_A->table[26] 0.73e-4  \
        X_A->table[27] 0.80e-4 X_A->table[28] 0.80e-4  \
        X_A->table[29] 0.80e-4 X_A->table[30] 0.80e-4

	x = -0.1
	dx = 0.15/30.0

	for (i = 0; i <= 30; i = i + 1)
		y = 1.0/(1.0 + {exp {-(x + 0.041)/0.0086}})
		setfield Na_rat_smsnn X_B->table[{i}] {y}
		x = x + dx
	end
	tau_tweak_tabchan Na_rat_smsnn X
	setfield Na_rat_smsnn X_A->calc_mode 0 X_B->calc_mode 0
	call Na_rat_smsnn TABFILL X 3000 0


	call Na_rat_smsnn TABCREATE Y 30 -0.1 0.05
   // settab2const(Na_rat_smsnn,Y_A,0,10,6.4e-3)
	//-0.1 thru -0.05=>0.0

    // -0.100 Volts
    // -0.095 Volts
    // -0.090 Volts
    // -0.085 Volts
    // -0.080 Volts
    // -0.075 Volts
    // -0.070 Volts
    // -0.065 Volts
    // -0.060 Volts
    // -0.055 Volts
    // -0.050 Volts
    // -0.045 Volts
    // -0.040 Volts
    // -0.030
    // -0.020
    // -0.010
    // 0.0
    // 0.010
    // 0.020
    // 0.030
    // 0.040
    // 0.050
    setfield Na_rat_smsnn Y_A->table[0] 0.9e-3 Y_A->table[1] 1.0e-3  \
        Y_A->table[2] 1.2e-3 Y_A->table[3] 1.45e-3 Y_A->table[4] 1.7e-3  \
        Y_A->table[5] 2.05e-3 Y_A->table[6] 2.55e-3 Y_A->table[7] 3.2e-3 \
         Y_A->table[8] 4.0e-3 Y_A->table[9] 5.0e-3  \
        Y_A->table[10] 6.49e-3 Y_A->table[11] 6.88e-3  \
        Y_A->table[12] 4.07e-3 Y_A->table[13] 2.71e-3  \
        Y_A->table[14] 2.03e-3 Y_A->table[15] 1.55e-3  \
        Y_A->table[16] 1.26e-3 Y_A->table[17] 1.07e-3  \
        Y_A->table[18] 0.87e-3 Y_A->table[19] 0.78e-3  \
        Y_A->table[20] 0.68e-3 Y_A->table[21] 0.63e-3  \
        Y_A->table[22] 0.58e-3 Y_A->table[23] 0.53e-3  \
        Y_A->table[24] 0.48e-3 Y_A->table[25] 0.48e-3  \
        Y_A->table[26] 0.48e-3 Y_A->table[27] 0.48e-3  \
        Y_A->table[28] 0.48e-3 Y_A->table[29] 0.43e-3  \
        Y_A->table[30] 0.39e-3

	x = -0.1
	dx = 0.15/30.0
	for (i = 0; i <= 30; i = i + 1)
		y = 1.0/(1.0 + {exp {(x + 0.064)/0.0102}})
		setfield Na_rat_smsnn Y_B->table[{i}] {y}
		x = x + dx
	end
	tau_tweak_tabchan Na_rat_smsnn Y
	setfield Na_rat_smsnn Y_A->calc_mode 0 Y_B->calc_mode 0
	call Na_rat_smsnn TABFILL Y 3000 0
end

/*********************************************************************/
//genesis

/* FILE INFORMATION
** The 1991 Traub set of voltage and concentration dependent channels
** Implemented as tabchannels by : Dave Beeman
**      R.D.Traub, R. K. S. Wong, R. Miles, and H. Michelson
**	Journal of Neurophysiology, Vol. 66, p. 635 (1991)
**
** This file depends on functions and constants defined in defaults.g
** As it is also intended as an example of the use of the tabchannel
** object to implement concentration dependent channels, it has extensive
** comments.  Note that the original units used in the paper have been
** converted to SI (MKS) units.  Also, we define the ionic equilibrium 
** potentials relative to the resting potential, EREST_ACT.  In the
** paper, this was defined to be zero.  Here, we use -0.060 volts, the
** measured value relative to the outside of the cell.
*/

// CONSTANTS
float EREST_ACT = -0.060 /* hippocampal cell resting potl */
float ENA = 0.115 + EREST_ACT // 0.055
float EK = -0.015 + EREST_ACT // -0.075
float ECA = 0.140 + EREST_ACT // 0.080
float SOMA_A = 3.320e-9       // soma area in square meters

/*
For these channels, the maximum channel conductance (Gbar) has been
calculated using the CA3 soma channel conductance densities and soma
area.  Typically, the functions which create these channels will be used
to create a library of prototype channels.  When the cell reader creates
copies of these channels in various compartments, it will set the actual
value of Gbar by calculating it from the cell parameter file.
*/

//========================================================================
//                      Tabulated Ca Channel
//========================================================================

function make_Ca_hip_traub91
        if ({exists Ca_hip_traub91})
                return
        end

        create  tabchannel      Ca_hip_traub91
                setfield        ^       \
                Ek              {ECA}   \               //      V
                Gbar            { 40 * SOMA_A }      \  //      S
                Ik              0       \               //      A
                Gk              0       \               //      S
                Xpower  2       \
                Ypower  1       \
                Zpower  0

/*
Often, the alpha and beta rate parameters can be expressed in the functional
form y = (A + B * x) / (C + {exp({(x + D) / F})}).  When this is the case,
case, the command "setupalpha chan gate AA AB AC AD AF BA BB BC BD BF" can be
used to simplify the process of initializing the A and B tables for the X, Y
and Z gates.  Although setupalpha has been implemented as a compiled GENESIS
command, it is also defined as a script function in the neurokit/prototypes
defaults.g file.  Although this command can be used as a "black box", its
definition shows some nice features of the tabchannel object, and some tricks
we will need when the rate parameters do not fit this form.
*/

// Converting Traub's expressions for the gCa/s alpha and beta functions
// to SI units and entering the A, B, C, D and F parameters, we get:

        setupalpha Ca_hip_traub91 X 1.6e3  \
                 0 1.0 {-1.0 * (0.065 + EREST_ACT) } -0.01389       \
                 {-20e3 * (0.0511 + EREST_ACT) }  \
                 20e3 -1.0 {-1.0 * (0.0511 + EREST_ACT) } 5.0e-3 

/* 
   The Y gate (gCa/r) is not quite of this form.  For V > EREST_ACT, alpha =
   5*{exp({-50*(V - EREST_ACT)})}.  Otherwise, alpha = 5.  Over the entire
   range, alpha + beta = 5.  To create the Y_A and Y_B tables, we use some
   of the pieces of the setupalpha function.
*/

// Allocate space in the A and B tables with room for xdivs+1 = 50 entries,
// covering the range xmin = -0.1 to xmax = 0.05.
        float   xmin = -0.1
        float   xmax = 0.05
        int     xdivs = 49
	call Ca_hip_traub91 TABCREATE Y {xdivs} {xmin} {xmax}

// Fill the Y_A table with alpha values and the Y_B table with (alpha+beta)
        int i
        float x,dx,y
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
	    if (x > EREST_ACT)
                y = 5.0*{exp {-50*(x - EREST_ACT)}}
	    else
		y = 5.0
	    end
            setfield Ca_hip_traub91 Y_A->table[{i}] {y}
            setfield Ca_hip_traub91 Y_B->table[{i}] 5.0
            x = x + dx
        end

// For speed during execution, set the calculation mode to "no interpolation"
// and use TABFILL to expand the table to 3000 entries.
           setfield Ca_hip_traub91 Y_A->calc_mode 0   Y_B->calc_mode 0
           call Ca_hip_traub91 TABFILL Y 3000 0
end

/****************************************************************************
Next, we need an element to take the Calcium current calculated by the Ca
channel and convert it to the Ca concentration.  The "Ca_concen" object
solves the equation dC/dt = B*I_Ca - C/tau, and sets Ca = Ca_base + C.  As
it is easy to make mistakes in units when using this Calcium diffusion
equation, the units used here merit some discussion.

With Ca_base = 0, this corresponds to Traub's diffusion equation for
concentration, except that the sign of the current term here is positive, as
GENESIS uses the convention that I_Ca is the current flowing INTO the
compartment through the channel.  In SI units, the concentration is usually
expressed in moles/m^3 (which equals millimoles/liter), and the units of B
are chosen so that B = 1/(ion_charge * Faraday * volume). Current is
expressed in amperes and one Faraday = 96487 coulombs.  However, in this
case, Traub expresses the concentration in arbitrary units, current in
microamps and uses tau = 13.33 msec.  If we use the same concentration units,
but express current in amperes and tau in seconds, our B constant is then
10^12 times the constant (called "phi") used in the paper.  The actual value
used will be typically be determined by the cell reader from the cell
parameter file.  However, for the prototype channel we wlll use Traub's
corrected value for the soma.  (An error in the paper gives it as 17,402
rather than 17.402.)  In our units, this will be 17.402e12.

****************************************************************************/

//========================================================================
//                      Ca conc
//========================================================================

function make_Ca_hip_conc
        if ({exists Ca_hip_conc})
                return
        end
        create Ca_concen Ca_hip_conc
        setfield Ca_hip_conc \
                tau     0.01333   \      // sec
                B       17.402e12 \      // Curr to conc for soma
                Ca_base 0.0
        addfield Ca_hip_conc addmsg1
        setfield Ca_hip_conc \
                addmsg1        "../Ca_hip_traub91 . I_Ca Ik"
end
/*
This Ca_concen element should receive an "I_Ca" message from the calcium
channel, accompanied by the value of the calcium channel current.  As we
will ordinarily use the cell reader to create copies of these prototype
elements in one or more compartments, we need some way to be sure that the
needed messages are established.  Although the cell reader has enough
information to create the messages which link compartments to their channels
and to other adjacent compartments, it most be provided with the information
needed to establish additional messages.  This is done by placing the
message string in a user-defined field of one of the elements which is
involved in the message.  The cell reader recognizes the added field names
"addmsg1", "addmsg2", etc. as indicating that they are to be
evaluated and used to set up messages.  The paths are relative to the
element which contains the message string in its added field.  Thus,
"../Ca_hip_traub91" refers to the sibling element Ca_hip_traub91 and "."
refers to the Ca_hip_conc element itself.
*/

//========================================================================
//             Tabulated Ca-dependent K AHP Channel
//========================================================================

/* This is a tabchannel which gets the calcium concentration from Ca_hip_conc
   in order to calculate the activation of its Z gate.  It is set up much
   like the Ca channel, except that the A and B tables have values which are
   functions of concentration, instead of voltage.
*/

function make_Kahp_hip_traub91
        if ({exists Kahp_hip_traub91})
                return
        end

        create  tabchannel      Kahp_hip_traub91
                setfield        ^       \
                Ek              {EK}   \               //      V
                Gbar            { 8 * SOMA_A }    \    //      S
                Ik              0       \              //      A
                Gk              0       \              //      S
                Xpower  0       \
                Ypower  0       \
                Zpower  1

// Allocate space in the Z gate A and B tables, covering a concentration
// range from xmin = 0 to xmax = 1000, with 50 divisions
        float   xmin = 0.0
        float   xmax = 1000.0
        int     xdivs = 50

        call Kahp_hip_traub91 TABCREATE Z {xdivs} {xmin} {xmax}
        int i
        float x,dx,y
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
            if (x < 500.0)
                y = 0.02*x
            else
                y = 10.0
            end
            setfield Kahp_hip_traub91 Z_A->table[{i}] {y}
            setfield Kahp_hip_traub91 Z_B->table[{i}] {y + 1.0}
            x = x + dx
        end
// For speed during execution, set the calculation mode to "no interpolation"
// and use TABFILL to expand the table to 3000 entries.
        setfield Kahp_hip_traub91 Z_A->calc_mode 0   Z_B->calc_mode 0
        call Kahp_hip_traub91 TABFILL Z 3000 0
// Use an added field to tell the cell reader to set up the
// CONCEN message from the Ca_hip_concen element
        addfield Kahp_hip_traub91 addmsg1
        setfield Kahp_hip_traub91 \
                addmsg1        "../Ca_hip_conc . CONCEN Ca"
end

//========================================================================
//  Ca-dependent K Channel - K(C) - (vdep_channel with table and tabgate)
//========================================================================
/*
The expression for the conductance of the potassium C-current channel has a
typical voltage and time dependent activation gate, where the time
dependence arises from the solution of a differential equation containing
the rate parameters alpha and beta.  It is multiplied by a function of
calcium concentration which is given explicitly rather than being obtained
from a differential equation.  Therefore, we need a way to multiply the
activation by a concentration dependent value which is determined from a
lookup table.  GENESIS 1.3 doesn't have a way to implement this with a
tabchannel, so we use the "vdep_channel" object here.  These channels
contain no gates and get their activation gate values from external gate
elements, via a "MULTGATE" message.  These gates are usually created with
"tabgate" objects, which are similar to the internal gates of the
tabchannels.  However, any object which can send the value of one of its
fields to the vdep_channel can be used as the gate.  Here, we use the
"table" object.  This generality makes the vdep_channel very useful, but it
is slower than the tabchannel because of the extra message passing involved.
*/

function make_Kc_hip_traub91
        if ({exists Kc_hip_traub91})
                return
        end

        create  vdep_channel    Kc_hip_traub91
                setfield        ^       \
                Ek              {EK}    \                       //      V
                gbar            { 100.0 * SOMA_A }      \       //      S
                Ik              0       \                       //      A
                Gk              0                               //      S

// Create a table for the function of concentration, allowing a
// concentration range of 0 to 1000, with 50 divisions.  Note that the
// internal field for the table object is called "table".
        float   xmin = 0.0
        float   xmax = 1000.0
        int     xdivs = 50

        create table            Kc_hip_traub91/tab
        call Kc_hip_traub91/tab TABCREATE {xdivs} {xmin} {xmax}
        int i
        float x,dx,y
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
            if (x < 250.0)
                y = x/250.0
            else
                y = 1.0
            end
            setfield Kc_hip_traub91/tab table->table[{i}] {y}
            x = x + dx
        end
// Expand the table to 3000 entries to use without interpolation.  The
// TABFILL syntax is slightly different from that used with tabchannels.
// Here there is only one internal table, so the table name is not specified.

	setfield Kc_hip_traub91/tab table->calc_mode 0
	call Kc_hip_traub91/tab TABFILL 3000 0

// Now make a tabgate for the voltage-dependent activation parameter.
        float   xmin = -0.1
        float   xmax = 0.05
        int     xdivs = 49
        create  tabgate         Kc_hip_traub91/X
        call Kc_hip_traub91/X TABCREATE alpha {xdivs} {xmin} {xmax}
        call Kc_hip_traub91/X TABCREATE beta  {xdivs} {xmin} {xmax}

// The tabgate has two internal tables, alpha and beta.  These are filled
// like those of the tabchannel.  Note that the "beta" table is really beta,
// not alpha + beta, as with the tabchannel.

        int i
        float x,dx,alpha,beta
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
            if (x < EREST_ACT + 0.05)
                alpha = {exp {53.872*(x - EREST_ACT) - 0.66835}}/0.018975
		beta = 2000*{exp {(EREST_ACT + 0.0065 - x)/0.027}} - alpha
            else
		alpha = 2000*{exp {(EREST_ACT + 0.0065 - x)/0.027}}
		beta = 0.0
            end
            setfield Kc_hip_traub91/X alpha->table[{i}] {alpha}
            setfield Kc_hip_traub91/X beta->table[{i}] {beta}
            x = x + dx
        end

// Expand the tables to 3000 entries to use without interpolation
	setfield Kc_hip_traub91/X alpha->calc_mode 0 beta->calc_mode 0
	call Kc_hip_traub91/X TABFILL alpha 3000 0
	call Kc_hip_traub91/X TABFILL beta  3000 0

        addmsg Kc_hip_traub91/tab  Kc_hip_traub91 MULTGATE output 1
        addmsg Kc_hip_traub91/X  Kc_hip_traub91  MULTGATE m 1
        addfield Kc_hip_traub91 addmsg1
        addfield Kc_hip_traub91 addmsg2
        setfield Kc_hip_traub91 \
                addmsg1        "../Ca_hip_conc  tab INPUT Ca" \
                addmsg2        "..  X  VOLTAGE Vm"
end
/*
The MULTGATE message is used to give the vdep_channel the value of the
activation variable and the power to which it should be raised.  As the
tabgate and table are sub-elements of the channel, they and their messages
to the channel will accompany it when copies are made.  However, we also
need to provide for messages which link to external elements.  The message
which sends the Ca concentration to the table and the one which sends the
compartment membrane potential to the tabgate are stored in environment
variables of the channel, so that they may be found by the cell reader.
*/

// The remaining channels are straightforward tabchannel implementations

//========================================================================
//                Tabchannel Na Hippocampal cell channel
//========================================================================
function make_Na_hip_traub91
        if ({exists Na_hip_traub91})
                return
        end

        create  tabchannel      Na_hip_traub91
                setfield        ^       \
                Ek              {ENA}   \               //      V
                Gbar            { 300 * SOMA_A }    \   //      S
                Ik              0       \               //      A
                Gk              0       \               //      S
                Xpower  2       \
                Ypower  1       \
                Zpower  0

        setupalpha Na_hip_traub91 X {320e3 * (0.0131 + EREST_ACT)}  \
                 -320e3 -1.0 {-1.0 * (0.0131 + EREST_ACT) } -0.004       \
                 {-280e3 * (0.0401 + EREST_ACT) } \
                 280e3 -1.0 {-1.0 * (0.0401 + EREST_ACT) } 5.0e-3 

        setupalpha Na_hip_traub91 Y 128.0 0.0 0.0  \
                {-1.0 * (0.017 + EREST_ACT)} 0.018  \
                4.0e3 0.0 1.0 {-1.0 * (0.040 + EREST_ACT) } -5.0e-3 
end

//========================================================================
//                Tabchannel K(DR) Hippocampal cell channel
//========================================================================
function make_Kdr_hip_traub91
        if ({exists Kdr_hip_traub91})
                return
        end

        create  tabchannel      Kdr_hip_traub91
                setfield        ^       \
                Ek              {EK}	\	           //      V
                Gbar            { 150 * SOMA_A }    \      //      S
                Ik              0       \                  //      A
                Gk              0       \                  //      S
                Xpower  1       \
                Ypower  0       \
                Zpower  0

                setupalpha Kdr_hip_traub91 X               \
                           {16e3 * (0.0351 + EREST_ACT)}   \  // AA
                           -16e3                           \  // AB
                           -1.0                            \  // AC
                           {-1.0 * (0.0351 + EREST_ACT) }  \  // AD
                           -0.005                          \  // AF
                           250                             \  // BA
                           0.0                             \  // BB
                           0.0                             \  // BC
                           {-1.0 * (0.02 + EREST_ACT)}     \  // BD
                           0.04                               // BF

end

//========================================================================
//                Tabchannel K(A) Hippocampal cell channel
//========================================================================
function make_Ka_hip_traub91
        if ({exists Ka_hip_traub91})
                return
        end

        create  tabchannel      Ka_hip_traub91
                setfield        ^       \
                Ek              {EK}    \	          //      V
                Gbar            { 50 * SOMA_A }     \     //      S
                Ik              0       \                 //      A
                Gk              0       \                 //      S
                Xpower  1       \
                Ypower  1       \
                Zpower  0

        setupalpha Ka_hip_traub91 X {20e3 * (0.0131 + EREST_ACT)}  \
                 -20e3 -1.0 {-1.0 * (0.0131 + EREST_ACT) } -0.01    \
                 {-17.5e3 * (0.0401 + EREST_ACT) }  \
                 17.5e3 -1.0 {-1.0 * (0.0401 + EREST_ACT) } 0.01 

        setupalpha Ka_hip_traub91 Y 1.6 0.0 0.0  \
                {0.013 - EREST_ACT} 0.018  \
                50.0 0.0 1.0 {-1.0 * (0.0101 + EREST_ACT) } -0.005 
end
//genesis

/* FILE INFORMATION
** The Traub set of voltage and conc dependent channels
** Implemented by : Upinder S. Bhalla.
** 	R.D.Traub, Neuroscience Vol 7 No 5 pp 1233-1242 (1982)
**
** This file depends on functions and constants defined in defaults.g
*/

// CONSTANTS
/* hippocampal cell resting potl */
float EREST_ACT = -0.070
float ENA = 0.045
float EK = -0.085
float ECA = 0.070
float SOMA_A = 1e-9 /* Square meters */

//========================================================================
//			Tabulated Ca Channel
//========================================================================


function make_Ca_hip_traub
	if (({exists Ca_hip_traub}))
		return
	end


	create vdep_channel Ca_hip_traub
		setfield ^ Ek {ECA} gbar {1.2e3*SOMA_A} Ik 0 Gk 0

	create tabgate Ca_hip_traub/s
	/* there is a singularity at x=0, so I hack around that by using
	** an odd number of sample points */
	setup_table Ca_hip_traub/s alpha 101 {40e3*(0.060 + EREST_ACT)}  \
	    -40e3 -1.0 {-1.0*(0.060 + EREST_ACT)} -0.010
	setup_table Ca_hip_traub/s beta 101 {-5e3*(0.045 + EREST_ACT)}  \
	    5e3 -1.0 {-1.0*(0.045 + EREST_ACT)} 10.0e-3

	create tabgate Ca_hip_traub/r
	call Ca_hip_traub/r TABCREATE alpha 1 -1 1000
	setfield Ca_hip_traub/r alpha->table[0] 5.0
	setfield Ca_hip_traub/r alpha->table[1] 5.0

	setupgate Ca_hip_traub/r beta  {25.0*200.0} -25.0 \
	     -1.0 -200.0 -20.0  -size 1000 -range -1 1000

/*
	create Ca_concen Ca_hip_traub/conc
	set Ca_hip_traub/conc \
		tau		0.01	\			// sec
		B		{5.2e-6/(SOMA_XA* \
	    SOMA_L)} \	// Curr to conc
		Ca_base		0.0
*/

	addmsg Ca_hip_traub/s Ca_hip_traub MULTGATE m 5
	addmsg Ca_hip_traub/r Ca_hip_traub MULTGATE m 1
	addfield  Ca_hip_traub	addmsg1
	addfield  Ca_hip_traub	addmsg2
	addfield  Ca_hip_traub	addmsg3
	setfield  Ca_hip_traub  \
	    addmsg1 "../Ca_mit_conc	r	VOLTAGE	Ca" \
	    addmsg2 ".		../Ca_mit_conc	I_Ca	Ik" \
	    addmsg3 "..	s		VOLTAGE	Vm"

end

//========================================================================
//			Ca conc
//========================================================================

function make_Ca_mit_conc
	if (({exists Ca_mit_conc}))
		return
	end
	create Ca_concen Ca_mit_conc
	// sec
	// Curr to conc
	setfield Ca_mit_conc tau 0.01 B 5.2e-6 Ca_base 0.00001
end

//========================================================================
//			Tabulated K channel - 
//========================================================================


function make_K_hip_traub
	if (({exists K_hip_traub}))
		return
	end

	create vdep_channel K_hip_traub
	setfield ^ Ek {EK} gbar {360.0*SOMA_A} Ik 0 Gk 0

	create tabgate K_hip_traub/n
	setup_table K_hip_traub/n alpha 100 {32e3*(0.015 + EREST_ACT)}  \
	    -32e3 -1.0 {-1.0*(0.015 + EREST_ACT)} -0.005
	setup_table K_hip_traub/n beta 100 500.0 0.0 0.0  \
	    {-1.0*(0.010 + EREST_ACT)} 40.0e-3

	create table K_hip_traub/ya2
	call K_hip_traub/ya2 TABCREATE 100 -0.1 0.1
	setup_table3 K_hip_traub/ya2 table 100 -0.1 0.1 2000 0 1  \
	    {-1.0*(0.085 + EREST_ACT)} -0.010
	create tabgate K_hip_traub/y
	setup_table K_hip_traub/y alpha 100 28 0 0  \
	    {-1.0*(0.015 + EREST_ACT)} 0.015
	setup_table K_hip_traub/y beta 100 400 0 1  \
	    {-1.0*(0.040 + EREST_ACT)} -0.010

	addmsg K_hip_traub/n K_hip_traub MULTGATE m 4
	addmsg K_hip_traub/y K_hip_traub MULTGATE m 1
	addmsg K_hip_traub/ya2 K_hip_traub/y SUM_ALPHA output
	addfield K_hip_traub  addmsg1
	addfield K_hip_traub  addmsg2
	addfield K_hip_traub  addmsg3
	setfield  K_hip_traub addmsg1 "..	n	VOLTAGE	Vm"  \
	    addmsg2 "..	y	VOLTAGE	Vm"  \
	    addmsg3 "..	ya2	INPUT	Vm"
end

//========================================================================
//			Tabulated Ca dependent K - channel.
//========================================================================

function make_Kca_hip_traub
	if (({exists Kca_hip_traub}))
		return
	end

	create vdep_channel Kca_hip_traub

	setfield ^ Ek {EK} gbar {360.0*SOMA_A} Ik 0 Gk 0

	create table Kca_hip_traub/qv
	call Kca_hip_traub/qv TABCREATE 100 -0.1 0.1
	int i
	float x, dx, y
	x = -0.1
	dx = 0.2/100.0
	for (i = 0; i <= 100; i = i + 1)
		y = {exp {(x - EREST_ACT)/0.027}}
		setfield Kca_hip_traub/qv table->table[{i}] {y}
		x = x + dx
	end

	create tabgate Kca_hip_traub/qca

	setupgate Kca_hip_traub/qca alpha  {5.0*200.0}  \
	    -5.0 -1.0 -200.0 -20.0 -size 1000 -range -1 100

	call Kca_hip_traub/qca TABCREATE beta 1 -1 100
	setfield Kca_hip_traub/qca beta->table[0] 2.0
	setfield Kca_hip_traub/qca beta->table[1] 2.0

	addmsg Kca_hip_traub/qv Kca_hip_traub/qca PRD_ALPHA output
	addmsg Kca_hip_traub/qca Kca_hip_traub MULTGATE m 1
	addfield  Kca_hip_traub addmsg1
	addfield  Kca_hip_traub addmsg2
	setfield  Kca_hip_traub  \
	    addmsg1 "../Ca_mit_conc	qca	VOLTAGE		Ca" \
	    addmsg2 "..		qv	INPUT		Vm"
end

//========================================================================
//genesis

/* FILE INFORMATION
** tabulated implementation of bsg cell voltage and Ca-dependent
** channels.
** Implemented in Neurokit format by : Upinder S. Bhalla.
**
**	Source : Yamada, Koch, and Adams
**	Methods in Neuronal Modeling, MIT press, ed Koch and Segev.
**
** This file depends on functions and constants defined in library.g
*/

// CONSTANTS

float V_OFFSET = 0.0
float VKTAU_OFFSET = 0.0
float VKMINF_OFFSET = 0.02
float ECA = 0.070
float ENa = 0.050
float EK = -0.07

/********************************************************************
**                       Fast Na Current
********************************************************************/
function make_Na_bsg_yka
	if (({exists Na_bsg_yka}))
		return
	end

    create tabchannel Na_bsg_yka
    setfield Na_bsg_yka Ek {ENa} Gbar {1200.0*{SOMA_A}} Ik 0 Gk 0  \
        Xpower 2 Ypower 1 Zpower 0

	setupalpha Na_bsg_yka X {-360e3*(0.033 - V_OFFSET)} -360e3  \
	    -1.0 {0.033 - V_OFFSET} -0.003 {4.0e5*(0.042 - V_OFFSET)}  \
	    4.0e5 -1.0 {0.042 - V_OFFSET} 0.020

	setupalpha Na_bsg_yka Y {1e5*(0.055 - V_OFFSET)} 1.0e5 -1.0  \
	    {0.055 - V_OFFSET} 0.006 4500.0 0.0 1.0 {-V_OFFSET} -0.010
end

/********************************************************************
**                       Fast Ca Current
********************************************************************/

function make_Ca_bsg_yka
	if (({exists Ca_bsg_yka}))
		return
	end

	int ndivs, i
	float x, dx, y

	create vdep_channel Ca_bsg_yka
	setfield Ca_bsg_yka Ek {ECA} gbar {200*{SOMA_A}}

	create tabgate Ca_bsg_yka/mgate
	call Ca_bsg_yka/mgate TABCREATE alpha 100 -0.2 0.1
	call Ca_bsg_yka/mgate TABCREATE beta 100 -0.2 0.1
	x = {getfield Ca_bsg_yka/mgate alpha->xmin}
	dx = {getfield Ca_bsg_yka/mgate alpha->dx}
	ndivs = {getfield Ca_bsg_yka/mgate alpha->xdivs}
	V_OFFSET = -0.065 	// Mit definitions

	for (i = 0; i <= ndivs; i = i + 1)
		if (x < -0.032)
		    setfield Ca_bsg_yka/mgate alpha->table[{i}] 0.0

		     y = ({exp {(x + 0.006 - V_OFFSET)/0.016}} + {exp {-(x + 0.006 - V_OFFSET)/0.016}})/7.8e-3
		     setfield Ca_bsg_yka/mgate beta->table[{i}] {y}
		else
		     y = ({exp {(x + 0.006 - V_OFFSET)/0.016}} + {exp {-(x + 0.006 - V_OFFSET)/0.016}})/(7.8e-3*(1.0 + {exp {-(x - 0.003 - V_OFFSET)/0.008}}))
		     setfield Ca_bsg_yka/mgate alpha->table[{i}] {y}
		     y = ({exp {(x + 0.006 - V_OFFSET)/0.016}} + {exp {-(x + 0.006 - V_OFFSET)/0.016}})/7.8e-3*(1.0 - 1.0/(1.0 + {exp {-(x - 0.003 - V_OFFSET)/0.008}}))
		     setfield Ca_bsg_yka/mgate beta->table[{i}] {y}
		end
		x = x + dx
	end


	create table Ca_bsg_yka/hgate
	call Ca_bsg_yka/hgate TABCREATE 100 0.0 1.0
	x = {getfield Ca_bsg_yka/hgate table->xmin}
	dx = {getfield Ca_bsg_yka/hgate table->dx}
	ndivs = {getfield Ca_bsg_yka/hgate table->xdivs}
	for (i = 0; i <= ndivs; i = i + 1)
		y = 0.01/(0.01 + x)
		setfield Ca_bsg_yka/hgate table->table[{i}] {y}
		x = x + dx
	end

	create table Ca_bsg_yka/nernst
	call Ca_bsg_yka/nernst TABCREATE 1000 0.00005 0.01
	x = {getfield Ca_bsg_yka/nernst table->xmin}
	dx = {getfield Ca_bsg_yka/nernst table->dx}
	ndivs = {getfield Ca_bsg_yka/nernst table->xdivs}
	for (i = 0; i <= ndivs; i = i + 1)
		y = 12.5e-3*{log {4.0/x}}
		setfield Ca_bsg_yka/nernst table->table[{i}] {y}
		x = x + dx
	end

/*
	create Ca_concen Ca_bsg_yka/conc
	set  Ca_bsg_yka/conc \
		tau		0.00001 \					// sec
		B		5.2e-6 \	// Moles per coulomb, later scaled to conc
		Ca_base	1.0e-4						// Moles per cubic m
*/

/* Send messages to and from conc, which is not on Ca_bsg_yka */
	addfield  Ca_bsg_yka addmsg1
	addfield  Ca_bsg_yka addmsg2
	addfield  Ca_bsg_yka addmsg3
	addfield  Ca_bsg_yka addmsg4
	setfield  Ca_bsg_yka  \
	    addmsg1 ".			../Ca_bsg_conc I_Ca Ik"  \
	    addmsg2 "../Ca_bsg_conc	nernst	INPUT Ca"  \
	    addmsg3 "../Ca_bsg_conc	hgate	INPUT Ca"  \
	    addmsg4 ".. mgate VOLTAGE Vm"

/*
	addmsg Ca_bsg_yka Ca_bsg_yka/conc I_Ca Ik
	addmsg Ca_bsg_yka/conc Ca_bsg_yka/nernst INPUT Ca
	addmsg Ca_bsg_yka/conc Ca_bsg_yka/hgate INPUT Ca
*/
	addmsg Ca_bsg_yka/nernst Ca_bsg_yka EK output
	addmsg Ca_bsg_yka/hgate Ca_bsg_yka MULTGATE output 1
	addmsg Ca_bsg_yka/mgate Ca_bsg_yka MULTGATE m 1
end

function make_Ca_bsg_conc
	if (({exists Ca_bsg_conc}))
		return
	end

	create Ca_concen Ca_bsg_conc
	setfield Ca_bsg_conc \
                tau     0.00001 \       // sec
                B       5.2e-6 \ // Moles per coulomb, later scaled to conc
                Ca_base 1.0e-4          // Moles per cubic m
end

/********************************************************************
**            Transient outward K current
********************************************************************/
function make_KA_bsg_yka
	if (({exists KA_bsg_yka}))
		return
	end

    create tabchannel KA_bsg_yka
    setfield KA_bsg_yka Ek {EK} Gbar {1200.0*{SOMA_A}} Ik 0 Gk 0  \
        Xpower 1 Ypower 1 Zpower 0

	setuptau KA_bsg_yka X 1.38e-3 0.0 1.0 -1.0e3 1.0 1.0  \
	    0.0 1.0 {0.042 - V_OFFSET} -0.013

	setuptau KA_bsg_yka Y 0.150 0.0 1.0 -1.0e3 1.0 1.0 0.0  \
	    1.0 {0.110 - V_OFFSET} 0.018
end

/********************************************************************
**            Non-inactivating Muscarinic K current
********************************************************************/
function make_KM_bsg_yka
	if (({exists KM_bsg_yka}))
		return
	end

	int i
	float x, dx, y

    create tabchannel KM_bsg_yka
    setfield KM_bsg_yka Ek {EK} Gbar {1200.0*{SOMA_A}} Ik 0 Gk 0  \
        Xpower 1 Ypower 0 Zpower 0

	call KM_bsg_yka TABCREATE X 49 -0.1 0.1
	x = -0.1
	dx = 0.2/49.0

	for (i = 0; i <= 49; i = i + 1)
		y = 1.0/(3.3*({exp {(x + 0.035 - V_OFFSET)/0.04}}) + {exp {-(x + 0.035 - V_OFFSET)/0.02}})
		setfield KM_bsg_yka X_A->table[{i}] {y}

		y = 1.0/(1.0 + {exp {-(x + 0.035 - V_OFFSET)/0.01}})
		setfield KM_bsg_yka X_B->table[{i}] {y}
		x = x + dx
	end
	tweaktau KM_bsg_yka X
	setfield KM_bsg_yka X_A->calc_mode 0 X_B->calc_mode 0
	call KM_bsg_yka TABFILL X 3000 0
end

/********************************************************************
**                  Delayed rectifying K current
********************************************************************/

function make_K_bsg_yka
	if (({exists K_bsg_yka}))
		return
	end
	int i
	float x, dx, y
	float a, b


    create tabchannel K_bsg_yka
    setfield K_bsg_yka Ek {EK} Gbar 1.17e-6 Ik 0 Gk 0 Xpower 2 Ypower 1  \
        Zpower 0

	call K_bsg_yka TABCREATE X 49 -0.1 0.1
	x = -0.1
	dx = 0.2/49.0

	for (i = 0; i <= 49; i = i + 1)
		a = -4.7e3*(x + 0.012 - VKTAU_OFFSET)/({exp {(x + 0.012 - VKTAU_OFFSET)/-0.012}} - 1.0)
		b = 1.0e3*{exp {-(x + 0.147 - VKTAU_OFFSET)/0.030}}
		setfield K_bsg_yka X_A->table[{i}] {1.0/(a + b)}

		a = -4.7e3*(x + 0.012 - VKMINF_OFFSET)/({exp {(x + 0.012 - VKMINF_OFFSET)/-0.012}} - 1.0)
		b = 1.0e3*{exp {-(x + 0.147 - VKMINF_OFFSET)/0.030}}
		setfield K_bsg_yka X_B->table[{i}] {a/(a + b)}
		x = x + dx
	end
	tweaktau K_bsg_yka X
	setfield K_bsg_yka X_A->calc_mode 0 X_B->calc_mode 0
	call K_bsg_yka TABFILL X 3000 0


	call K_bsg_yka TABCREATE Y 49 -0.1 0.1
	x = -0.1
	dx = 0.2/49.0

	for (i = 0; i <= 49; i = i + 1)
		if (x < -0.025)
			setfield K_bsg_yka Y_A->table[{i}] 6.0
		else
			setfield K_bsg_yka Y_A->table[{i}] 0.050
		end

		y = 1.0 + {exp {(x + 0.025 - V_OFFSET)/0.004}}
		setfield K_bsg_yka Y_B->table[{i}] {1.0/y}
		x = x + dx
	end
	tweaktau K_bsg_yka Y
	setfield K_bsg_yka Y_A->calc_mode 0 Y_B->calc_mode 0
	call K_bsg_yka TABFILL Y 3000 0
end

/********************************************************************/
