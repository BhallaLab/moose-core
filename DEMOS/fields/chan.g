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
		tweakalpha K_mit_usb X

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
		tweakalpha K_mit_usb Y

		/* Filling the tables using B-SPLINE interpolation */
		call K_mit_usb TABFILL Y 3000 0

//      Small bug here in Moose.
/*      setfield K_mit_usb X_A->sy 5.0 X_B->sy 5.0 Y_A->sy 5.0  \
		    Y_B->sy 5.0 Ek {EK} */

//      Workaround..
        setfield K_mit_usb Ek {EK} X_A->sy 5.0 X_B->sy 5.0 Y_A->sy 5.0  \
		    Y_B->sy 5.0

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

	setupalpha Na_mit_usb X {320e3*(0.013 + THRESH)} -320e3 -1.0  \
	    {-1.0*(0.013 + THRESH)} -0.004 {-280e3*(0.040 + THRESH)}  \
	    280e3 -1.0 {-1.0*(0.040 + THRESH)} 5.0e-3

	setupalpha Na_mit_usb Y 128.0 0.0 0.0 {-1.0*(0.017 + THRESH)} \
	     0.018 4.0e3 0.0 1.0 {-1.0*(0.040 + THRESH)} -5.0e-3
end
