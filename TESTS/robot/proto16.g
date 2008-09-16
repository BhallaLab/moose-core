//genesis

/* FILE INFORMATION
** Based mostly on the traub91proto.g by Dave Beeman
** Main difference is addition of Glu and NMDA channels
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

/* November 1999 update for GENESIS 2.2: Previous versions of this file used
   a combination of a table, tabgate, and vdep_channel to implement the
   Ca-dependent K Channel - K(C).  This new version uses the new tabchannel
   "instant" field, introduced in GENESIS 2.2, to implement an
   "instantaneous" gate for the multiplicative Ca-dependent factor in the
   conductance.   This allows these channels to be used with the fast
   hsolve chanmodes > 1.
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

function make_Ca
        if ({exists Ca})
                return
        end

        create  tabchannel      Ca
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

        setupalpha Ca X 1.6e3  \
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
	call Ca TABCREATE Y {xdivs} {xmin} {xmax}

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
            setfield Ca Y_A->table[{i}] {y}
            setfield Ca Y_B->table[{i}] 5.0
            x = x + dx
        end

// For speed during execution, set the calculation mode to "no interpolation"
// and use TABFILL to expand the table to 3000 entries.
           setfield Ca Y_A->calc_mode 0   Y_B->calc_mode 0
           call Ca TABFILL Y 3000 0
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

function make_Ca_conc
        if ({exists Ca_conc})
                return
        end
        create Ca_concen Ca_conc
        setfield Ca_conc \
                tau     0.01333   \      // sec
                B       17.402e12 \      // Curr to conc for soma
                Ca_base 0.0
        addfield Ca_conc addmsg1
        setfield Ca_conc \
                addmsg1        "../Ca . I_Ca Ik"
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
"../Ca" refers to the sibling element Ca and "."
refers to the Ca_conc element itself.
*/

//========================================================================
//             Tabulated Ca-dependent K AHP Channel
//========================================================================

/* This is a tabchannel which gets the calcium concentration from Ca_conc
   in order to calculate the activation of its Z gate.  It is set up much
   like the Ca channel, except that the A and B tables have values which are
   functions of concentration, instead of voltage.
*/

function make_K_AHP
        if ({exists K_AHP})
                return
        end

        create  tabchannel      K_AHP
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
        float   xmax = 500.0
        int     xdivs = 50

        call K_AHP TABCREATE Z {xdivs} {xmin} {xmax}
        int i
        float x,dx,y
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
            if (x < (xmax / 2.0 ))
                y = 0.02*x
            else
                y = 10.0
            end
            setfield K_AHP Z_A->table[{i}] {y}
            setfield K_AHP Z_B->table[{i}] {y + 1.0}
            x = x + dx
        end
// For speed during execution, set the calculation mode to "no interpolation"
// and use TABFILL to expand the table to 3000 entries.
        setfield K_AHP Z_A->calc_mode 0   Z_B->calc_mode 0
        call K_AHP TABFILL Z 3000 0
// Use an added field to tell the cell reader to set up the
// CONCEN message from the Ca_concen element
        addfield K_AHP addmsg1
        setfield K_AHP \
                addmsg1        "../Ca_conc . CONCEN Ca"
end

//========================================================================
//  Ca-dependent K Channel - K(C) - (vdep_channel with table and tabgate)
//========================================================================

/*
The expression for the conductance of the potassium C-current channel has a
typical voltage and time dependent activation gate, where the time dependence
arises from the solution of a differential equation containing the rate
parameters alpha and beta.  It is multiplied by a function of calcium
concentration that is given explicitly rather than being obtained from a
differential equation.  Therefore, we need a way to multiply the activation
by a concentration dependent value which is determined from a lookup table.
This is accomplished by using the Z gate with the new tabchannel "instant"
field, introduced in GENESIS 2.2, to implement an "instantaneous" gate for
the multiplicative Ca-dependent factor in the conductance.
*/

function make_K_C
        if ({exists K_C})
                return
        end

        create  tabchannel    K_C
                setfield        ^       \
                Ek              {EK}    \                       //      V
                Gbar            { 100.0 * SOMA_A }      \       //      S
                Ik              0       \                       //      A
                Gk              0                               //      S

// Now make a X-table for the voltage-dependent activation parameter.
        float   xmin = -0.1
        float   xmax = 0.05
        int     xdivs = 49
        call K_C TABCREATE X {xdivs} {xmin} {xmax}
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
            setfield K_C X_A->table[{i}] {alpha}
            setfield K_C X_B->table[{i}] {alpha+beta}
            x = x + dx
        end
// Expand the tables to 3000 entries to use without interpolation
	setfield K_C X_A->calc_mode 0 X_B->calc_mode 0
	setfield K_C Xpower 1
	call K_C TABFILL X 3000 0

// Create a table for the function of concentration, allowing a
// concentration range of 0 to 1000, with 50 divisions.  This is done
// using the Z gate, which can receive a CONCEN message.  By using
// the "instant" flag, the A and B tables are evaluated as lookup tables,
//  rather than being used in a differential equation.

        float   xmin = 0.0
        float   xmax = 500.0
        int     xdivs = 50

        call K_C TABCREATE Z {xdivs} {xmin} {xmax}
        int i
        float x,dx,y
        dx = (xmax - xmin)/xdivs
        x = xmin
        for (i = 0 ; i <= {xdivs} ; i = i + 1)
            if (x < (xmax / 4.0))
                // y = x/250.0
				y = x * 4.0 / xmax
            else
                y = 1.0
            end
	    /* activation will be computed as Z_A/Z_B */
            setfield K_C Z_A->table[{i}] {y}
            setfield K_C Z_B->table[{i}] 1.0
            x = x + dx
        end

	setfield K_C Z_A->calc_mode 0 Z_B->calc_mode 0
	setfield K_C Zpower 1
// Make it an instantaneous gate (no time constant)
	setfield K_C instant {INSTANTZ}
// Expand the table to 3000 entries to use without interpolation. 
	call K_C TABFILL Z 3000 0

// Now we need to provide for messages that link to external elements.
// The message that sends the Ca concentration to the Z gate tables is stored
// in an added field of the channel, so that it may be found by the cell
// reader.
        addfield K_C addmsg1
        setfield K_C addmsg1  "../Ca_conc  . CONCEN Ca" 
end

// The remaining channels are straightforward tabchannel implementations

//========================================================================
//                Tabchannel Na Hippocampal cell channel
//========================================================================
function make_Na
        if ({exists Na})
                return
        end

        create  tabchannel      Na
                setfield        ^       \
                Ek              {ENA}   \               //      V
                Gbar            { 300 * SOMA_A }    \   //      S
                Ik              0       \               //      A
                Gk              0       \               //      S
                Xpower  2       \
                Ypower  1       \
                Zpower  0

        setupalpha Na X {320e3 * (0.0131 + EREST_ACT)}  \
                 -320e3 -1.0 {-1.0 * (0.0131 + EREST_ACT) } -0.004       \
                 {-280e3 * (0.0401 + EREST_ACT) } \
                 280e3 -1.0 {-1.0 * (0.0401 + EREST_ACT) } 5.0e-3 

        setupalpha Na Y 128.0 0.0 0.0  \
                {-1.0 * (0.017 + EREST_ACT)} 0.018  \
                4.0e3 0.0 1.0 {-1.0 * (0.040 + EREST_ACT) } -5.0e-3 
end

//========================================================================
//                Tabchannel K(DR) Hippocampal cell channel
//========================================================================
function make_K_DR
        if ({exists K_DR})
                return
        end

        create  tabchannel      K_DR
                setfield        ^       \
                Ek              {EK}	\	           //      V
                Gbar            { 150 * SOMA_A }    \      //      S
                Ik              0       \                  //      A
                Gk              0       \                  //      S
                Xpower  1       \
                Ypower  0       \
                Zpower  0

        setupalpha K_DR X                          \
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
function make_K_A
        if ({exists K_A})
                return
        end

        create  tabchannel      K_A
                setfield        ^       \
                Ek              {EK}    \	          //      V
                Gbar            { 50 * SOMA_A }     \     //      S
                Ik              0       \                 //      A
                Gk              0       \                 //      S
                Xpower  1       \
                Ypower  1       \
                Zpower  0

        setupalpha K_A X {20e3 * (0.0131 + EREST_ACT)}  \
                 -20e3 -1.0 {-1.0 * (0.0131 + EREST_ACT) } -0.01    \
                 {-17.5e3 * (0.0401 + EREST_ACT) }  \
                 17.5e3 -1.0 {-1.0 * (0.0401 + EREST_ACT) } 0.01 

        setupalpha K_A Y 1.6 0.0 0.0  \
                {0.013 - EREST_ACT} 0.018  \
                50.0 0.0 1.0 {-1.0 * (0.0101 + EREST_ACT) } -0.005 
end

function make_glu
        if ({exists glu})
                return
        end
		create synchan glu
		setfield glu \
			Ek		0.0 \
			tau1	2.0e-3 \
			tau2	9.0e-3 \
			gmax	{ 40 * SOMA_A }
end

// This version makes an Mg_block as the surrogate for the channel.
// The readcell fails to recognize it and the channel conductance
// is not set correctly. The messaging is OK but the values not.
/*
function make_NMDA
	float CMg = 2	// [Mg] in mM
	float eta = 0.33                    // per mM
    float gamma = 60                    // per Volt


        if ({exists NMDA})
                return
        end
		create Mg_block NMDA
		setfield NMDA \
			CMg                     {CMg} \
			KMg_A                   {1.0/eta} \
			KMg_B                   {1.0/gamma}

		create synchan NMDA/chan
		setfield NMDA/chan \
			Ek		0 \
			tau1	100e-3 \
			tau2	200e-3 \
			gmax	{ 5 * SOMA_A }

		addmsg NMDA/chan NMDA CHANNEL Gk Ek

        addfield NMDA addmsg1
        setfield NMDA addmsg1        ".. ./chan VOLTAGE Vm"
        addfield NMDA addmsg2
        setfield NMDA addmsg2        ". .. CHANNEL Gk Ek"
        addfield NMDA addmsg3
        setfield NMDA addmsg3        ".. . VOLTAGE Vm"
end
*/

// This version has the NMDA channel on the compt, and the Mg_block
// on the channel.
// Let's see if readcell can do it.
function make_NMDA
	float CMg = 1.2	// [Mg] in mM
	float eta = 0.28                    // per mM
    float gamma = 62                    // per Volt
	float tau1_NMDA = 20e-3				// 20 msec
	float tau2_NMDA = 40e-3				// 20 msec

    if ({exists NMDA})
            return
    end
	create synchan NMDA
	setfield NMDA \
			Ek		0.0 \
			tau1	{tau1_NMDA} \
			tau2	{tau2_NMDA} \
			gmax	{ 5 * SOMA_A }
	create Mg_block NMDA/block
	setfield NMDA/block \
			CMg                     {CMg} \
			Zk						2 \
			KMg_A                   {1.0/eta} \
			KMg_B                   {1.0/gamma}


	addmsg NMDA NMDA/block CHANNEL Gk Ek

        addfield NMDA addmsg1
        setfield NMDA addmsg1        ".. ./block VOLTAGE Vm"
        addfield NMDA addmsg2
        setfield NMDA addmsg2        "./block .. CHANNEL Gk Ek"

		// Not sure if the readcell will spontaneously put in the
		// CHANNEL message from the NMDA channel to compt. Should not.
		// If it does, needs to be deleted.
end

// The Ca_NMDA channel is a subset of the NMDA channel that carries Ca.
// It is identical to above, except that the Ek for Ca is much higher:
// 0.1 V. This is about the reversal potl for 1 uM Ca_in, 2 mM out.
// Also we do not want this channel to contribute to the current,
// which is already accounted for in the main channel. So there is
// no CHANNEL message to the parent compartment.
// I would like to have used the Nernst to do the Ca potential, but
// Synchans do not take Ek messages so we can't change that.
function make_Ca_NMDA
	float CMg = 1.2	// [Mg] in mM
	float eta = 0.28                    // per mM
    float gamma = 62                    // per Volt
	float tau1_NMDA = 20e-3				// 20 msec
	float tau2_NMDA = 40e-3				// 20 msec

    if ({exists Ca_NMDA})
            return
    end
	create synchan Ca_NMDA
	// Note the difference in Ek.
	setfield Ca_NMDA \
			Ek		0.1 \
			tau1	{tau1_NMDA} \
			tau2	{tau2_NMDA} \
			gmax	{ 5 * SOMA_A }
	create Mg_block Ca_NMDA/block
	setfield Ca_NMDA/block \
			CMg                     {CMg} \
			Zk						2 \
			KMg_A                   {1.0/eta} \
			KMg_B                   {1.0/gamma}


	addmsg Ca_NMDA Ca_NMDA/block CHANNEL Gk Ek

        addfield Ca_NMDA addmsg1
        setfield Ca_NMDA addmsg1        ".. ./block VOLTAGE Vm"
		// Note that we do not connect up the CHANNEL message to
		// the compartment, as we will use this channel only for the
		// Ca conc changes.

/*
// This is really dumb. The synchan does not take the EK message
// so we cannot use the nernst with it.
	create nernst Ca_NMDA/nernst
	setfield Ca_NMDA/nernst \
		Cout	2.0	\	// External Ca conc is ~2 mM in mammals
		T		21	\	// Room temp
		valency	2	\
		scale	1		// Units of volts
	
	addmsg Ca_NMDA/nernst	Ca_NMDA	EK	E

    addfield Ca_NMDA addmsg3
    setfield Ca_NMDA addmsg3        "../NMDA_Ca_conc ./nernst CIN Ca"
	*/
end

/*
// This is some of the original version, did not work
function make_NMDA
		create table NMDA/vdep
		float xmin = -0.1
		float xmax = 0.05
		int xdivs = 100
		float dx = (xmax - xmin) / xdivs
		float x
		float a = 100	// units 1/V
		float b = -2	// no units
		int i
		call NMDA/vdep TABCREATE {xdivs} {xmin} {xmax}
		i = 0
		for ( x = xmin; x <= xmax; x = x + dx )
			setfield NMDA/vdep table->table[{i}] \
				{ 1.0 / ( 1.0 + { exp { b - a * x } } ) }
			i = i + 1
		end
		setfield NMDA/vdep step_mode 0
		addmsg NMDA NMDA/vdep 
		addmsg NMDA/vdep NMDA MOD output 

        addfield NMDA addmsg1
        setfield NMDA addmsg1        ".. ./vdep INPUT Vm"
end
		*/

function make_NMDA_Ca_conc
        if ({exists NMDA_Ca_conc})
                return
        end
        create Ca_concen NMDA_Ca_conc
        setfield NMDA_Ca_conc \
                tau     0.0133   \      // sec
                B       17.402e12 \      // overridden by cellreader.
                Ca_base 0.0
        addfield NMDA_Ca_conc addmsg1
        addfield NMDA_Ca_conc addmsg2
        setfield NMDA_Ca_conc \
                addmsg1        "../Ca_NMDA/block . I_Ca Ik" \
                addmsg2        "../Ca . I_Ca Ik"
end

//=====================================================================
//                        SPIKE DETECTOR
//=====================================================================
function make_axon
    if ({exists axon})
           return
    end
//  create axon /library/axon
    create  spikegen    /library/axon
        setfield /library/axon \
        thresh      -40e-3 \            // V
        abs_refract { 10e-3 } \         // sec
        output_amp      1
    
        //addmsg axon/spike axon BUFFER name
end
