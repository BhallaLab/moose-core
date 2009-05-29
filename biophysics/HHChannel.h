#ifndef _HHChannel_h
#define _HHChannel_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/

/**
 * The HHChannel class sets up a Hodkin-Huxley type ion channel.
 * The form used here is quite general and can handle up to 3 
 * gates, named X, Y and Z. The Z gate can be a function of 
 * concentration as well as voltage. The gates are normally computed
 * using the form
 *
 *            alpha(V)
 * closed <------------> open
 *          beta(V)
 *
 * where the rates for the transition are alpha and beta, and both
 * are functions of V.
 * The state variables for each gate (X_, Y_, and Z_) are 
 * the fraction in the open state.
 *
 * Gates can also be computed instantaneously, giving the instantaneous
 * ratio of alpha to beta rather than solving the above conversion
 * process.
 * The actual functions alpha and beta are provided by an auxiliary
 * class, the HHGate, which communicates with the channel using 
 * messages. The idea is that all copies of a channel share the
 * same gate, thus saving a great deal of space. It also makes it 
 * possible to cleanly change the parameters of all the channels of
 * a give class, all at once. Should one want to mutate a subset
 * of channels, they just need to set up separate gates.
 *
 * The communication between channel and gates is done through a
 * return message. The channel sends a request for a calculation to
 * the gate, and it requires that the gate computation is done
 * and returned immediately to the originating channel. This kind
 * of message is illegal to send between nodes, and the idea is that
 * a local copy of the gate should be made on each node if needed.
 *
 */

typedef double ( *PFDD )( double, double );

class HHChannel
{
#ifdef DO_UNIT_TESTS
	friend void testHHChannel();
#endif // DO_UNIT_TESTS
	public:
		HHChannel()
			: Xpower_( 0.0 ), Ypower_( 0.0 ), Zpower_( 0.0 ),
			Vm_( 0.0 ), conc_( 0.0 ),
			Gbar_( 0.0 ), Ek_( 0.0 ),
			instant_( 0 ),
			Gk_( 0.0 ), Ik_( 0.0 ),
			X_( 0.0 ), Y_( 0.0 ), Z_( 0.0 ),
                        xInited_( false ), yInited_( false ), zInited_( false ),
			g_( 0.0 ),
			useConcentration_( 0 )
		{
			;
		}
		
		virtual ~HHChannel() { ; }

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////

		static void setGbar( const Conn* c, double Gbar );
		static double getGbar( Eref );
		static void setEk( const Conn* c, double Ek );
		static double getEk( Eref );
		static void setXpower( const Conn* c, double Xpower );
		static double getXpower( Eref );
		static void setYpower( const Conn* c, double Ypower );
		static double getYpower( Eref );
		static void setZpower( const Conn* c, double Zpower );
		static double getZpower( Eref );
		static void setSurface( const Conn* c, double Surface );
		static double getSurface( Eref );
		static void setInstant( const Conn* c, int Instant );
		static int getInstant( Eref );
		static void setGk( const Conn* c, double Gk );
		static double getGk( Eref );
		// Ik is read-only
		static double getIk( Eref );
		static void setX( const Conn* c, double X );
		static double getX( Eref );
		static void setY( const Conn* c, double Y );
		static double getY( Eref );
		static void setZ( const Conn* c, double Z );
		static double getZ( Eref );
		static void setUseConcentration( const Conn* c, int value );
		static int getUseConcentration( Eref );

		void innerSetXpower( Eref e, double Xpower );
		void innerSetYpower( Eref e, double Ypower );
		void innerSetZpower( Eref e, double Zpower );

		/////////////////////////////////////////////////////////////
		// Dest function definitions
		/////////////////////////////////////////////////////////////

		/**
		 * processFunc handles the update and calculations every
		 * clock tick. It first sends the request for evaluation of
		 * the gate variables to the respective gate objects and
		 * recieves their response immediately through a return 
		 * message. This is done so that many channel instances can
		 * share the same gate lookup tables, but do so cleanly.
		 * Such messages should never go to a remote node.
		 * Then the function does its own little calculations to
		 * send back to the parent compartment through regular 
		 * messages.
		 */
		static void processFunc( const Conn* c, ProcInfo p );
		void innerProcessFunc( Eref e, ProcInfo p );

		/**
		 * Reinitializes the values for the channel. This involves
		 * computing the steady-state value for the channel gates
		 * using the provided Vm from the parent compartment. It 
		 * involves a similar cycle through the gates and then 
		 * updates to the parent compartment as for the processFunc.
		 */
		static void reinitFunc( const Conn* c, ProcInfo p );
		void innerReinitFunc( Eref e, ProcInfo p );

		/**
		 * Assign the local Vm_ to the incoming Vm from the compartment
		 */
		static void channelFunc( const Conn* c, double Vm );

		/**
		 * Assign the local conc_ to the incoming conc from the
		 * concentration calculations for the compartment. Typically
		 * the message source will be a CaConc object, but there
		 * are other options for computing the conc.
		 */
		static void concFunc( const Conn* c, double conc );

		/**
		 * These three functions receive the return values for
		 * gate state variable X,Y,Z and the conductance, as evaluated
		 * by the specified gate.
		 */
		static void xGateFunc( const Conn* c, double X, double g );
		static void yGateFunc( const Conn* c, double Y, double g );
		static void zGateFunc( const Conn* c, double Z, double g );

		static void createGateFunc(
			const Conn* c,
			string gateType,
			IdGenerator idGen );

	protected:
		virtual string chanFinfo( string gateType ) const
		{
			if ( gateType == "X" ) return "xGate";
			else if ( gateType == "Y" ) return "yGate";
			else if ( gateType == "Z" ) return "zGate";
			else assert( 0 );
		}

		virtual string gateFinfo( string gateType ) const { return "gate"; }

		virtual string gateClass( string gateType ) const { return "HHGate"; }

		virtual void lookupXrates( Eref e );
		virtual void lookupYrates( Eref e );
		virtual void lookupZrates( Eref e );

		// Utility function for destroying gate
		static void destroyGate( Eref e, string name );

		static PFDD selectPower( double power);

		/// Exponent for X gate
		double Xpower_;
		/// Exponent for Y gate
		double Ypower_;
		/// Exponent for Z gate
		double Zpower_;
		/// Vm_ is input variable from compartment, used for most rates
		double Vm_;
		/// Conc_ is input variable for Ca-dependent channels.
		double conc_;

		double ( *takeXpower_ )( double, double );
		double ( *takeYpower_ )( double, double );
		double ( *takeZpower_ )( double, double );

	private:
		/// Channel maximal conductance
		double Gbar_;
		/// Reversal potential of channel
		double Ek_;

		/// bitmapped flag for X, Y, Z, to do equil calculation for gate
		int instant_;
		/// Channel actual conductance depending on opening of gates.
		double Gk_;
		/// Channel current
		double Ik_;

		/// State variable for X gate
		double X_;
		/// State variable for Y gate
		double Y_;
		/// State variable for Z gate
		double Z_;
                bool xInited_, yInited_, zInited_; // true when a state variable
                                                       // has been initialized
		/// Internal variable used to calculate conductance
		double g_;	

		/// Flag for use of conc for input to Z gate calculations.
		bool useConcentration_;	

		// Internal variables for return values
		double A_;
		double B_;
		virtual double integrate( double state, double dt );

		static const double EPSILON;
		static const int INSTANT_X;
		static const int INSTANT_Y;
		static const int INSTANT_Z;

		static double power1( double x, double p ) {
			return x;
		}
		static double power2( double x, double p ) {
			return x * x;
		}
		static double power3( double x, double p ) {
			return x * x * x;
		}
		static double power4( double x, double p ) {
			return power2( x * x, p );
		}
		static double powerN( double x, double p );
};

// Used by solver, readcell, etc.
extern const Cinfo* initHHChannelCinfo();

#endif // _HHChannel_h
