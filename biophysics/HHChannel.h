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
 * class, the HHGate. The idea is that all copies of a channel share the
 * same gate, thus saving a great deal of space. It also makes it 
 * possible to cleanly change the parameters of all the channels of
 * a give class, all at once. Should one want to mutate a subset
 * of channels, they just need to set up separate gates.
 *
 * HHGates are implemented as a special category of FieldElement, so that
 * they can be accessed as readonly pointers available to the HHChannel. 
 * The FieldElement containing the HHGate appears as a child Element of
 * the HHChannel. The HHChannel Element can be an array; the associated
 * HHGate is a singleton. So there has to be a local copy of the HHGate
 * on each node.
 */

typedef double ( *PFDD )( double, double );

class HHChannel
{
#ifdef DO_UNIT_TESTS
	friend void testHHChannel();
#endif // DO_UNIT_TESTS
	public:
		HHChannel();
		virtual ~HHChannel() { ; }

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////

		void setGbar( double Gbar );
		double getGbar() const;
		void setEk( double Ek );
		double getEk() const;
		void setXpower( double Xpower );
		double getXpower() const;
		void setYpower( double Ypower );
		double getYpower() const;
		void setZpower( double Zpower );
		double getZpower() const;
		void setSurface( double Surface );
		double getSurface() const;
		void setInstant( int Instant );
		int getInstant() const;
		void setGk( double Gk );
		double getGk() const;
		// Ik is read-only
		double getIk() const;
		void setX( double X );
		double getX() const;
		void setY( double Y );
		double getY() const;
		void setZ( double Z );
		double getZ() const;
		void setUseConcentration( int value );
		int getUseConcentration() const;

		void innerSetXpower( double Xpower );
		void innerSetYpower( double Ypower );
		void innerSetZpower( double Zpower );

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
		void process( const Eref& e, ProcPtr p );

		/**
		 * Reinitializes the values for the channel. This involves
		 * computing the steady-state value for the channel gates
		 * using the provided Vm from the parent compartment. It 
		 * involves a similar cycle through the gates and then 
		 * updates to the parent compartment as for the processFunc.
		 */
		void reinit( const Eref& e, ProcPtr p );

		/**
		 * Assign the local Vm_ to the incoming Vm from the compartment
		 */
		void handleVm( double Vm );

		/**
		 * Assign the local conc_ to the incoming conc from the
		 * concentration calculations for the compartment. Typically
		 * the message source will be a CaConc object, but there
		 * are other options for computing the conc.
		 */
		void handleConc( double conc );

		/////////////////////////////////////////////////////////////
		// Gate handling functions
		/////////////////////////////////////////////////////////////
		HHGate* getXgate( unsigned int i );
		HHGate* getYgate( unsigned int i );
		HHGate* getZgate( unsigned int i );

		void setNumGates( unsigned int num );
		unsigned int getNumGates() const;

		void createGate( const Eref& e, const Qinfo* q, string gateType );

		void innerCreateGate( 
			 const string& gateName,
			HHGate** gatePtr, Id chanId,
			HHGate* ( HHChannel::*getGate )( unsigned int ) );
		bool checkOriginal( Id chanId ) const;

		// Utility function for destroying gate
		void destroyGate( string gateType );

		/////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

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

		virtual void lookupXrates() {;}
		virtual void lookupYrates() {;}
		virtual void lookupZrates() {;}


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
		virtual double integrate( double state, double dt, double A, double B );

		HHGate* xGate_;
		HHGate* yGate_;
		HHGate* zGate_;

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


#endif // _HHChannel_h
