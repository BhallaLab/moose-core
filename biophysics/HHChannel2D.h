#ifndef _HHChannel2D_h
#define _HHChannel2D_h
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
 * In this version, the Channel contains the gates and refers to them
 * through pointers. When a copy of the Channel is made, then the original
 * pointer is used. Only the original Channel can modify the Gate.
 * If the original Channel is deleted before the copies are, Bad Things
 * will occur.
 * There must be an instance of the original Channel and its Gates on
 * every node.
 *
 * In HHChannel2D, there are three possible arguments to each gate:
 * The Vm, a conc and a conc2. Two are used at a time to look up a
 * 2-D array.
 */

typedef double ( *PFDD )( double, double );
class HHChannel2D: public ChanBase
{
#ifdef DO_UNIT_TESTS
	friend void testHHChannel2D();
#endif // DO_UNIT_TESTS

	public:
		HHChannel2D();

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////
		void setUseConcentration( int value );
		int getUseConcentration( );
		void setXindex( string index );
		string getXindex() const;
		void setYindex( string index );
		string getYindex() const;
		void setZindex( string index );
		string getZindex() const;

		void setXpower( const Eref& e, const Qinfo* q, double Xpower );
		double getXpower( const Eref& e, const Qinfo* q ) const;
		void setYpower( const Eref& e, const Qinfo* q, double Ypower );
		double getYpower( const Eref& e, const Qinfo* q ) const;
		void setZpower( const Eref& e, const Qinfo* q, double Zpower );
		double getZpower( const Eref& e, const Qinfo* q ) const;
		void setInstant( int Instant );
		int getInstant() const;
		void setX( double X );
		double getX() const;
		void setY( double Y );
		double getY() const;
		void setZ( double Z );
		double getZ() const;

		/// Access function used for the X gate. The index is ignored.
		HHGate2D* getXgate( unsigned int i );
		/// Access function used for the Y gate. The index is ignored.
		HHGate2D* getYgate( unsigned int i );
		/// Access function used for the Z gate. The index is ignored.
		HHGate2D* getZgate( unsigned int i );
		/// Dummy assignment function for the number of gates.
		void setNumGates( unsigned int num );

		/**
		 * Access function for the number of Xgates. Gives 1 if present,
		 * otherwise 0.
		 */
		unsigned int getNumXgates() const;
		/// Returns 1 if Y gate present, otherwise 0
		unsigned int getNumYgates() const;
		/// Returns 1 if Z gate present, otherwise 0
		unsigned int getNumZgates() const;

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
		 * Assign the local conc_ to the incoming conc from the
		 * concentration calculations for the compartment. Typically
		 * the message source will be a CaConc object, but there
		 * are other options for computing the conc.
		 */
		void conc( double conc );
		void conc2( double conc );

		double ( *takeXpower_ )( double, double );
		double ( *takeYpower_ )( double, double );
		double ( *takeZpower_ )( double, double );

		/**
		 * Function for safely creating each gate, identified by strings
		 * as X, Y and Z. Will only work on a new channel, not on a
		 * copy. The idea is that the gates are always referred to the
		 * original 'library' channel, and their contents cannot be touched
		 * except by the original.
		 */
		void createGate( const Eref& e, const Qinfo* q, string gateType );

		/// Inner utility function for creating the gate.
		void innerCreateGate(
			 const string& gateName,
			HHGate2D** gatePtr, Id chanId, Id gateId );

		/// Returns true if channel is original, false if copy.
		bool checkOriginal( Id chanId ) const;

		/**
		 * Utility function for destroying gate. Works only on original
		 * HHChannel. Somewhat dangerous, should never be used after a 
		 * copy has been made as the pointer of the gate will be in use
		 * elsewhere.
		 */
		void destroyGate( const Eref& e, const Qinfo* q, string gateType );

		/**
		 * Inner utility for destroying the gate
		 */
		void innerDestroyGate( const string& gateName,
			HHGate2D** gatePtr, Id chanId );

		/**
		 * Utility for altering gate powers
		 */
		bool setGatePower( const Eref& e, const Qinfo* q, double power,
			double* assignee, const string& gateType );

		static PFDD selectPower( double power);

		static const Cinfo* initCinfo();
		
	private:
		const double* dependency( string index, unsigned int dim );
		double integrate( double state, double dt, double A, double B );

		double Xpower_; /// Exponent for X gate
		double Ypower_; /// Exponent for Y gate
		double Zpower_; /// Exponent for Z gate

		/// bitmapped flag for X, Y, Z, to do equil calculation for gate
		int instant_;	
		double X_;	 /// State variable for X gate
		double Y_;	 /// State variable for Y gate
		double Z_;	 /// State variable for Z gate

		/**
		 * true when the matching state variable has been initialized
		 */
        bool xInited_, yInited_, zInited_; 

		double g_;	/// Internal variable used to calculate conductance
		
		double conc_;
		double conc2_;
		
		string Xindex_;
		string Yindex_;
		string Zindex_;

		const double* Xdep0_;
		const double* Xdep1_;
		const double* Ydep0_;
		const double* Ydep1_;
		const double* Zdep0_;
		const double* Zdep1_;
		
		/*
		int Xdep0_;
		int Xdep1_;
		int Ydep0_;
		int Ydep1_;
		int Zdep0_;
		int Zdep1_;
		*/

		/**
		 * HHGate2D data structure for the xGate. This is writable only
		 * on the HHChannel that originally created the HHGate, for others
		 * it must be treated as readonly.
		 */
		HHGate2D* xGate_;
		HHGate2D* yGate_; /// HHGate2D for the yGate
		HHGate2D* zGate_; /// HHGate2D for the zGate

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


#endif // _HHChannel2D_h
