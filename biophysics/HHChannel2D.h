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
 * The communication between channel and gates is done through a
 * return message. The channel sends a request for a calculation to
 * the gate, and it requires that the gate computation is done
 * and returned immediately to the originating channel. This kind
 * of message is illegal to send between nodes, and the idea is that
 * a local copy of the gate should be made on each node if needed.
 *
 */

class HHChannel2D: public HHChannel
{
#ifdef DO_UNIT_TESTS
	friend void testHHChannel2D();
#endif // DO_UNIT_TESTS

	public:
		HHChannel2D()
		{ ; }

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////
		static void setUseConcentration( const Conn* c, int value );
		static int getUseConcentration( Eref );
		static void setXindex( const Conn* c, string index );
		static string getXindex( Eref e );
		static void setYindex( const Conn* c, string index );
		static string getYindex( Eref e );
		static void setZindex( const Conn* c, string index );
		static string getZindex( Eref e );

		/////////////////////////////////////////////////////////////
		// Dest function definitions
		/////////////////////////////////////////////////////////////
		/**
		 * Assign the local conc_ to the incoming conc from the
		 * concentration calculations for the compartment. Typically
		 * the message source will be a CaConc object, but there
		 * are other options for computing the conc.
		 */
		static void conc2Func( const Conn* c, double conc );

	protected:
		unsigned int dimension( string gateType ) const;
		virtual string chanFinfo( string gateType ) const;
		virtual string gateFinfo( string gateType ) const;
		virtual string gateClass( string gateType ) const;

		virtual void lookupXrates( Eref e );
		virtual void lookupYrates( Eref e );
		virtual void lookupZrates( Eref e );
		
	private:
		void innerSetXindex( Eref e, string Xindex );
		void innerSetYindex( Eref e, string Yindex );
		void innerSetZindex( Eref e, string Zindex );
		
		static int dependency( string index, unsigned int dim );
		
		double conc2_;
		
		string Xindex_;
		string Yindex_;
		string Zindex_;
		
		int Xdep0_;
		int Xdep1_;
		int Ydep0_;
		int Ydep1_;
		int Zdep0_;
		int Zdep1_;
};

// Used by solver, readcell, etc.
extern const Cinfo* initHHChannel2DCinfo();

#endif // _HHChannel2D_h
