#ifndef _Mg_block_h
#define _Mg_block_h
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

//typedef double ( *PFDD )( double, double );

class Mg_block
{
	public:
		Mg_block();

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////

		static void setKMg_A( const Conn* c, double Gbar );
		static double getKMg_A( Eref );
		static void setKMg_B( const Conn* c, double Ek );
		static double getKMg_B( Eref );
		static void setCMg( const Conn* c, double CMg );
		static double getCMg( Eref );
		static void setIk( const Conn* c, double Ik );
		static double getIk( Eref );
		static void setGk( const Conn* c, double Gk );
		static double getGk( Eref );
		static void setEk( const Conn* c, double Ek );
		static double getEk( Eref );
		static void setZk( const Conn* c, double Zk );
		static double getZk( Eref );
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
		static void origChannelFunc( const Conn* c, double Gk, double Ek );
		
		/**
		 * Assign the local conc_ to the incoming conc from the
		 * concentration calculations for the compartment. Typically
		 * the message source will be a CaConc object, but there
		 * are other options for computing the conc.
		 */
		static void concFunc( const Conn* c, double conc );


	private:
		/// charge
		double Zk_;
		/// 1/eta
		double KMg_A_;
		/// 1/gamma
		double KMg_B_;
		/// [Mg] in mM
		double CMg_;
		double Ik_;
		double Gk_;
		double Ek_;
		double Vm_;
};

// Used by the solver
//extern const Cinfo* initMg_blockCinfo();

#endif // _Mg_block_h
