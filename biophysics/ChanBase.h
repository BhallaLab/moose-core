#ifndef _ChanBase_h
#define _ChanBase_h
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
 * The ChanBase is the base class for all ion channel classes in MOOSE.
 * It knows how to communicate with the parent compartment, not much else.
 */

class ChanBase
{
	public:
		ChanBase();
		~ChanBase();

		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////

		void setGbar( double Gbar );
		double getGbar() const;
		void setEk( double Ek );
		double getEk() const;
		void setInstant( int Instant );
		int getInstant() const;
		void setGk( double Gk );
		double getGk() const;
		// Ik is read-only
		double getIk() const;

		/////////////////////////////////////////////////////////////
		// Dest function definitions
		/////////////////////////////////////////////////////////////

		/**
		 * Assign the local Vm_ to the incoming Vm from the compartment
		 */
		void handleVm( double Vm );

		/////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	protected:
		/// Vm_ is input variable from compartment, used for most rates
		double Vm_;

	private:
		/// Channel maximal conductance
		double Gbar_;
		/// Reversal potential of channel
		double Ek_;

		/// Channel actual conductance depending on opening of gates.
		double Gk_;
		/// Channel current
		double Ik_;

		/// Internal variable used to calculate conductance
		double g_;	
};


#endif // _ChanBase_h
