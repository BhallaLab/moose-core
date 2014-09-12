/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/
#ifndef _ChanBase_h
#define _ChanBase_h

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
		virtual void vSetGbar( double Gbar ) = 0;
		double getGbar() const;
		virtual double vGetGbar() const = 0;
		void setEk( double Ek );
		virtual void vSetEk( double Ek ) = 0;
		double getEk() const;
		virtual double vGetEk() const = 0;
		void setGk( double Gk );
		virtual void vSetGk( double Gk ) = 0;
		double getGk() const;
		virtual double vGetGk() const = 0;
		void setIk( double Ic );
		virtual void vSetIk( double Ik ) = 0;
		double getIk() const;
		virtual double vGetIk() const = 0;

		/////////////////////////////////////////////////////////////
		// Dest function definitions
		/////////////////////////////////////////////////////////////

		/**
		 * Assign the local Vm_ to the incoming Vm from the compartment
		 */
		void handleVm( double Vm );
		virtual void vHandleVm( double Vm ) = 0;

		/////////////////////////////////////////////////////////////
		/**
		 * Takes over the Process and reinit functions of all derived
		 * classes.
		 */
		void process( const Eref& e, const ProcPtr info );
		void reinit( const Eref& e, const ProcPtr info );
		virtual void vProcess( const Eref& e, const ProcPtr info ) = 0;
		virtual void vReinit( const Eref& e, const ProcPtr info ) = 0;

		/////////////////////////////////////////////////////////////
		static SrcFinfo1< double >* permeability();
		static SrcFinfo2< double, double >* channelOut();
		static SrcFinfo1< double >* IkOut();
		/////////////////////////////////////////////////////////////
		/// Specify the Class Info static variable for initialization.
		static const Cinfo* initCinfo();
	private:
};




#endif // _ChanBase_h
