/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Reaction_h
#define _Reaction_h
class Reaction
{
	friend class ReactionWrapper;
	public:
		Reaction()
		{
			kf_ = 0.1;
			kb_ = 0.1;
		}


		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static void setKf( const Conn& c, double value );
		static double getKf( const Element* e );
		static void setKb( const Conn& c, double value );
		static double getKb( const Element* e );
		
		///////////////////////////////////////////////////
		// Shared message function definitions
		///////////////////////////////////////////////////
		void innerProcessFunc( Element* e, ProcInfo info );
		static void processFunc( const Conn& c, ProcInfo p );
		void innerReinitFunc();
		static void reinitFunc( const Conn& c, ProcInfo p );
		static void substrateFunc( const Conn& c, double n );
		static void productFunc( const Conn& c, double n );

		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////

		static void scaleKfFunc( const Conn& c, double k );
		static void scaleKbFunc( const Conn& c, double k );

	private:
		double kf_;
		double kb_;
		double A_;
		double B_;
};
#endif // _Reaction_h
