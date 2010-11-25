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
		Reaction();


		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static void setRawKf( const Conn* c, double value );
		static double getRawKf( Eref e );
		static void setRawKb( const Conn* c, double value );
		static double getRawKb( Eref e );
		static void setKf( const Conn* c, double value );
		static double getKf( Eref e );
		static void setKb( const Conn* c, double value );
		static double getKb( Eref e );
		static double getX( Eref e );
		static void setX( const Conn* c, double value );
		static double getY( Eref e );
		static void setY( const Conn* c, double value );
		static string getColor( Eref e );
		static void setColor( const Conn* c, string value );
		static string getBgColor( Eref e );
		static void setBgColor( const Conn* c, string value );
		
		///////////////////////////////////////////////////
		// Shared message function definitions
		///////////////////////////////////////////////////
		void innerProcessFunc( Eref e, ProcInfo info );
		static void processFunc( const Conn* c, ProcInfo p );
		void innerReinitFunc();
		static void reinitFunc( const Conn* c, ProcInfo p );
		static void substrateFunc( const Conn* c, double n );
		static void productFunc( const Conn* c, double n );

		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////

		static void scaleKfFunc( const Conn* c, double k );
		static void scaleKbFunc( const Conn* c, double k );

		/**
 		 * Ratio is ratio of new vol to old vol.
 		 * Kf, Kb have units of 1/(conc^(order-1) * sec )
 		 * new conc = old conc / ratio.
 		 * so kf = old_kf * ratio^(order-1)
 		 */
		static void rescaleRates( const Conn* c, double ratio );

	private:
		double kf_;
		double kb_;
		double A_;
		double B_;
		double x_;
		double y_;
		string xtree_textfg_req_;
		string xtree_fg_req_;

};

// Used by the solver
extern const Cinfo* initReactionCinfo();

#endif // _Reaction_h
