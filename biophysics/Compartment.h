/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _COMPARTMENT_H
#define _COMPARTMENT_H

/**
 * The Compartment class sets up an asymmetric compartment for
 * branched nerve calculations. Handles electronic structure and
 * also channels. This is not a particularly efficient way of doing
 * this, so we should use a solver for any substantial calculations.
 */
class Compartment
{
	public:
			Compartment( )
			{
					Vm_ = -0.06;
					Em_ = -0.06;
					Cm_ = 1.0;
					Rm_ = 1.0;
					invRm_ = 1.0;
					Ra_ = 1.0;
					Im_ = 0.0;
					Inject_ = 0.0;
					sumInject_ = 0.0;
					initVm_ = -0.06;
					A_ = 0.0;
					B_ = 0.0;
			}
			
			// Value Field access function definitions.
			static void setVm( const Conn& c, double Vm );
			static double getVm( const Element* );
			static void setEm( const Conn& c, double Em );
			static double getEm( const Element* );
			static void setCm( const Conn& c, double Cm );
			static double getCm( const Element* );
			static void setRm( const Conn& c, double Rm );
			static double getRm( const Element* );
			static void setRa( const Conn& c, double Ra );
			static double getRa( const Element* );
			static void setIm( const Conn& c, double Im );
			static double getIm( const Element* );
			static void setInject( const Conn& c, double Inject );
			static double getInject( const Element* );
			static void setInitVm( const Conn& c, double initVm );
			static double getInitVm( const Element* );
			static void setDiameter( const Conn& c, double diameter );
			static double getDiameter( const Element* );
			static void setLength( const Conn& c, double length );
			static double getLength( const Element* );
			static void setX( const Conn& c, double value );
			static double getX( const Element* );
			static void setY( const Conn& c, double value );
			static double getY( const Element* );
			static void setZ( const Conn& c, double value );
			static double getZ( const Element* );

			// Dest function definitions.
			static void processFunc( const Conn& c, ProcInfo p );
			static void reinitFunc( const Conn& c, ProcInfo p );
			static void initFunc( const Conn& c, ProcInfo p );
			static void dummyInitFunc( const Conn& c, ProcInfo p );
			static void channelFunc( const Conn& c, double Gk, double Ek);
			static void raxialFunc(const Conn& c, double Ra, double Vm);
			static void axialFunc(const Conn& c, double Vm);
			static void injectMsgFunc(const Conn& c, double I);
			static void randInjectFunc(const Conn& c, double prob, double I);
			// A utility function
			static bool rangeWarning( 
					const Conn& c, const string& field, double value );

	private:
			void innerProcessFunc( Element* e, ProcInfo p );
			void innerReinitFunc( Element* e, ProcInfo p );
			void innerRaxialFunc( double Ra, double Vm );
			void innerAxialFunc( double Vm );

			double Vm_;
			double Em_;
			double Cm_;
			double Rm_;
			double Ra_;
			double Im_;
			double initVm_;
			double Inject_;
			double diameter_;
			double length_;
			double A_;
			double B_;
			double invRm_;
			double sumInject_;
			double x_;
			double y_;
			double z_;
			static const double EPSILON;
};

#endif // _COMPARTMENT_H
