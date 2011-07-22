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
namespace moose
{
class Compartment
{
	public:
			Compartment();
			virtual ~Compartment() {;}

			// Value Field access function definitions.
			static void setVm( const Conn* c, double Vm );
			static double getVm( Eref );
			static void setEm( const Conn* c, double Em );
			static double getEm( Eref );
			static void setCm( const Conn* c, double Cm );
			static double getCm( Eref );
			static void setRm( const Conn* c, double Rm );
			static double getRm( Eref );
			static void setRa( const Conn* c, double Ra );
			static double getRa( Eref );
			static void setIm( const Conn* c, double Im );
			static double getIm( Eref );
			static void setInject( const Conn* c, double Inject );
			static double getInject( Eref );
			static void setInitVm( const Conn* c, double initVm );
			static double getInitVm( Eref );
			static void setDiameter( const Conn* c, double diameter );
			static double getDiameter( Eref );
			static void setLength( const Conn* c, double length );
			static double getLength( Eref );
			static void setX0( const Conn* c, double value );
			static double getX0( Eref );
			static void setY0( const Conn* c, double value );
			static double getY0( Eref );
			static void setZ0( const Conn* c, double value );
			static double getZ0( Eref );
			static void setX( const Conn* c, double value );
			static double getX( Eref );
			static void setY( const Conn* c, double value );
			static double getY( Eref );
			static void setZ( const Conn* c, double value );
			static double getZ( Eref );

			// Dest function definitions.
			static void processFunc( const Conn* c, ProcInfo p );
			static void reinitFunc( const Conn* c, ProcInfo p );
			static void initFunc( const Conn* c, ProcInfo p );
			static void initReinitFunc( const Conn* c, ProcInfo p );
			static void channelFunc( const Conn* c, double Gk, double Ek);
			static void raxialFunc(const Conn* c, double Ra, double Vm);
			static void axialFunc(const Conn* c, double Vm);
			static void injectMsgFunc(const Conn* c, double I);
			static void randInjectFunc(const Conn* c, double prob, double I);
			// A utility function
			static bool rangeWarning( 
					const Conn* c, const string& field, double value );

	protected:
			double Ra_;
			double Vm_;
			double Im_;
                        double tmpIm_;
			double A_;
			double B_;

			/**
			 * The innerReinitFunc reinitializes all fields.
			 */
			virtual void innerReinitFunc( Eref e, ProcInfo p );

	private:
			/**
			 * The innerProcessFunc does the object updating and sends out
			 * messages to channels, nernsts, and so on.
			 */
			virtual void innerProcessFunc( Eref e, ProcInfo p );

			/**
			 * The innerInitFunc sends the axial and raxial messages
			 * to other compartments. It has to be executed out of phase
			 * with the process so that all compartments are equivalent and
			 * there is no calling order dependence in the results.
			 */
			virtual void innerInitFunc( Eref e, ProcInfo p );

			/**
			 * Unused function to do another reinit step out of phase
			 * with the main one.
			 */
			virtual void innerInitReinitFunc( Eref e, ProcInfo p );

			/**
			 * innerRaxialFunc handles incoming raxial message data.
			 */
			virtual void innerRaxialFunc( double Ra, double Vm );

			/**
			 * innerAxialFunc handles incoming axial message data.
			 */
			void innerAxialFunc( double Vm );

			//double Vm_;

			double Em_;
			double Cm_;
			double Rm_;
			// double Ra_;
			// double Im_;
			double initVm_;
			double Inject_;
			double diameter_;
			double length_;
			// double A_;
			// double B_;
			double invRm_;
			double sumInject_;
			double x0_;
			double y0_;
			double z0_;
			double x_;
			double y_;
			double z_;
			static const double EPSILON;
};
}

// Used by solver, readcell, etc.
extern const Cinfo* initCompartmentCinfo();

#endif // _COMPARTMENT_H
