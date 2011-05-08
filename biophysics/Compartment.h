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
			virtual ~Compartment();

			// Value Field access function definitions.
			void setVm( double Vm );
			double getVm() const;
			void setEm( double Em );
			double getEm() const;
			void setCm( double Cm );
			double getCm() const;
			void setRm( double Rm );
			double getRm() const;
			void setRa( double Ra );
			double getRa() const;
			void setIm( double Im );
			double getIm() const;
			void setInject( double Inject );
			double getInject() const;
			void setInitVm( double initVm );
			double getInitVm() const;
			void setDiameter( double diameter );
			double getDiameter() const;
			void setLength( double length );
			double getLength() const;
			void setX0( double value );
			double getX0() const;
			void setY0( double value );
			double getY0() const;
			void setZ0( double value );
			double getZ0() const;
			void setX( double value );
			double getX() const;
			void setY( double value );
			double getY() const;
			void setZ( double value );
			double getZ() const;

			// Dest function definitions.
			/**
			 * The process function does the object updating and sends out
			 * messages to channels, nernsts, and so on.
			 */
			void process( const Eref& e, ProcPtr p );

			/**
			 * The reinit function reinitializes all fields.
			 */
			void reinit( const Eref& e, ProcPtr p );

			/**
			 * The initProc function is for a second phase of 'process'
			 * operations. It sends the axial and raxial messages
			 * to other compartments. It has to be executed out of phase
			 * with the main process so that all compartments are 
			 * equivalent and there is no calling order dependence in 
			 * the results.
			 */
			void initProc( const Eref& e, ProcPtr p );

			/**
			 * Empty function to do another reinit step out of phase
			 * with the main one. Nothing needs doing there.
			 */
			void initReinit( const Eref& e, ProcPtr p );

			/**
			 * handleChannel handles information coming from the channel
			 * to the compartment
			 */
			void handleChannel( double Gk, double Ek);

			/**
			 * handleRaxial handles incoming raxial message data.
			 */
			void handleRaxial( double Ra, double Vm);

			/**
			 * handleAxial handles incoming axial message data.
			 */
			void handleAxial( double Vm);

			/**
			 * Injects a constantly updated current into the compartment.
			 * Unlike the 'inject' field, this injected current is
			 * applicable only for a single timestep. So this is meant to
			 * be used as the destination of a message rather than as a
			 * one-time assignment.
			 */
			void injectMsg( double current);

			/**
			 * Injects a constantly updated current into the
			 * compartment, with a probability prob. Note that it isn't
			 * the current amplitude that is random, it is the presence
			 * or absence of the current that is probabilistic.
			 */
			void randInject( const Eref& e, const Qinfo* q,
				double prob, double current);

			/**
			 * Dummy function to act as recipient of 'cable' message,
			 * which is just for grouping compartments.
			 */
			void cable();

			/**
			 * A utility function to check for assignment to fields that
			 * must be > 0
			 */
			bool rangeWarning( const string& field, double value );

			/**
			 * Initializes the class info.
			 */
			static const Cinfo* initCinfo();
	protected:
			double Ra_;
			double Vm_;
			double Im_;
			double A_;
			double B_;

	private:
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

#endif // _COMPARTMENT_H
