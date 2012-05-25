/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_COMPARTMENT_H
#define _ZOMBIE_COMPARTMENT_H

/**
 * Zombie object that lets HSolve do its calculations, while letting the user
 * interact with this object as if it were the original object.
 */
class ZombieCompartment
{
	public:
			ZombieCompartment();
			virtual ~ZombieCompartment();

			/*
			 * Value Field access function definitions.
			 */
			
			// Fields handled by solver.
			void setVm( const Eref& e, const Qinfo* q, double Vm );
			double getVm( const Eref& e, const Qinfo* q ) const;
			void setEm( const Eref& e, const Qinfo* q, double Em );
			double getEm( const Eref& e, const Qinfo* q ) const;
			void setCm( const Eref& e, const Qinfo* q, double Cm );
			double getCm( const Eref& e, const Qinfo* q ) const;
			void setRm( const Eref& e, const Qinfo* q, double Rm );
			double getRm( const Eref& e, const Qinfo* q ) const;
			void setRa( const Eref& e, const Qinfo* q, double Ra );
			double getRa( const Eref& e, const Qinfo* q ) const;
			void setIm( const Eref& e, const Qinfo* q, double Im );
			double getIm( const Eref& e, const Qinfo* q ) const;
			void setInject( const Eref& e, const Qinfo* q, double Inject );
			double getInject( const Eref& e, const Qinfo* q ) const;
			void setInitVm( const Eref& e, const Qinfo* q, double initVm );
			double getInitVm( const Eref& e, const Qinfo* q ) const;
			
			// Locally stored fields.
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
			void dummy( const Eref& e, ProcPtr p );
			
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
			void randInject( double prob, double current);

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

			/**
			 * Virtual function to handle Reinit.
			 */
			virtual void innerReinit( const Eref& e, ProcPtr p );

			//////////////////////////////////////////////////////////////////
			// utility funcs
			//////////////////////////////////////////////////////////////////
			static void zombify( Element* solver, Element* orig );
			static void unzombify( Element* zombie );
	
	private:
			HSolve* hsolve_;
			
			double diameter_;
			double length_;
			double x0_;
			double y0_;
			double z0_;
			double x_;
			double y_;
			double z_;
			static const double EPSILON;
			
			void copyFields( moose::Compartment* c );
};

#endif // _ZOMBIE_COMPARTMENT_H
