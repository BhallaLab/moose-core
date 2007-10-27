/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** 
** This part of the program uses Smoldyn which is developed by 
** Steven Andrews. The Smoldyn code is in a separate subdirectory.
**********************************************************************/

#ifndef _SmoldynHub_h
#define _SmoldynHub_h

// Forward declaration
struct simstruct;

/**
 * SmoldynHub provides an interface between MOOSE and the internal
 * Smoldyn operations and data structures.
 * This class is a wrapper for the Smoldyn simptr class, which
 * is a complete definition of the Smoldyn simulation.
 */
class SmoldynHub
{
	public:
		SmoldynHub();
		///////////////////////////////////////////////////
		// Zombie utility functions
		///////////////////////////////////////////////////
		static SmoldynHub* getHubFromZombie( const Element* e, 
			const Finfo *f, unsigned int& index );
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		void setPos( unsigned int molIndex, double value, 
			unsigned int i, unsigned int dim );
		double getPos( unsigned int molIndex, unsigned int i, 
			unsigned int dim );

		void setPosVector( unsigned int molIndex, 
			const vector< double >& value, unsigned int dim );
		void getPosVector( unsigned int molIndex,
			vector< double >& value, unsigned int dim );

		void setNinit( unsigned int molIndex, unsigned int value );
		unsigned int getNinit( unsigned int molIndex );

		void setNmol( unsigned int molIndex, unsigned int value );
		unsigned int getNmol( unsigned int molIndex );

		void setD( unsigned int molIndex, double value );
		double getD( unsigned int molIndex );

		unsigned int numSpecies() const;
		static unsigned int getNspecies( const Element* e );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void reinitFunc( const Conn& c, ProcInfo info );
		void reinitFuncLocal( Element* e );
		static void processFunc( const Conn& c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );

		static const Finfo* particleFinfo;

	private:
		// A pointer to the entire Smoldyn data structure
		struct simstruct* simptr_;	
};

// Used by the solver
extern const Cinfo* initSmoldynHubCinfo();

#endif // _SmoldynHub_h
