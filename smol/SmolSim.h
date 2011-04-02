/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_SIM_H
#define _SMOL_SIM_H

class SmolSim
{
	public: 
		SmolSim();
		~SmolSim();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		unsigned int getNumSpecies() const;

		void setPath( const Eref& e, const Qinfo* q, string v );
		string getPath( const Eref& e, const Qinfo* q ) const;

		/*
		Port* getPort( unsigned int i );
		unsigned int getNumPorts() const;
		void setNumPorts( unsigned int num );
		*/

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		/**
		 * Handles incoming messages representing influx of molecules
		void influx( DataId port, vector< double > mol );
 		 */

		/**
		 * Scans through incoming and self molecule list, matching up Ids
		 * to use in the port. Sets up the data structures to do so.
		 * Sends out a message indicated the selected subset.
		void handleAvailableMolsAtPort( DataId port, vector< SpeciesId > mols );
		 */

		/**
		 * Scans through incoming and self molecule list, checking that
		 * all match. Sets up the data structures for the port.
		void handleMatchedMolsAtPort( DataId port, vector< SpeciesId > mols );
		 */

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		void zombifyModel( const Eref& e, const vector< Id >& elist );
		/*
		void allocateObjMap( const vector< Id >& elist );
		void allocateModel( const vector< Id >& elist );
		void zombifyChemCompt( Id compt );

		*/
		unsigned int convertIdToReacIndex( Id id ) const;
		unsigned int convertIdToMolIndex( Id id ) const;
		/*
		unsigned int convertIdToFuncIndex( Id id ) const;

		const double* S() const;
		double* varS();
		const double* Sinit() const;
		double* getY();
		*/

		//////////////////////////////////////////////////////////////////
		// Compute functions
		//////////////////////////////////////////////////////////////////


		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	protected:
		
		/**
		 * This is the path of MOOSE molecules and reactions managed by
		 * this instance of the Smoldyn solver.
		 */
		string path_;


		/**
		 * sim points to the instance of Smoldyn running here.
		 */
		struct simstruct* sim;


		/**
		 * The Ports are interfaces to other solvers by way of a spatial
		 * junction between the solver domains. They manage 
		 * the info about which molecules exchange, 
		 * They are also the connection point for the messages that 
		 * handle the port data transfer.
		 * Each Port connects to exactly one other solver.
		vector< Port > ports_;
		 */
};

#endif	// _SMOL_SIM_H
