/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_MOL_H
#define _SMOL_MOL_H

// enum MolecState { MSsoln,MSfront,MSback,MSup,MSdown,MSbsoln,MSall,MSnone,MSsome};

/**
 * This class represents a single molecule, that is a particle.
 */
class SmolMol: public SmolSim
{
	public: 
		SmolMol();
		~SmolMol();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setX( const Eref& e, const Qinfo* q, double v );
		double getX( const Eref& e, const Qinfo* q ) const;
		void setY( const Eref& e, const Qinfo* q, double v );
		double getY( const Eref& e, const Qinfo* q ) const;
		void setZ( const Eref& e, const Qinfo* q, double v );
		double getZ( const Eref& e, const Qinfo* q ) const;
		void setState( const Eref& e, const Qinfo* q, MolecState s );
		MolecState getState( const Eref& e, const Qinfo* q ) const;
		unsigned int getSpecies( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		/*
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void reac( double A, double B );
		*/

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		/*
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );
		*/

		static const Cinfo* initCinfo();
	private:
};

#endif	// _SMOL_MOL_H
