/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Port
{
	public:
		Port();
		~Port();

		/**
		 * Index to S_ vector (beyond VarPools) into which incoming
		 * molecules should add. inStart is the first entry.
		 */
		unsigned int getInStart() const;

		/**
		 * Index to S_ vector (beyond VarPools) into which incoming
		 * molecules should add. inEnd is the last entry plus one.
		 */
		unsigned int getInEnd() const;

		/**
		 * Index to location in S_ vector (within VarPools) from where 
		 * outgoing molecules begin. 
		 */
		unsigned int getOutStart() const;

		/**
		 * Index to location in S_ vector (within VarPools) one past the
		 * last of the outgoing molecules for this port.
		 */
		unsigned int getOutEnd() const;

		/**
		 * Scaling factor for outgoing rates. This could be a diffusion-
		 * related term, or it could reflect the permeability of the
		 * port.
		 */
		void setScaleOutRate( double rate );

		/**
		 * Scaling factor for outgoing rates. This could be a diffusion-
		 * related term, or it could reflect the permeability of the
		 * port.
		 */
		double getScaleOutRate() const;

		void assignPools( const vector< Id >& pools );

		void findMatchingMolSpecies( const vector< SpeciesId >& other, 
			vector< SpeciesId >& ret );


		static const Cinfo* initCinfo();
	private:
		/**
		 * Index to S_ vector (beyond VarPools) into which incoming
		 * molecules should add. inStart_ is the first entry.
		 */
		unsigned int inStart_;

		/**
		 * Index to S_ vector (beyond VarPools) into which incoming
		 * molecules should add. inEnd_ is the last entry plus one.
		 */
		unsigned int inEnd_;

		/**
		 * Index to location in S_ vector (within VarPools) from where 
		 * outgoing molecules begin. 
		 */
		unsigned int outStart_;

		/**
		 * Index to location in S_ vector (within VarPools) one past the
		 * last of the outgoing molecules for this port.
		 */
		unsigned int outEnd_;

		/**
		 * Scaling factor for outgoing rates. This could be a diffusion-
		 * related term, or it could reflect the permeability of the
		 * port.
		 */
		double scaleOutRate_;

		/**
		 * This specifies molecules available at this port. 
		 * We assume that only one molecule belonging to any given species
		 * is present here. This is a
		 * superset of the mols that will actually be used.
		 */
		map< SpeciesId, Id > availableMols_;


		// Later: Geometry info

		Stoich* parent_;
};
