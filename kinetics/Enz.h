/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ENZ_H
#define _ENZ_H

class Enz: public EnzBase
{
	public:
		Enz();
		~Enz();

		//////////////////////////////////////////////////////////////////
		// Virtual field stuff to overwrite EnzBase
		//////////////////////////////////////////////////////////////////
		void vSetKm( const Eref& e, double v );
		void vSetKcat( const Eref& e, double v );

		//////////////////////////////////////////////////////////////////
		// functions for setting rates specific to mass-action enz.
		//////////////////////////////////////////////////////////////////
		void setK1( const Eref& e, double v );
		double getK1( const Eref& e ) const;
		void setK2( const Eref& e, double v );
		double getK2( const Eref& e ) const;
		void setRatio( const Eref& e, double v );
		double getRatio( const Eref& e ) const;
		void setConcK1( const Eref& e, double v );
		double getConcK1( const Eref& e ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs, not virtual
		//////////////////////////////////////////////////////////////////
		void setKmK1( double Km, double k1 );

		//////////////////////////////////////////////////////////////////
		
		void setSolver( const Eref& e, ObjId solver );

		static const Cinfo* initCinfo();
	private:
		Stoich* stoich_;
		double k2_;
};

#endif // ENZ_H
