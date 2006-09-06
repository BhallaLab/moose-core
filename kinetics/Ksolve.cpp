/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "RateTerm.h"
#include "Ksolve.h"

void Ksolve::setPath( const string& path, Element* wrapper )
{
	vector< Element* > ret;
	vector< Element* >::iterator i;
	Field molSolve( wrapper, "molSolve" );
	Field bufSolve( wrapper, "bufSolve" );
	Field sumTotSolve( wrapper, "sumtotSolve" );
	Field enzSolve( wrapper, "enzSolve" );
	Field mmEnzSolve( wrapper, "mmEnzSolve" );
	Field reacSolve( wrapper, "reacSolve" );
	Element::startFind( path, ret );

	// Build up the segment array, that counts # of targets of
	// each type.

	const Cinfo* molci = Cinfo::find( "Molecule" );
	const Cinfo* reacci = Cinfo::find( "Reaction" );
	const Cinfo* enzci = Cinfo::find( "Enzyme" );
	vector< Element* > varMols;
	vector< Element* > bufMols;
	vector< Element* > sumTotMols;
	vector< Element* > reacs;
	vector< Element* > enzs;
	vector< Element* > mmEnzs;
	int mode;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		// Would really prefer an 'isa' operation.
		if ( ( *i )->cinfo() == molci ) {
			if ( Ftype1< int >::get( *i, "mode", mode ) ) {
				if ( mode == 0 ) { //  Variables
					varMols.push_back( *i );
				} else if ( mode == 4 ) { // buffering
					bufMols.push_back( *i );
				} else { // sumtotals
					sumTotMols.push_back( *i );
				}
			}
		} else if ( ( *i )->cinfo() == enzci ) {
			if ( Ftype1< int >::get( *i, "mode", mode ) ) {
				if ( mode == 0 ) // regular
					enzs.push_back( *i );
				else
					mmEnzs.push_back( *i );
			}
		} else if ( ( *i )->cinfo() == reacci ) {
			reacs.push_back( *i );
		}
	}
	vector< unsigned long > segments( 3, 0 );
	bufOffset_ = segments[0] = varMols.size();
	sumTotOffset_ = segments[1] = segments[0] + bufMols.size();
	segments[2] = segments[1] + sumTotMols.size();
	unsigned long nMols = 
		varMols.size() + bufMols.size() + sumTotMols.size();
	S_.resize( nMols );
	Sinit_.resize( nMols );
	molSolve->resize( molSolve.getElement(), segments );
	segments.resize( 1, 0 );
	segments[0] = enzs.size();
	enzSolve->resize( enzSolve.getElement(), segments );
	segments[0] = mmEnzs.size();
	mmEnzSolve->resize( mmEnzSolve.getElement(), segments );
	segments[0] = reacs.size();
	reacSolve->resize( reacSolve.getElement(), segments );

	// Now that it is allocated, go out and connect them.
	for ( i = varMols.begin(); i != varMols.end(); i++ )
		zombify( *i, molSolve );
	for ( i = bufMols.begin(); i != bufMols.end(); i++ )
		zombify( *i, bufSolve );
	for ( i = sumTotMols.begin(); i != sumTotMols.end(); i++ )
		zombify( *i, sumTotSolve );
	for ( i = enzs.begin(); i != enzs.end(); i++ )
		zombify( *i, enzSolve );
	for ( i = mmEnzs.begin(); i != mmEnzs.end(); i++ )
		zombify( *i, mmEnzSolve );
	for ( i = reacs.begin(); i != reacs.end(); i++ )
		zombify( *i, reacSolve );
}

void Ksolve::zombify( Element* e, Field& solveSrc )
{
	Field f( e, "process" );
	if ( !f.dropAll() ) {
		cerr << "Error: Failed to delete process message into " <<
			e->path() << "\n";
	}
	Field ms( e, "solve" );
	if ( !solveSrc.add( ms ) ) {
		cerr << "Error: Failed to add solve message from solver " <<
			solveSrc.path() << " to zombie " << e->path() << "\n";
	}
	Field update( e, "nInit" );
	// This fills in all the rate and other terms.
	e->solverUpdate( update.getFinfo(), SOLVER_SET );
}

