/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../biophysics/Compartment.h"
#include "../biophysics/SymCompartment.h"
#include "SigNeur.h"

static const double PI = 3.1415926535;


/**
 * For now don't deal with taper
 */
unsigned int numSegments( Id compt, double lambda )
{
	double length = 0.0;
	// double dia = 0.0;
	assert( compt.good() );
	bool ret = get< double >( compt.eref(), "length", length );
	assert( ret );
	// ret = get< double >( compt.eref(), "diameter", dia );
	assert( ret );
	assert( length > 0.0 && lambda > 0.0 );
	return ( 1 + length / lambda );
}

// Count signaling compts, also subdivide for long dend compts.
void SigNeur::assignSignalingCompts()
{
	numSoma_ = 0;
	numDend_ = 0;
	numSpine_ = 0;
	numNeck_ = 0;
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		if ( i->category == SOMA ) {
			unsigned int ns = numSegments( i->compt, lambda_ );
			i->sigStart = numSoma_;
			i->sigEnd = numSoma_ = numSoma_ + ns;
		} else if ( i->category == DEND ) {
			unsigned int ns = numSegments( i->compt, lambda_ );
			i->sigStart = numDend_;
			i->sigEnd = numDend_ = numDend_ + ns;
			// cout << " " << numSegments;
		} else if ( i->category == SPINE ) {
			i->sigStart = numSpine_;
			i->sigEnd = ++numSpine_;
		} else if ( i->category == SPINE_NECK ) {
			++numNeck_;
		}
	}
	// cout << endl;
	// Now reposition the indices for the dends and spines, depending on
	// the numerical methods.
	// if ( dendMethod_ == "rk5" && somaMethod_ == dendMethod_ ) {
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == DEND ) {
				i->sigStart += numSoma_;
				i->sigEnd += numSoma_;
			}
		}
	// }
	// if ( dendMethod_ == "rk5" && spineMethod_ == dendMethod_ ) {
		unsigned int offset = numSoma_ + numDend_;
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == SPINE ) {
				i->sigStart += offset;
				i->sigEnd += offset;
			}
		}
	// }
//	reportTree();
}

/**
 * Print out some diagnostics about the tree subdivisions.
 */
void SigNeur::reportTree( 
	vector< double >& volume, vector< double >& xByL )
{
	cout << "SigNeur: Tree size = " << tree_.size() << 
		", s=" << numSoma_ << 
		", d=" << numDend_ << 
		", sp=" << numSpine_ <<
		", neck=" << numNeck_ << endl;

	/*
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		assert( i->parent < tree_.size() );
		cout << "pa: " << tree_[ i->parent ].compt.path() << ", el: " << i->compt.path() << ", sig: " << i->sigModel.path() << "[" << i->sigStart << ".." << i->sigEnd << "]\n";
		for ( unsigned int j = i->sigStart; j < i ->sigEnd; ++j )
			cout << "	sig[" << j << "]: vol=" << volume[j] << 
				", xByL=" << xByL[j] << endl;
	}
	*/
}

Id SigNeur::findSoma( const vector< Id >& compts )
{
	double maxDia = 0;
	Id maxCompt;
	vector< Id > somaCompts; // Theoretically possible to have an array.
	for ( vector< Id >::const_iterator i = compts.begin(); 
		i != compts.end(); ++i )
	{
		string className = i->eref()->className();
		if ( className == "Compartment" || className == "SymCompartment" ) {
			string name = i->eref().e->name();
			if ( name == "soma" || name == "Soma" || name == "SOMA" )
				somaCompts.push_back( *i );
			double dia;
			get< double >( i->eref(), "diameter", dia );
			if ( dia > maxDia )
				maxCompt = *i;
		}
	}
	if ( somaCompts.size() == 1 ) // First, go by name.
		return somaCompts[0];
	if ( somaCompts.size() == 0 & maxCompt.good() ) //if no name, use maxdia
		return maxCompt;
	if ( somaCompts.size() > 1 ) { // Messy but unlikely cases.
		if ( maxCompt.good() ) {
			if ( find( somaCompts.begin(), somaCompts.end(), maxCompt ) != somaCompts.end() )
				return maxCompt;
			else
				cout << "Error, soma '" << somaCompts.front().path() << 
					"' != biggest compartment '" << maxCompt.path() << 
					"'\n";
		}
		return somaCompts[0]; // Should never happen, but an OK response.
	}
	cout << "Error: SigNeur::findSoma failed to find soma\n";
	return Id();
}

void SigNeur::buildTree( Id soma, const vector< Id >& compts )
{
	const Finfo* axialFinfo;
	const Finfo* raxialFinfo;
	if ( soma.eref().e->cinfo() == initSymCompartmentCinfo() ) {
		axialFinfo = initSymCompartmentCinfo()->findFinfo( "raxial1" );
		raxialFinfo = initSymCompartmentCinfo()->findFinfo( "raxial2" );
	} else {
		axialFinfo = initCompartmentCinfo()->findFinfo( "axial" );
		raxialFinfo = initCompartmentCinfo()->findFinfo( "raxial" );
	}
	assert( axialFinfo != 0 );
	assert( raxialFinfo != 0 );
	
	// Soma may be in middle of messaging structure for cell, so we need
	// to traverse both ways. But nothing below soma should 
	// change direction in the traversal.
	innerBuildTree( 0, soma.eref(), soma.eref(), 
		axialFinfo->msg(), raxialFinfo->msg() );
	// innerBuildTree( 0, soma.eref(), soma.eref(), raxialFinfo->msg() );
}

void SigNeur::innerBuildTree( unsigned int parent, Eref paE, Eref e, 
	int msg1, int msg2 )
{
	unsigned int paIndex = tree_.size();
	TreeNode t( e.id(), parent, guessCompartmentCategory( e ) );
	tree_.push_back( t );
	// cout << e.name() << endl;
	Conn* c = e->targets( msg1, e.i );

	// Things are messy here because src/dest directions are flawed
	// in Element::targets.
	// The parallel moose fixes this mess, simply by checking against
	// which the originating element is. Here we need to do the same
	// explicitly.
	for ( ; c->good(); c->increment() ) {
		Eref tgtE = c->target();
		if ( tgtE == e )
			tgtE = c->source();
		if ( !( tgtE == paE ) ) {
			// cout << "paE=" << paE.name() << ", e=" << e.name() << ", msg1,2= " << msg1 << "," << msg2 << ", src=" << c->source().name() << ", tgt= " << tgtE.name() << endl;
			innerBuildTree( paIndex, e, tgtE, msg1, msg2 );
		}
	}
	delete c;
	c = e->targets( msg2, e.i );
	for ( ; c->good(); c->increment() ) {
		Eref tgtE = c->target();
		if ( tgtE == e )
			tgtE = c->source();
		if ( !( tgtE == paE ) ) {
			// cout << "paE=" << paE.name() << ", e=" << e.name() << ", msg1,2= " << msg1 << "," << msg2 << ", src=" << c->source().name() << ", tgt= " << tgtE.name() << endl;
			innerBuildTree( paIndex, e, tgtE, msg1, msg2 );
		}
	}
	delete c;
}


/**
 * 	This function uses naming heuristics to decide which signaling model
 * 	belongs in which compartment. By default, it puts spine signaling in all
 * 	compartments which have 'spine' in the name, except for those which
 * 	have 'neck' or 'shaft as well. It puts soma signaling in compartments
 * 	with soma in the name, and dend signaling everywhere else.
 * 	In addition, it has two optional
 * 	fields to use: dendInclude and dendExclude. If dendInclude is set,
 * 	then it only puts dends in the specified compartments.
 * 	Whether or not dendInclude is set, dendExclude eliminates dends from
 * 	the specified compartments.
 */

CompartmentCategory SigNeur::guessCompartmentCategory( Eref e )
{
	if ( e.e->name().find( "spine" ) != string::npos ||
		e.e->name().find( "Spine" ) != string::npos ||
		e.e->name().find( "SPINE" ) != string::npos )
	{
		if ( e.e->name().find( "neck" ) != string::npos ||
			e.e->name().find( "Neck" ) != string::npos ||
			e.e->name().find( "NECK" ) != string::npos ||
			e.e->name().find( "shaft" ) != string::npos ||
			e.e->name().find( "Shaft" ) != string::npos ||
			e.e->name().find( "SHAFT" ) != string::npos
		)
			return SPINE_NECK;
		else
			return SPINE;
	}
	if ( e.e->name().find( "soma" ) != string::npos ||
		e.e->name().find( "Soma" ) != string::npos ||
		e.e->name().find( "SOMA" ) != string::npos)
	{
		return SOMA;
	}
	CompartmentCategory ret = EMPTY;
	if ( dendInclude_ == "" )
		ret = DEND;
	else if ( e.e->name().find( dendInclude_ ) != string::npos ) 
		ret = DEND;

	if ( ret == DEND && ( dendExclude_.length() > 0 ) && 
		e.e->name().find( dendExclude_ ) != string::npos ) 
		return EMPTY;

	return ret;
}

