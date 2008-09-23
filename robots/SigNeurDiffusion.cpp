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
#include "../element/Neutral.h"
#include "../kinetics/Molecule.h"
#include "../kinetics/Reaction.h"
#include "SigNeur.h"

static const double PI = 3.1415926535;

void SigNeur::insertDiffusion( Element* base )
{
	static const double EPSILON = 1.0e-20;
	static const Finfo* childListFinfo = 
		initNeutralCinfo()->findFinfo( "childList" );
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* dFinfo = 
		initMoleculeCinfo()->findFinfo( "D" );
	static const Finfo* modeFinfo = 
		initMoleculeCinfo()->findFinfo( "mode" );
	static const Finfo* subFinfo = 
		initReactionCinfo()->findFinfo( "sub" );

	// Traverse all zero index children, find ones that have D > 0
	// Create an array of diffs on these children
	// Connect up to parent using One2OneMap
	// Connect up to next index parent using SimpleConn for now
	// Assign rates.
	if ( base == 0 )
		return;
	
	// Traverse children.
	vector< Id > kids;
	get< vector< Id > >( base, childListFinfo, kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
		insertDiffusion( i->eref().e );

	// Be sure to compete traversal before skipping this element.
	if ( !base->cinfo()->isA( initMoleculeCinfo() ) )
		return;
	
	// Ensure D > 0 
	double D = 0.0;
	bool ret = get< double >( base, dFinfo, D );
	if ( D <= EPSILON )
		return;
	int mode = 0;
	ret = get< int >( base, modeFinfo, mode );
	if ( mode != 0 ) // Should neither be buffered nor sumtotalled.
		return;

	// Create array of diffusion reactions.
	Id baseId = base->id();
	Id childId = Id::childId( baseId );
	Element* diff = Neutral::create( "Reaction", "diff", baseId, childId );

	assert( diff != 0 );
	// Connect up to parent
	ret = baseId.eref().add( reacFinfo->msg(), 
		childId.eref(), subFinfo->msg(), ConnTainer::Default );
	assert( ret );

	// assign rates.
	
}

/**
 * Return the diffusion reaction that is a child of this molecule, if
 * present. Otherwise return 0
 */
Element* findDiff( Element* pa )
{
	static const Finfo* lookupChildFinfo =
		initNeutralCinfo()->findFinfo( "lookupChild" );
	Id ret;
	string temp( "diff" );
	lookupGet< Id, string >( pa, lookupChildFinfo, ret, temp );

	if ( ret.good() )
		return ret();

	return 0;
}

/**
 * Utility function to do the diffusion calculations for the 
 * diffusing molecules m0 and m1, and the reaction diff.
 * Should be called at the point where the diffusion messages are set up.
 */
void diffCalc( double Dscale, Eref m0, Eref m1, Eref diff )
{
	static const Finfo* kfFinfo = 
		initReactionCinfo()->findFinfo( "kf" );
	static const Finfo* kbFinfo = 
		initReactionCinfo()->findFinfo( "kb" );
	static const Finfo* dFinfo = 
		initMoleculeCinfo()->findFinfo( "D" );
	static const Finfo* volFinfo = 
		initMoleculeCinfo()->findFinfo( "volumeScale" );

	double v0;
	double v1;
	double D0;
	double D1;
	bool ret = get< double >( m0, volFinfo, v0 );
	assert( ret && v0 > 0.0 );
	ret = get< double >( m0, dFinfo, D0 );
	assert( ret && D0 > 0.0 );
	ret = get< double >( m1, volFinfo, v1 );
	assert( ret && v1 > 0.0 );
	ret = get< double >( m1, dFinfo, D1 );
	assert( ret && D1 > 0.0 );
	double D = Dscale * ( D0 + D1 ) / 2.0;

	// volscale is conversion factor from # to uM: # / volscale = uM.
	// We need concs in #/m^3 to be consistent in unit terms.
	// #/( volscale) = 6e20 / m^3
	//
	// ( 1000 * # / 6e23 ) / vol(in m^3) = uM
	// # / ( 6e20 * vol ) = uM so volscale = 6e20 * vol
	// or vol = volscale / 6e20
	v0 /= 6.0e20;
	v1 /= 6.0e20;

	double kf = 0.0; // kf already set to Xarea / len
	ret = get< double >( diff, kfFinfo, kf );
	assert( ret && kf > 0.0 );
	kf *= D / v0;
	ret = set< double >( diff, kfFinfo, kf );
	assert( ret );

	double kb = 0.0; // kb already set to Xarea / len
	ret = get< double >( diff, kbFinfo, kb );
	assert( ret && kb > 0.0 );
	kb *= D / v1;
	ret = set< double >( diff, kbFinfo, kb );
	assert( ret );
}

/**
 * The first diffusion reaction (i.e., the one on sigStart) is the one
 * that crosses electrical compartment junctions. 
 *
 * For starters, we simply set the diameter at this and all other
 * diffusion reactions to that of the local electrical compartment.
 *
 * To represent a tapering dend cylinder, we could take the el dia as
 * that at sigStart, and the next compt dia as at sigEnd. But need
 * to rethink for branches.
 *
 * For spines, just use their spineNeck dimensions.
 *
 * For soma, ignore the soma dimensions except within it?
 *
 * It would be cleaner to take the el dia as the middle dia.
 *
 */

void SigNeur::completeSomaDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = somaMap_.begin(); 
		i != somaMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numSoma_; ++j ) {
				if ( junctions[ j ] != UINT_MAX ) {
					assert( junctions[ j ] < i->second->numEntries() );
					Eref e2( diff, j );
					Eref e1( i->second, junctions[ j ] );
					bool ret = e1.add( reacFinfo->msg(), 
						e2, prdFinfo->msg(), 
						ConnTainer::Simple );
					assert( ret );
					Eref e0( i->second, j );
					diffCalc( Dscale_, e0, e1, e2 );
				}
			}
		}
	}
}

// void diffCalc( Eref m0, Eref m1, Eref diff )

void SigNeur::completeDendDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = dendMap_.begin(); 
		i != dendMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numDend_; ++j ) {
				unsigned int tgt = junctions[ j + numSoma_ ];
				if ( tgt < numSoma_ ) { // connect to soma, if mol present
					map< string, Element* >::iterator mol = 
						somaMap_.find( i->first );
					if ( mol != somaMap_.end() ) {
						assert( tgt < mol->second->numEntries() );
						Eref e2( diff, j );
						Eref e1( i->second, tgt );
						bool ret = e1.add( reacFinfo->msg(), 
							e2, prdFinfo->msg(), 
							ConnTainer::Simple );
						assert( ret );
						Eref e0( i->second, j );
						diffCalc( Dscale_, e0, e1, e2 );
					}
				} else if 
					( tgt >= numSoma_ && tgt < numDend_ + numSoma_ ) {
				// Look for other dend diffn compartments. Here tgt
				// is the same molecule, different index.
					tgt -= numSoma_;
					assert( tgt < i->second->numEntries() );
					Eref e2( diff, j );
					Eref e1( i->second, tgt );
					bool ret = e1.add( reacFinfo->msg(), 
						e2, prdFinfo->msg(), 
						ConnTainer::Simple );
					assert( ret );
					Eref e0( i->second, j );
					diffCalc( Dscale_, e0, e1, e2 );
				} else { // Should not connect into spine.
					assert( 0 );
				}
			}
		}
	}
}

void SigNeur::completeSpineDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = spineMap_.begin(); 
		i != spineMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numSpine_; ++j ) {
				unsigned int tgt = junctions[ j + numSoma_ + numDend_ ];
				if ( tgt >= numSoma_ && tgt < numSoma_ + numDend_ ) {
					tgt -= numSoma_; // Fix up offset into dend array.
					// connect to dend, if mol present
					map< string, Element* >::iterator mol = 
						dendMap_.find( i->first );
					if ( mol != dendMap_.end() ) {
						assert( tgt < mol->second->numEntries() );
						Eref e0( i->second, j ); // Parent spine molecule
						Eref e2( diff, j ); // Diffusion object
						Eref e1( mol->second, tgt ); //Target dend molecule
						bool ret = e1.add( reacFinfo->msg(), 
							e2, prdFinfo->msg(), 
							ConnTainer::Simple );
						assert( ret );
						diffCalc( Dscale_, e0, e1, e2 );
					}
				}
			}
		}
	}
}

/**
 * Traverses the cell tree to work out where the diffusion reactions
 * must connect to each other. junction[i] is the index of the 
 * compartment connected to compartment[i]. The indexing of compartments
 * themselves is first the soma block, then the dend block, then the
 * spine block.
 */
void SigNeur::buildDiffusionJunctions( vector< unsigned int >& junctions )
{
	// TreeNode 0 is soma, has parent 0, rest should be a different compt. 
	// The first soma compartment should not connect to anything.
	// The next connects to the first soma, and so on.
	// The diffusion compartments start at the proximal compartment and
	// end at the distal compartment, with respect to the soma.
	// Because of the tree structure, sigStart must connect to the parent,
	// so that every branch point has a diffusion reaction for each branch.
	// Thus sigStart+1 must connect to sigStart, and so on.
	// We ignore diffusive coupling between sibling branches.
	//
	// Need to figure out how to put things at opposite ends of compt.
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		// cout << i - tree_.begin() << "	" << i->compt.path() << ", p=" << i->parent << ", sig=" << i->sigStart << "," << i->sigEnd << endl;;

		
		// First we assign the links within the electrical compartment,
		//  this is easy.
		for ( unsigned int j = i->sigStart + 1; j < i->sigEnd; j++ )
			junctions[ j ] = j - 1;

		//////////////// Now we assign sigSTart ///////////////
		
		if ( i == tree_.begin() ) 
		// Skip sigStart for zero diffusion compartment
			continue;

		TreeNode* tn = &( tree_[ i->parent ] );
		// Attach spine not to neck, but to parent dend.
		if ( i->category == SPINE && 
			tn->category == SPINE_NECK && 
			i->sigStart != i->sigEnd ) {
			TreeNode* dend = &( tree_[ tn->parent ] );
			// Should position based on coordinates and length and numSeg
			// of parent dend. For now put it in middle
			assert( dend->sigStart != dend->sigEnd );
			junctions[ i->sigStart ] = ( dend->sigStart + dend->sigEnd )/ 2;
		} else if ( i->category == DEND && i->sigStart != i->sigEnd ) {
			// Attach it to last sig compt of parent electrical compartment
			// Actually should do something spatial here.
			junctions[ i->sigStart ] = tn->sigEnd - 1;
		}
	}
}

