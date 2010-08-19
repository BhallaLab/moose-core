/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <sstream>
#include "moose.h"
#include "../element/Neutral.h"
#include "../biophysics/Compartment.h"
#include "../kinetics/Molecule.h"
#include "../kinetics/Reaction.h"
#include "../kinetics/KinCompt.h"
#include "SigNeur.h"

static const double PI = 3.1415926535;
extern Element* findDiff( Element* pa );

void getSigComptSize( const Eref& compt, unsigned int numSeg,
	double& volume, double& xByL )
{
	static const Finfo* diaFinfo = 
		initCompartmentCinfo()->findFinfo( "diameter" );
	static const Finfo* lenFinfo = 
		initCompartmentCinfo()->findFinfo( "length" );
	assert( diaFinfo != 0 );
	assert( lenFinfo != 0 );
	double dia = 0.0;
	double len = 0.0;
	bool ret = get< double >( compt, diaFinfo, dia );
	assert( ret && dia > 0.0 );
	ret = get< double >( compt, lenFinfo, len );
	assert( ret && len > 0.0 );

	if ( numSeg > 0 ) // Zero means there are no diffusion compts.
		len /= numSeg;
	
	// Conversion factor from uM to #/m^3
	volume = len * dia * dia * ( PI / 4.0 );
	xByL = dia * dia * ( PI / 4.0 ) / len;
}

/**
 * setAllVols traverses all signaling compartments  in the model and
 * assigns volumes.
 * This must be called before completeDiffusion because the vols
 * computed here are needed to compute diffusion rates.
 */
void SigNeur::setAllVols()
{
	static const Finfo* volumeFinfo = 
		initKinComptCinfo()->findFinfo( "size" );
	for ( vector< TreeNode >::iterator i = tree_.begin();
		i != tree_.end(); ++i ) {
		double volume;
		double xByL;
		getSigComptSize( i->compt.eref(), i->sigEnd - i->sigStart, volume, xByL );
		Eref e;
		unsigned int offset = 0;
		if ( i->category == SOMA ) {
			e = soma_.eref();
			i->sigModel = soma_;
		} else if ( i->category == DEND ) {
			e = dend_.eref();
			offset = numSoma_;
			i->sigModel = dend_;
		} else if ( i->category == SPINE ) {
			e = spine_.eref();
			offset = numSoma_ + numDend_;
			i->sigModel = spine_;
		} else {
			continue;
		}
		if ( e.e == Element::root() ) 
			continue; // Tihs is when there is no signalling in that compt.

		for ( unsigned int j = i->sigStart; j < i->sigEnd; ++j ) {
			assert( e.e->numEntries() > ( j - offset ) );
			e.i = j - offset;
			assert( e.e->cinfo()->isA( initKinComptCinfo() ) );
			set< double >( e, volumeFinfo, volume );
		}
	}
}

/*
void rescaleDiff( Element* e, double xByL, unsigned int begin, unsigned int end )
{
	static const Finfo* kfFinfo = 
		initReactionCinfo()->findFinfo( "kf" );
	static const Finfo* kbFinfo = 
		initReactionCinfo()->findFinfo( "kb" );
	for ( unsigned int i = begin; i != end; ++i ) {
		Element* de = findDiff( e );
		if ( de ) {
			Eref diff( de, i );
			ret = set< double >( diff, kfFinfo, xByL );
			assert( ret );
			ret = set< double >( diff, kbFinfo, xByL );
			assert( ret );
		}
	}
}
*/

#if 0
/**
 * This figures out dendritic segment dimensions. It assigns the 
 * volumeScale for each signaling compt, and puts Xarea / len into
 * each diffusion element for future use in setting up diffusion rates.
 */
void SigNeur::setComptVols( Eref compt, 
	map< string, Element* >& molMap,
	unsigned int index, unsigned int numSeg )
{
	static const Finfo* diaFinfo = 
		initCompartmentCinfo()->findFinfo( "diameter" );
	static const Finfo* lenFinfo = 
		initCompartmentCinfo()->findFinfo( "length" );
	static const Finfo* volFinfo = 
		initMoleculeCinfo()->findFinfo( "volumeScale" );
	static const Finfo* nInitFinfo = 
		initMoleculeCinfo()->findFinfo( "nInit" );
	static const Finfo* kfFinfo = 
		initReactionCinfo()->findFinfo( "kf" );
	static const Finfo* kbFinfo = 
		initReactionCinfo()->findFinfo( "kb" );
	assert( diaFinfo != 0 );
	assert( lenFinfo != 0 );
	assert( numSeg > 0 );
	double dia = 0.0;
	double len = 0.0;
	bool ret = get< double >( compt, diaFinfo, dia );
	assert( ret && dia > 0.0 );
	ret = get< double >( compt, lenFinfo, len );
	assert( ret && len > 0.0 );
	len /= numSeg;

	// Conversion factor from uM to #/m^3
	double volscale = len * dia * dia * ( PI / 4.0 ) * 6.0e20;
	double xByL = dia * dia * ( PI / 4.0 ) / len;
	
	// Set all the volumes. 
	for ( map< string, Element* >::iterator i = molMap.begin(); 
		i != molMap.end(); ++i ) {
		Eref mol( i->second, index );
		double oldvol;
		double nInit;
		ret = get< double >( mol, volFinfo, oldvol );
		assert( ret != 0 && oldvol > 0.0 );
		ret = get< double >( mol, nInitFinfo, nInit );
		assert( ret );

		ret = set< double >( mol, volFinfo, volscale );
		assert( ret );
		ret = set< double >( mol, nInitFinfo, nInit * volscale / oldvol );
		assert( ret );

		Element* de = findDiff( i->second );
		if ( de ) {
			Eref diff( de, index );
			ret = set< double >( diff, kfFinfo, xByL );
			assert( ret );
			ret = set< double >( diff, kbFinfo, xByL );
			assert( ret );
		}
	}
	// Scale all the reaction rates and enzyme rates.
	foreach ( reacVec.begin(), reacVec.end(), rescaleReac );
	foreach ( enzVec.begin(), enzVec.end(), rescaleEnz );
	/*
	for ( vector< Element* >::iterator i = reacVec.begin(); 
		i != reacVec.end(); ++i ) {
		rescaleReac( volscale / oldvol );
		// Each reac knows how to scale kf and kb according to order.
	}
	*/
}
#endif

#include "../element/Wildcard.h"
void displayElists( const Element* orig, const Element* copy )
{
	vector< Id > olist;
	vector< Id > clist;

	unsigned int okids = allChildren( orig->id(), "", Id::AnyIndex, olist );
	unsigned int ckids = allChildren( copy->id(), "", Id::AnyIndex, clist );
	// Print out names of orig and copy
	// do loop only up to highest common index
	cout << "okids=" << okids << ", ckids=" << ckids << endl;
	cout << "okids=" << orig->name() << ", ckids=" << copy->name() << endl;
	unsigned int max = ( okids < ckids ) ? okids : ckids;
	for ( unsigned int i =0; i < max; ++i ) {
		cout << i << "	" << olist[i].path() << 
			"	" << clist[i].path() << endl;
	}
}

/*
 * This function copies a signaling model. It first traverses the model and
 * inserts any required diffusion reactions into the model. These are
 * created as children of the molecule that diffuses, and are connected
 * up locally for one-half of the diffusion process. Subsequently the
 * system needs to connect up to the next compartment, to set up the 
 * other half of the diffusion. Also the last diffusion reaction
 * needs to have its rates nulled out.
 *
 * Returns the root element of the copy.
 * Kinid is destination of copy
 * proto is prototype
 * Name is the name to assign to the copy.
 * num is the number of duplicates needed.
 */
Element* SigNeur::copySig( Id kinId, Id proto, 
	const string& name, unsigned int num )
{
	Element* ret = 0;
	if ( proto.good() ) {
		Id lib( "/library" );
		/*
		Element* temp = Neutral::create( "Neutral", "temp", libId, 
			Id::childId( libId ) );
		*/
		ret = proto()->copy( lib(), name + ".msgs" );
		assert( ret );
		insertDiffusion( ret ); // Scan through putting in diffs.

		if ( num == 1 ) {
			ret = ret->copy( kinId(), name );
		} else if ( num > 1 ) {
			ret = ret->copyIntoArray( kinId, name, num );
		}
		// cout << "regular copy:\n";
		// displayElists( proto(), ret );
	}
	return ret;
}

/**
 * This variant of copySig makes multiple copies of a signaling model,
 * but does NOT place them into an array. This is a temporary 
 * work-around necessitated because solvers don't know how to deal with
 * parts of arrays. The base element of the whole mess is a neutral
 * so that there is a single handle for the next stage of operations.
 * I would have preferred an array KineticManager, but that gets messy.
 */ 
Element* SigNeur::separateCopySig( Id kinId, Id proto, 
	const string& name, unsigned int num )
{
	Element* ret = 0;
	if ( proto.good() ) {
		Id lib( "/library" );
		/*
		Element* temp = Neutral::create( "Neutral", "temp", libId, 
			Id::childId( libId ) );
		*/
		Element* diffproto = proto()->copy( lib(), name + ".msgs" );
		assert( diffproto );
		insertDiffusion( diffproto ); // Insert reactions for diffusion

		if ( num == 1 ) {
			ret = diffproto->copy( kinId(), name );
		} else if ( num > 1 ) {
			ret = Neutral::create( "Neutral", "spines", 
				kinId, Id::childId( kinId ) );
			for ( unsigned int i = 0; i < num; ++i ) {
				ostringstream s;
				s << name << "[" << i << "]";
				diffproto->copy( ret, s.str() );
			}
		}
	}
	// cout << "separate copy:\n";
	// displayElists( proto(), ret );
	return ret;
}

void SigNeur::makeSignalingModel( Eref me )
{
	// Make kinetic manager on sigNeur
	// Make array copy of soma model.
	// Make array copy of dend model.
	// Make array copy of spine model.
	// Traverse tree, set up diffusion messages.
	// If any are nonzero, activate kinetic manager
	
	Element* kin = Neutral::create( "KineticManager", "kinetics", 
		me.id(), Id::childId( me.id() ) );
 	Id kinId = kin->id();
	// Each of these protos should be a KinCompt or derived class.
	Element* soma = copySig( kinId, somaProto_, "soma", numSoma_ );
	Element* dend = copySig( kinId, dendProto_, "dend", numDend_ );
	Element* spine = 0;
	if ( separateSpineSolvers_ ) 
		spine = separateCopySig( kinId, spineProto_, "spine", numSpine_ );
	else
		spine = copySig( kinId, spineProto_, "spine", numSpine_ );

	if ( soma )
		soma_ = soma->id();
	if ( dend )
		dend_ = dend->id();
	if ( spine )
		spine_ = spine->id();

	// first soma indices, then dend, then spines.
	vector< unsigned int > junctions( 
		numSoma_ + numDend_ + numSpine_, UINT_MAX );
	xByL_.resize( numSoma_ + numDend_ + numSpine_, 0.0 );
	volume_.resize( numSoma_ + numDend_ + numSpine_, 0.0 );
	buildDiffusionJunctions( junctions );
	buildMoleculeNameMap( soma, somaMap_ );
	buildMoleculeNameMap( dend, dendMap_ );
	buildMoleculeNameMap( spine, spineMap_ );

	/*
	for ( unsigned int j = 0; j < junctions.size(); ++j )
		cout << " " << j << "," << junctions[j];
	cout << endl;
	*/

	setAllVols();

	completeSomaDiffusion( junctions );
	completeDendDiffusion( junctions );
	completeSpineDiffusion( junctions );

	// set< string >( kin, "method", dendMethod_ );
}

/**
 * Traverses the signaling tree to build a map of molecule Elements 
 * looked up by name.
 */
void SigNeur::buildMoleculeNameMap( Element* e,
	map< string, Element* >& molMap )
{
	static const Finfo* childSrcFinfo = 
		initNeutralCinfo()->findFinfo( "childSrc" );
	static const Finfo* parentFinfo = 
		initNeutralCinfo()->findFinfo( "parent" );
	if ( e == 0 )
		return;
	
	if ( e->cinfo()->isA( initMoleculeCinfo() ) ) {
		string ename = e->name();
		if ( ename == "kenz_cplx" ) {
			Id pa;
			get< Id >( e, parentFinfo, pa );
			Id grandpa;
			get< Id >( pa(), parentFinfo, grandpa );
			ename = grandpa()->name() + "__" + ename;
		}
		map< string, Element* >::iterator i = molMap.find( ename );
		if ( i != molMap.end() ) {
			cout << "buildMoleculeNameMap:: Warning: duplicate molecule: "
				<< i->second->id().path() << ", " << e->id().path() << endl;
		} else {
			molMap[ ename ] = e;
		}
	}
	// Traverse children.
	const Msg* m = e->msg( childSrcFinfo->msg() );
	for ( vector< ConnTainer* >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		if ( (*i)->e2() != e )
			buildMoleculeNameMap( (*i)->e2(), molMap );
		else
			buildMoleculeNameMap( (*i)->e1(), molMap );
	}
}
