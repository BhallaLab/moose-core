/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include <math.h>
#include "../element/Neutral.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "InterHubFlux.h"
#include "KineticHub.h"
#include "Molecule.h"
#include "Reaction.h"
#include "Enzyme.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

void dummyStringFunc( const Conn& c, string s )
{
	;
}

const Cinfo* initKineticHubCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::reinitFunc ) ),
	};

	/**
	 * This is identical to the message sent from clock Ticks to
	 * objects. Here it is used to take over the Process message,
	 * usually only as a handle from the solver to the object.
	 * Here we are using the non-deprecated form.
	 */
	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};

	/**
	 * This is the destination of the several messages from the 
	 * Stoich object.
	 */
	static Finfo* hubShared[] =
	{
		new DestFinfo( "rateTermInfo", 
			Ftype3< vector< RateTerm* >*, SparseMatrix*, bool >::global(),
			RFCAST( &KineticHub::rateTermFunc )
		),
		new DestFinfo( "rateSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &KineticHub::rateSizeFunc )
		),
		new DestFinfo( "molSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &KineticHub::molSizeFunc )
		),
		new DestFinfo( "molConnection",
			Ftype3< vector< double >* , 
				vector< double >* , 
				vector< Element *>*  
				>::global(),
			RFCAST( &KineticHub::molConnectionFunc )
		),
		new DestFinfo( "reacConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::reacConnectionFunc )
		),
		new DestFinfo( "enzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::enzConnectionFunc )
		),
		new DestFinfo( "mmEnzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::mmEnzConnectionFunc )
		),
		new DestFinfo( "completeSetup",
			Ftype1< string >::global(),
			RFCAST( &dummyStringFunc )
		), 
			// The Kinetic hub doesn't need this call.
		new DestFinfo( "clear",
			Ftype0::global(),
			RFCAST( &KineticHub::clearFunc )
		),
	};

	/**
	 * This is used to handle fluxes between sets of molecules
	 * solved in this KineticHub and solved by other Hubs. It is
	 * implemented as a reciprocal vector of influx and efflux.
	 * The influx during each timestep is added directly to the 
	 * molecule number in S_. The efflux is computed by the
	 * Hub, and subtracted from S_, and sent on to the target Hub.
	 * Its main purpose, as the name implies, is for diffusive flux
	 * across an interface. Typically used for mixed simulations where
	 * the molecules in different spatial domains are solved differently.
	 */
	static Finfo* fluxShared[] =
	{
		new SrcFinfo( "efflux", Ftype1< vector < double > >::global() ),
		new DestFinfo( "influx", Ftype1< vector< double > >::global(), 
			RFCAST( &KineticHub::flux )),
	};

	static Finfo* kineticHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nMol", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNmol ), 
			&dummyFunc
		),
		new ValueFinfo( "nReac", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNreac ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNenz ), 
			&dummyFunc
		),

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "destroy", Ftype0::global(),
			&KineticHub::destroy ),
		new DestFinfo( "molSum", Ftype1< double >::global(),
			RFCAST( &KineticHub::molSum ) ),
		// override the Neutral::childFunc here, so that when this
		// is deleted all the zombies are reanimated.
		new DestFinfo( "child", Ftype1< int >::global(),
			RFCAST( &KineticHub::childFunc ) ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processShared,
			      sizeof( processShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "hub", hubShared, 
			      sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "molSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "reacSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "enzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "mmEnzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "flux", fluxShared, 
			      sizeof( fluxShared ) / sizeof( Finfo* ) ),
		/*
		new SolveFinfo( "molSolve", molFields, 
			sizeof( molFields ) / sizeof( const Finfo* ) );
			*/
	};

	static Cinfo kineticHubCinfo(
		"KineticHub",
		"Upinder S. Bhalla, 2007, NCBS",
		"KineticHub: Object for controlling reaction systems on behalf of the\nStoich object. Interfaces both with the reaction system\n(molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
		initNeutralCinfo(),
		kineticHubFinfos,
		sizeof(kineticHubFinfos )/sizeof(Finfo *),
		ValueFtype1< KineticHub >::global()
	);

	return &kineticHubCinfo;
}

static const Cinfo* kineticHubCinfo = initKineticHubCinfo();
static const Finfo* molSolveFinfo = 
	initKineticHubCinfo()->findFinfo( "molSolve" );
static const Finfo* reacSolveFinfo = 
	initKineticHubCinfo()->findFinfo( "reacSolve" );
static const Finfo* enzSolveFinfo = 
	initKineticHubCinfo()->findFinfo( "enzSolve" );
static const Finfo* mmEnzSolveFinfo = 
	initKineticHubCinfo()->findFinfo( "mmEnzSolve" );
static const Finfo* molSumFinfo = 
	initKineticHubCinfo()->findFinfo( "molSum" );

static const unsigned int molSumSlot =
	initKineticHubCinfo()->getSlotIndex( "molSum" );

static const unsigned int fluxSlot =
	initKineticHubCinfo()->getSlotIndex( "flux.efflux" );

void redirectDestMessages(
	Element* hub, Element* e, const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex, vector< unsigned int >& map,
	vector< Element* >* elist, bool retain = 0 );

void redirectDynamicMessages( Element* e );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

KineticHub::KineticHub()
	: S_( 0 ), Sinit_( 0 ), rates_( 0 ), 
	useHalfReacs_( 0 ), rebuildFlag_( 0 ),
	nMol_( 0 ), nBuf_( 0 ), nSumTot_( 0 )
{
	;
}

/**
 * In this destructor we need to put messages back to process,
 * and we need to replace the SolveFinfos on zombies with the
 * original ThisFinfo.
 * This should really just use the clearFunc. Try it once the
 * Smoldyn stuff is done.
 */
void KineticHub::destroy( const Conn& c)
{
	static Finfo* origMolFinfo =
		const_cast< Finfo* >(
		initMoleculeCinfo()->getThisFinfo( ) );
	static Finfo* origReacFinfo =
		const_cast< Finfo* >(
		initReactionCinfo()->getThisFinfo( ) );
	Element* hub = c.targetElement();
	vector< Conn > targets;
	vector< Conn >::iterator i;

	// First (todo) put the messages back onto the scheduler.
	// Second, replace the SolveFinfos
	molSolveFinfo->outgoingConns( hub, targets );
	for ( i = targets.begin(); i != targets.end(); i++ )
		i->targetElement()->setThisFinfo( origMolFinfo );

	reacSolveFinfo->outgoingConns( hub, targets );
	for ( i = targets.begin(); i != targets.end(); i++ )
		i->targetElement()->setThisFinfo( origReacFinfo );

	Neutral::destroy( c );
}

void KineticHub::childFunc( const Conn& c, int stage )
{
	if ( stage == 1 ) // clear messages: first clean out zombies before
		// the messages are all deleted.
		clearFunc( c );
	// Then fall back into what the Neutral version does
	Neutral::childFunc( c, stage );
}

/**
 * Here we add external inputs to a molecule. This message replaces
 * the SumTot input to a molecule when it is used for controlling its
 * number. It is not meant as a substitute for SumTot between molecules.
 */
void KineticHub::molSum( const Conn& c, double val )
{
	Element* hub = c.targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, molSumSlot );
	KineticHub* kh = static_cast< KineticHub* >( hub->data() );

	assert( index < kh->molSumMap_.size() );
	( *kh->S_ )[ kh->molSumMap_[ index ] ] = val;
}


/**
 * flux:
 * This function handles molecule transfer into and out of the solver.
 * Useful as an interface between solvers operating on different
 * scales.
 *
 * Flux will typically work with a large number of molecules all diffusing
 * through the same spatial interface between the solvers.
 *
 * Each pair of solvers will typically exchange only a single Flux message.
 * A given solver may have to manage multiple target fluxes. Consider a
 * dendrite with numerous spines each solved using Smoldyn.
 *
 * Internally the solver will have to stuff the flux values into its
 * S_ entries, and figure out outgoing fluxes from each of the affected
 * S_ entries. Usually the outgoing flux should be :
 * S_ * dt * D * XA
 * dt and XA is common to all terms in the message, but D is 
 * molecule-specific. Simplest to precalculate this term (with possible
 * exception of dt) and have a local array in the hub to do the scaling.
 *
 * May need a further extension in case the timesteps differ. Here
 * we would accumulate the flux information each time the message was
 * invoked, and the Process operation would clean up the accumulated
 * flux entries. Could get numerically unpleasant. 
 *
 * The responsibility for sending the 'flux' message to a remote target
 * rests with the _processFunc_ of the hub. Need to use the system clock
 * here as a synchonization enforcer, because the solvers at either end
 * of the flux message may have internal step sizes. For the Kinetic
 * ODE solvers, this means that the KineticManager must work out a 
 * sensible timestep.
 *
 * This is good for handling flux terms, but requires an extension to
 * Molecules etc to handle a new message type if we want to operate 
 * without solvers.
 *
 * It requires that someone should provide the outflux rate constant.
 * This could perhaps be set directly in an array on the Kinetic Hub.
 *
 * The other drawback is that it means that all solvers will need an
 * equivalent flux operation.
 */

void KineticHub::flux( const Conn& c, vector< double > influx )
{
	Element* hub = c.targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, fluxSlot );
	KineticHub* kh = static_cast< KineticHub* >( hub->data() );

	assert( index < kh->flux_.size() );
	/**
	 * The fluxMap is a key data structure here. It has pointers to
	 * the array entries in S_, for the molecules encountering external
	 * fluxes.
	 */
	vector< FluxTerm >& term = kh->flux_[ index ].term_;
	assert ( influx.size() == term.size() );
	vector< FluxTerm >::iterator i;
	vector< double >::iterator j = influx.begin();
	for ( i = term.begin(); i != term.end(); i++ )
		*( i->map_ ) += *j++; // map points to the array entry in S_
}

/**
 * The processFunc handles the efflux from the hub.
 * For now I have a separate and persistent vector for all the efflux
 * rates, but this is not really needed and if it is expensive could
 * easily be made a local variable.
 */
void KineticHub::processFuncLocal( Element* hub, ProcInfo info )
{
	unsigned int j;
	vector< FluxTerm >::iterator i;

	for ( j = 0; j < flux_.size(); j++ ) {
		InterHubFlux& ihf = flux_[j];
		vector< double > efflux( ihf.term_.size() );
		vector< double >::iterator k = efflux.begin();
		for ( i = ihf.term_.begin(); i != ihf.term_.end(); i++ ) {
			double &n = *( i->map_ );
			assert( n >= 0 );
			double flux = ( ihf.individualParticlesFlag_ ) ? 
				round( n * info->dt_ * i->effluxRate_ ) :
				n * info->dt_ * i->effluxRate_;
			if ( n < flux ) {
			/* 
			 * Oops, tried to pump out more molecules than we have.
			 * BAD! Should emit a warning here. But in any case,
			 * don't allow -ve molecules, so we set the flux to the 
			 * total # of molecules available.
			 */
				flux = ( ihf.individualParticlesFlag_ ) ?  round( n ) : n;
			}
			n -= flux;

			*k++ = i->efflux_ = flux;
		}
		sendTo1< vector< double > >( hub, fluxSlot, j, efflux );
	}
}

/**
 * Deprecated version of flux, handling a single molecule

void KineticHub::flux( const Conn& c, double influx, double outRateConst )
{
	Element* hub = c.targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, fluxSlot );
	KineticHub* kh = static_cast< KineticHub* >( hub->data() );

	assert( index < kh->fluxMap_.size() );
	double& y = ( ( *kh->S_ )[ kh->fluxMap_[ index ] ] );
	double z = y * outRateConst * dt;
	sendTo1< double >( c.targetElement(), fluxSlot, 
		c.targetIndex(), z );

	y += influx - z;
}
*/

/*
 * We are likely to have only a single point of contact between any two
 * solvers. If we use fluxVec as a way to manage robots we need more
 * contact points.
 * But flux is not the best way to do robots. The input value should just
 * be a number equivalent to a concentration. This looks more like 
 * sumtotal. The output value should also just be a set of conc values.
 */
/**
 * This function is another attempt to handle molecule transfer outside 
 * the solver, but using the existing A, B format for molecule changes.
 *
 * This is nice for retaining old reaction messaging, but asks that the
 * external coupling figure out the rates. Not so useful.
void KineticHub::extReac( const Conn& c, double A, double B )
{
	Element* hub = c.targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, extReacSlot );
	KineticHub* kh = static_cast< KineticHub* >( hub->data() );

	assert( index < kh->extReacMap_.size() );
	double& y = ( ( *kh->S_ )[ kh->extReacMap_[ index ] ] );
	// Here we do exp Euler addition onto y.
	if ( y > EPSILON && B > EPSILON ) {
		double C = exp ( - B * dt / y );
		y *= C + ( A / B ) * ( 1.0 - C );
	} else {
		y += ( A - B ) * dt;
	}
	sendTo1< double >( c.targetElement(), extReacSlot, 
		c.targetIndex(), y );
}
 */

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int KineticHub::getNmol( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

unsigned int KineticHub::getNreac( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->reacMap_.size();
}

unsigned int KineticHub::getNenz( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->enzMap_.size();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void KineticHub::reinitFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

void KineticHub::processFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

void KineticHub::rateTermFunc( const Conn& c,
	vector< RateTerm* >* rates, SparseMatrix* N,  bool useHalfReacs )
{
	KineticHub* kh = static_cast< KineticHub* >( c.data() );
	kh->rates_ = rates;
	kh->useHalfReacs_ = useHalfReacs;
}

/**
 * This sets up the groundwork for setting up the molecule connections.
 * We need to know how they are subdivided into mol, buf and sumTot to
 * assign the correct messages. The next call should come from the
 * molConnectionFunc which will provide the actual vectors.
 */
void KineticHub::molSizeFunc(  const Conn& c,
	unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	static_cast< KineticHub* >( c.data() )->molSizeFuncLocal(
			nMol, nBuf, nSumTot );
}
void KineticHub::molSizeFuncLocal( 
		unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	nMol_ = nMol;
	nBuf_ = nBuf;
	nSumTot_ = nSumTot;
}

void KineticHub::molConnectionFunc( const Conn& c,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	Element* e = c.targetElement();
	static_cast< KineticHub* >( e->data() )->
		molConnectionFuncLocal( e, S, Sinit, elist );
}

void KineticHub::molConnectionFuncLocal( Element* hub,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	// These fields will replace the original molecule fields so that
	// the lookups refer to the solver rather than the molecule.
	static Finfo* molFields[] =
	{
		new ValueFinfo( "n",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolN ),
			RFCAST( &KineticHub::setMolN )
		),
		new ValueFinfo( "nInit",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolNinit ),
			RFCAST( &KineticHub::setMolNinit )
		),
		new ValueFinfo( "conc",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolN ),
			RFCAST( &KineticHub::setMolN )
		),
		new ValueFinfo( "concInit",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolNinit ),
			RFCAST( &KineticHub::setMolNinit )
		),
	};
	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initMoleculeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo molZombieFinfo( 
		molFields, 
		sizeof( molFields ) / sizeof( Finfo* ),
		tf
	);

	assert( nMol_ + nBuf_ + nSumTot_ == elist->size() );

	S_ = S;
	Sinit_ = Sinit;

	// cout << "in molConnectionFuncLocal\n";
	vector< Element* >::iterator i;
	// Note that here we have perfect alignment between the
	// order of the S_ and Sinit_ vectors and the elist vector.
	// This is used implicitly in the ordering of the process messages
	// that get set up between the Hub and the objects.
	const Finfo* sumTotFinfo = initMoleculeCinfo()->findFinfo( "sumTotal" );
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		zombify( hub, *i, molSolveFinfo, &molZombieFinfo );
		redirectDynamicMessages( *i );
	}
	// Here we should really set up a 'set' of mols to check if the
	// sumTotMessage is coming from in or outside the tree.
	// Since I'm hazy about the syntax, here I'm just using the elist.
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		// Here we replace the sumTotMessages from outside the tree.
		// The 'retain' flag at the end is 1: we do not want to delete
		// the original message to the molecule.
		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo, 
		i - elist->begin(), molSumMap_, elist, 1 );
	}
}

void KineticHub::rateSizeFunc(  const Conn& c,
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	static_cast< KineticHub* >( c.data() )->rateSizeFuncLocal(
		c.targetElement(), nReac, nEnz, nMmEnz );
}
void KineticHub::rateSizeFuncLocal( Element* hub, 
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	// Ensure we have enough space allocated in each of the maps
	reacMap_.resize( nReac );
	enzMap_.resize( nEnz );
	mmEnzMap_.resize( nMmEnz );

	// Not sure what to do here.
	// cout << "in rateSizeFuncLocal\n";
}

void KineticHub::reacConnectionFunc( const Conn& c,
	unsigned int index, Element* reac )
{
	Element* e = c.targetElement();
	static_cast< KineticHub* >( e->data() )->
		reacConnectionFuncLocal( e, index, reac );
}

void KineticHub::reacConnectionFuncLocal( 
		Element* hub, int rateTermIndex, Element* reac )
{
	// These fields will replace the original reaction fields so that
	// the lookups refer to the solver rather than the molecule.
	static Finfo* reacFields[] =
	{
		new ValueFinfo( "kf",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getReacKf ),
			RFCAST( &KineticHub::setReacKf )
		),
		new ValueFinfo( "kb",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getReacKb ),
			RFCAST( &KineticHub::setReacKb )
		),
	};
	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initReactionCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo reacZombieFinfo( 
		reacFields, 
		sizeof( reacFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, reac, reacSolveFinfo, &reacZombieFinfo );
	unsigned int connIndex = reacSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( reacMap_.size() >= connIndex );

	reacMap_[connIndex - 1] = rateTermIndex;
}

void KineticHub::enzConnectionFunc( const Conn& c,
	unsigned int index, Element* enz )
{
	Element* e = c.targetElement();
	static_cast< KineticHub* >( e->data() )->
		enzConnectionFuncLocal( e, index, enz );
}

void KineticHub::mmEnzConnectionFunc( const Conn& c,
	unsigned int index, Element* mmEnz )
{
	// cout << "in mmEnzConnectionFunc for " << mmEnz->name() << endl;
	Element* e = c.targetElement();
	static_cast< KineticHub* >( e->data() )->
		mmEnzConnectionFuncLocal( e, index, mmEnz );
}

void KineticHub::enzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* enz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getEnzK1 ),
			RFCAST( &KineticHub::setEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getEnzK2 ),
			RFCAST( &KineticHub::setEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getEnzK3 ),
			RFCAST( &KineticHub::setEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getEnzKm ),
			RFCAST( &KineticHub::setEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getEnzK3 ), // Same as k3
			RFCAST( &KineticHub::setEnzKcat ) // this is different.
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initEnzymeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo enzZombieFinfo( 
		enzFields, 
		sizeof( enzFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, enz, enzSolveFinfo, &enzZombieFinfo );
	unsigned int connIndex = enzSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( enzMap_.size() >= connIndex );

	enzMap_[connIndex - 1] = rateTermIndex;
}

void KineticHub::mmEnzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* mmEnz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMmEnzK1 ),
			RFCAST( &KineticHub::setMmEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMmEnzK2 ),
			RFCAST( &KineticHub::setMmEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMmEnzKcat ),
			RFCAST( &KineticHub::setMmEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMmEnzKm ),
			RFCAST( &KineticHub::setMmEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMmEnzKcat ),
			RFCAST( &KineticHub::setMmEnzKcat )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initEnzymeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo enzZombieFinfo( 
		enzFields, 
		sizeof( enzFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, mmEnz, mmEnzSolveFinfo, &enzZombieFinfo );
	unsigned int connIndex = mmEnzSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( mmEnzMap_.size() >= connIndex );

	mmEnzMap_[connIndex - 1] = rateTermIndex;
}

void unzombify( const Conn& c )
{
	Element* e = c.targetElement();
	const Cinfo* ci = e->cinfo();
	bool ret = ci->schedule( e );
	assert( ret );
	e->setThisFinfo( const_cast< Finfo* >( ci->getThisFinfo() ) );
	redirectDynamicMessages( e );
}


/**
 * Clears out all the messages to zombie objects
 */
void KineticHub::clearFunc( const Conn& c )
{
	// cout << "Starting clearFunc for " << c.targetElement()->name() << endl;
	Element* e = c.targetElement();

	// First unzombify all targets
	vector< Conn > list;
	vector< Conn >::iterator i;

	molSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: molSolveFinfo unzombified " << list.size() << " elements\n";
	molSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	reacSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: reacFinfo unzombified " << list.size() << " elements\n";
	reacSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	enzSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: enzFinfo unzombified " << list.size() << " elements\n";
	enzSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	mmEnzSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: mmEnzFinfo unzombified " << list.size() << " elements\n";
	mmEnzSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );


	// Need the original molecule info. Where is that?
	// The molSumMap indexes from the sum index to the mol elist index.
	// But how do I access the mol elist? I need it to find the original
	// molecule element that was conneced from the table.
	molSumFinfo->incomingConns( e, list );
	molSumFinfo->dropAll( e );
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements for molecules
///////////////////////////////////////////////////

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
KineticHub* getHubFromZombie( const Element* e, const Finfo* srcFinfo,
		unsigned int& index )
{
	const SolveFinfo* f = dynamic_cast< const SolveFinfo* > (
			       	e->getThisFinfo() );
	if ( !f ) return 0;
	const Conn& c = f->getSolvedConn( e );
	unsigned int slot;
       	srcFinfo->getSlotIndex( srcFinfo->name(), slot );
	Element* hub = c.targetElement();
	index = hub->connSrcRelativeIndex( c, slot );
	return static_cast< KineticHub* >( hub->data() );
}

/**
 * Here we provide the zombie function to set the 'n' field of the 
 * molecule. It first sets the solver location handling this
 * field, then the molecule itself.
 * For the molecule set/get operations, the lookup order is identical
 * to the message order. So we don't need an intermediate table.
 */
void KineticHub::setMolN( const Conn& c, double value )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), molSolveFinfo, molIndex );
	if ( kh && kh->S_ ) {
		assert ( molIndex < kh->S_->size() );
		( *kh->S_ )[molIndex] = value;
	}
	Molecule::setN( c, value );
}

double KineticHub::getMolN( const Element* e )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( e, molSolveFinfo, molIndex );
	if ( kh && kh->S_ ) {
		assert ( molIndex < kh->S_->size() );
		return ( *kh->S_ )[molIndex];
	}
	return 0.0;
}

void KineticHub::setMolNinit( const Conn& c, double value )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), molSolveFinfo, molIndex );
	if ( kh && kh->Sinit_ ) {
		assert ( molIndex < kh->Sinit_->size() );
		( *kh->Sinit_ )[molIndex] = value;
	}
	Molecule::setNinit( c, value );
}

double KineticHub::getMolNinit( const Element* e )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( e, molSolveFinfo, molIndex );
	if ( kh && kh->Sinit_ ) {
		assert ( molIndex < kh->Sinit_->size() );
		return ( *kh->Sinit_ )[molIndex];
	}
	return 0.0;
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements for reactions
///////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'kf' field of the 
 * Reaction. It first sets the solver location handling this
 * field, then the reaction itself.
 * For the reaction set/get operations, the lookup order is
 * different from the message order. So we need an intermediate
 * table, the reacMap_, to map from one to the other.
 */
void KineticHub::setReacKf( const Conn& c, double value )
{
	unsigned int index;
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), reacSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->reacMap_.size() );
		index = kh->reacMap_[ index ];
		assert ( index < kh->rates_->size() );
		( *kh->rates_ )[index]->setR1( value );
	}
	Reaction::setKf( c, value );
}

// getReacKf does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double KineticHub::getReacKf( const Element* e )
{
	unsigned int index;
	KineticHub* kh = getHubFromZombie( e, reacSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->reacMap_.size() );
		index = kh->reacMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[index]->getR1();
	}
	return 0.0;
}

void KineticHub::setReacKb( const Conn& c, double value )
{
	unsigned int index;
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), reacSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->reacMap_.size() );
		index = kh->reacMap_[ index ];
		assert ( index < kh->rates_->size() );
		( *kh->rates_ )[index]->setR2( value );
	}
	Reaction::setKb( c, value );
}

double KineticHub::getReacKb( const Element* e )
{
	unsigned int index;
	KineticHub* kh = getHubFromZombie( e, reacSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->reacMap_.size() );
		index = kh->reacMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[index]->getR2();
	}
	return 0.0;
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements for Enzymes
///////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'k1' field of the 
 * Enzyme. It first sets the solver location handling this
 * field, then the reaction itself.
 * For the reaction set/get operations, the lookup order is
 * different from the message order. So we need an intermediate
 * table, the enzMap_, to map from one to the other.
 */
void KineticHub::setEnzK1( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK1\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index < kh->rates_->size() );
		( *kh->rates_ )[index]->setR1( value );
	}
	Enzyme::setK1( c, value );
}

// getEnzK1 does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double KineticHub::getEnzK1( const Element* e )
{
	unsigned int index;
	// cout << "in getEnzK1\n";
	KineticHub* kh = getHubFromZombie( e, enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[index]->getR1();
	}
	return 0.0;
}

void KineticHub::setEnzK2( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK2\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index < kh->rates_->size() );
		( *kh->rates_ )[index]->setR2( value );
	}
	Enzyme::setK2( c, value );
}

double KineticHub::getEnzK2( const Element* e )
{
	unsigned int index;
	// cout << "in getEnzK2\n";
	KineticHub* kh = getHubFromZombie( e, enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[index]->getR2();
	}
	return 0.0;
}

void KineticHub::setEnzK3( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK3\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index + 1 < kh->rates_->size() );
		( *kh->rates_ )[ index + 1 ]->setR1( value );
	}
	Enzyme::setK3( c, value );
}

double KineticHub::getEnzK3( const Element* e )
{
	unsigned int index;
	// cout << "in getEnzK3\n";
	KineticHub* kh = getHubFromZombie( e, enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index + 1 < kh->rates_->size() );
		return ( *kh->rates_ )[ index + 1 ]->getR1();
	}
	return 0.0;
}

// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void KineticHub::setEnzKcat( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzKcat\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index + 1 < kh->rates_->size() );
		double k1 = ( *kh->rates_ )[ index ]->getR1();
		double k2 = ( *kh->rates_ )[ index ]->getR2();
		double k3 = ( *kh->rates_ )[ index + 1 ]->getR1();
		if ( value > 0.0 && k3 > 0.0 ) {
			k2 *= value / k3;
			k1 *= value / k3;
			k3 = value;
			( *kh->rates_ )[index]->setR1( k1 );
			( *kh->rates_ )[index]->setR2( k2 );
			( *kh->rates_ )[ index + 1 ]->setR1( k3 );
		}
	}
	Enzyme::setKcat( c, value );
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void KineticHub::setEnzKm( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzKm\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index + 1 < kh->rates_->size() );
		double k1 = ( *kh->rates_ )[ index ]->getR1();
		double k2 = ( *kh->rates_ )[ index ]->getR2();
		double k3 = ( *kh->rates_ )[ index + 1 ]->getR1();
		if ( value > 0.0 ) {
			k1 = ( k2 + k3 ) / value;
			( *kh->rates_ )[index]->setR1( k1 );
		}
	}
	Enzyme::setKm( c, value );
}

double KineticHub::getEnzKm( const Element* e )
{
	unsigned int index;
	// cout << "in getEnzKm\n";
	KineticHub* kh = getHubFromZombie( e, enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->enzMap_.size() );
		index = kh->enzMap_[ index ];
		assert ( index + 1 < kh->rates_->size() );
		double k1 = ( *kh->rates_ )[ index ]->getR1();
		double k2 = ( *kh->rates_ )[ index ]->getR2();
		double k3 = ( *kh->rates_ )[ index + 1 ]->getR1();
		if ( k1 > 0.0 )
			return ( k2 + k3 ) / k1;
	}
	return 0.0;
}


//////////////////////////////////////////////////////////////////
// Here we set up stuff for mmEnzymes. It is similar, but not identical,
// to what we did for ordinary enzymes.
//////////////////////////////////////////////////////////////////
void KineticHub::setMmEnzK1( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK1\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		if ( value > 0.0 ) {
			double oldK1 = Enzyme::getKm( c.targetElement() );
			double oldKm = ( *kh->rates_ )[ index ]->getR1();
			double Km = oldKm * oldK1 / value;
			( *kh->rates_ )[index]->setR1( Km );
		}
	}
	Enzyme::setK1( c, value );
}

// The Kinetic solver has no record of mmEnz::K1, so we simply go back to
// the object here. Should have a way to bypass this.
double KineticHub::getMmEnzK1( const Element* e )
{
	return Enzyme::getK1( e );
}

// Ugh. Should perhaps ignore this mess.
void KineticHub::setMmEnzK2( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK2\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		Element* e = c.targetElement();
		double k1 = Enzyme::getK1( e );
		double k3 = Enzyme::getK3( e );
		double Km = ( value + k3 ) / k1;
		( *kh->rates_ )[index]->setR1( Km );
	}
	Enzyme::setK2( c, value );
}

double KineticHub::getMmEnzK2( const Element* e )
{
	return Enzyme::getK2( e );
}

// Note that this differs from assigning kcat. k3 leads to changes
// in Km and kcat, whereas kcat only affects kcat.
void KineticHub::setMmEnzK3( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzK3\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), enzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		Element* e = c.targetElement();
		double k1 = Enzyme::getK1( e );
		double k2 = Enzyme::getK2( e );
		double Km = ( k2 + value ) / k1;
		double kcat = value;
		( *kh->rates_ )[index]->setR1( Km );
		( *kh->rates_ )[index]->setR2( kcat );
	}
	Enzyme::setK3( c, value );
}

double KineticHub::getMmEnzKcat( const Element* e )
{
	unsigned int index;
	// cout << "in getMmEnzKcat\n";
	KineticHub* kh = getHubFromZombie( e, mmEnzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[ index ]->getR2();
	}
	return 0.0;
}

void KineticHub::setMmEnzKcat( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzKcat\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), mmEnzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		if ( value > 0.0 )
			( *kh->rates_ )[index]->setR2( value );
	}
	Enzyme::setKcat( c, value );
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void KineticHub::setMmEnzKm( const Conn& c, double value )
{
	unsigned int index;
	// cout << "in setEnzKm\n";
	KineticHub* kh = getHubFromZombie( 
		c.targetElement(), mmEnzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		if ( value > 0.0 )
			( *kh->rates_ )[index]->setR1( value );
	}
	Enzyme::setKm( c, value );
}

double KineticHub::getMmEnzKm( const Element* e )
{
	unsigned int index;
	// cout << "in getEnzKm\n";
	KineticHub* kh = getHubFromZombie( e, mmEnzSolveFinfo, index );
	if ( kh && kh->rates_ ) {
		assert ( index < kh->mmEnzMap_.size() );
		index = kh->mmEnzMap_[ index ];
		assert ( index < kh->rates_->size() );
		return ( *kh->rates_ )[index]->getR1();
	}
	return 0.0;
}
/*
void KineticHubWrapper::reacConnectionFuncLocal( int rateTermIndex, Element* reac )
{
			Field reacSolve( this, "reacSolve" );
			zombify( reac, reacSolve );
			reacIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::enzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field enzSolve( this, "enzSolve" );
			zombify( enz, enzSolve );
			enzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::mmEnzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field mmEnzSolve( this, "mmEnzSolve" );
			zombify( enz, mmEnzSolve );
			mmEnzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::molFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				molSrc_.sendTo( index, (*S_)[index] );
			} else if ( mode == SOLVER_SET ) {
				(*S_)[index] = n;
				(*Sinit_)[index] = nInit;
			} else if ( mode == SOLVER_REBUILD ) {
				rebuildFlag_ = 1;
			}
}
void KineticHubWrapper::bufFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
			} else if ( mode == SOLVER_SET ) {
				(*S_)[ index + nMol_ + nSumTot_ ] = 
					(*Sinit_)[ index + nMol_ + nSumTot_ ] = nInit;
			}
}
void KineticHubWrapper::sumTotFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				sumTotSrc_.sendTo( index, (*S_)[index + nMol_ ] );
			}
}
void KineticHubWrapper::reacFuncLocal( double kf, double kb, long index )
{
			unsigned long i = reacIndex_[ index ];
			if ( i >= 0 && i < rates_->size() - useOneWayReacs_ ) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( kf, 0 );
					( *rates_ )[i + 1]->setRates( kb, 0 );
				} else {
					( *rates_ )[i]->setRates( kf, kb );
				}
			}
}
void KineticHubWrapper::enzFuncLocal( double k1, double k2, double k3, long index )
{
			unsigned int i = enzIndex_[ index ];
			if ( i < rates_->size() - useOneWayReacs_ - 1) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( k1, 0 );
					( *rates_ )[i + 1]->setRates( k2, 0 );
					( *rates_ )[i + 2]->setRates( k3, 0 );
				} else {
					( *rates_ )[i]->setRates( k1, k2 );
					( *rates_ )[i + 1]->setRates( k3, 0 );
				}
			}
}
void KineticHubWrapper::mmEnzFuncLocal( double k1, double k2, double k3, long index )
{
			double Km = ( k2 + k3 ) / k1 ;
			unsigned int i = mmEnzIndex_[ index ];
			if ( i >= 0 && i < rates_->size() )
				( *rates_ )[i]->setRates( Km, k3 );
}
*/

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
/**
 * This operation turns the target element e into a zombie controlled
 * by the hub/solver. It gets rid of any process message coming into 
 * the zombie and replaces it with one from the solver.
 */
void KineticHub::zombify( 
		Element* hub, Element* e, const Finfo* hubFinfo,
	       	Finfo* solveFinfo )
{
	// Replace the original procFinfo with one from the hub.
	const Finfo* procFinfo = e->findFinfo( "process" );
	procFinfo->dropAll( e );
	bool ret;
	ret = hubFinfo->add( hub, e, procFinfo );
	assert( ret );

	// Redirect original messages from the zombie to the hub.
	// Pending.

	// Replace the 'ThisFinfo' on the solved element
	e->setThisFinfo( solveFinfo );
}

/**
 * This function redirects messages arriving at zombie elements onto
 * the hub. 
 * e is the zombie element whose messages are being redirected to the hub.
 * eFinfo is the Finfo holding those messages.
 * hubFinfo is the Finfo on the hub which will now handle the messages.
 * eIndex is the index to look up the element.
*/
void redirectDestMessages(
	Element* hub, Element* e, const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex, vector< unsigned int >& map, 
		vector< Element *>*  elist, bool retain )
{
	vector< Conn > clist;
	if ( eFinfo->incomingConns( e, clist ) == 0 )
		return;

	unsigned int i;
	unsigned int max = clist.size();
	vector< Element* > srcElements;
	vector< const Finfo* > srcFinfos;
	
	
	for ( i = 0; i != max; i++ ) {
		Element* tgt = clist[i].targetElement();
		if ( find( elist->begin(), elist->end(), tgt ) == elist->end() )
		// Only add the molSum message if the src is outside the mol tree
		// Add as many references to this molecule as there are
		// messages coming in.
			map.push_back( eIndex );
	}

	// An issue here: Do I check if the src is on the solved tree?
	// This is a bad iteration: dropping connections is going to affect
	// the list size and position of i in the list.
	i = max;
	while ( i > 0 ) {
		i--;
		Conn& c = clist[ i ];
		if ( find( elist->begin(), elist->end(), c.targetElement() ) == elist->end() )  {
			srcElements.push_back( c.targetElement() );
			srcFinfos.push_back( c.targetElement()->findFinfo( c.targetIndex() ) );
				
			if ( !retain )
				eFinfo->drop( e, i );
		}
	}
	// eFinfo->dropAll( e );
	for ( i = 0; i != srcElements.size(); i++ ) {
		bool ret = srcFinfos[ i ]->add( srcElements[ i ], hub, hubFinfo );
		assert( ret );
	}
}

/**
 * Here we replace the existing DynamicFinfos and their messages with
 * new ones for the updated access functions
 */
void redirectDynamicMessages( Element* e )
{
	const Finfo* f;
	unsigned int finfoNum = 1;
	unsigned int i;

	vector< Conn > clist;

	while ( ( f = e->localFinfo( finfoNum ) ) ) {
		const DynamicFinfo *df = dynamic_cast< const DynamicFinfo* >( f );
		assert( df != 0 );
		f->incomingConns( e, clist );
		unsigned int max = clist.size();
		vector< Element* > srcElements( max );
		vector< const Finfo* > srcFinfos( max );
		// An issue here: Do I check if the src is on the solved tree?
		for ( i = 0; i != max; i++ ) {
			Conn& c = clist[ i ];
			srcElements[ i ] = c.targetElement();
			srcFinfos[ i ]= c.targetElement()->findFinfo( c.targetIndex() );
		}

		f->outgoingConns( e, clist );
		max = clist.size();
		vector< Element* > destElements( max );
		vector< const Finfo* > destFinfos( max );
		for ( i = 0; i != max; i++ ) {
			Conn& c = clist[ i ];
			destElements[ i ] = c.targetElement();
			destFinfos[ i ] = c.targetElement()->findFinfo( c.targetIndex() );
		}
		string name = df->name();
		bool ret = e->dropFinfo( df );
		assert( ret );

		const Finfo* origFinfo = e->findFinfo( name );
		assert( origFinfo );

		max = srcFinfos.size();
		for ( i =  0; i < max; i++ ) {
			ret = srcFinfos[ i ]->add( srcElements[ i ], e, origFinfo );
			assert( ret );
		}
		max = destFinfos.size();
		for ( i =  0; i < max; i++ ) {
			ret = origFinfo->add( e, destElements[ i ], destFinfos[ i ] );
			assert( ret );
		}

		finfoNum++;
	}
}
