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
#include "../element/Wildcard.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "SmoldynHub.h"
#include "Molecule.h"
#include "Particle.h"
#include "Reaction.h"
#include "Enzyme.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

extern "C"{
#include "Smoldyn/source/smolrun.h"
#include "Smoldyn/source/smolload.h"
}

const double SmoldynHub::MINIMUM_DT = 1.0e-6;

const Cinfo* initSmoldynHubCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};
	static Finfo* hubShared[] =
	{
		new DestFinfo( "rateTermInfo",
			Ftype3< vector< RateTerm* >*, KinSparseMatrix*, bool >::global(),
			RFCAST( &SmoldynHub::rateTermFunc )
		),
		new DestFinfo( "rateSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &SmoldynHub::rateSizeFunc )
		),
		new DestFinfo( "molSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &SmoldynHub::molSizeFunc )
		),
		new DestFinfo( "molConnection",
			Ftype3< vector< double >* , 
				vector< double >* , 
				vector< Element *>*  
				>::global(),
			RFCAST( &SmoldynHub::molConnectionFunc )
		),
		new DestFinfo( "reacConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::reacConnectionFunc )
		),
		new DestFinfo( "enzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::enzConnectionFunc )
		),
		new DestFinfo( "mmEnzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::mmEnzConnectionFunc )
		),
		new DestFinfo( "completeSetup",
			Ftype1< string >::global(),
			RFCAST( &SmoldynHub::completeReacSetupFunc )
		),
		new DestFinfo( "clear",
			Ftype0::global(),
			RFCAST( &SmoldynHub::clearFunc )
		),
	};

	static Finfo* fluxShared[] =
	{
		new SrcFinfo( "efflux", Ftype1< vector < double > >::global() ),
		new DestFinfo( "influx", Ftype1< vector< double > >::global(), 
			RFCAST( &SmoldynHub::flux )),
	};

	static Finfo* smoldynHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nMol", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNspecies ), 
			&dummyFunc
		),
		new ValueFinfo( "nReac", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNreac ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNenz ), 
			&dummyFunc
		),
		new ValueFinfo( "path", 
			ValueFtype1< string >::global(),
			GFCAST( &SmoldynHub::getPath ),
			RFCAST( &SmoldynHub::setPath )
		),
		new ValueFinfo( "dt", 
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getDt ),
			RFCAST( &SmoldynHub::setDt )
		),
		new ValueFinfo( "seed", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getSeed ),
			RFCAST( &SmoldynHub::setSeed )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	//	new SrcFinfo( "nSrc", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "destroy", Ftype0::global(),
			&SmoldynHub::destroy ),
		new DestFinfo( "molSum", Ftype1< double >::global(),
			RFCAST( &SmoldynHub::molSum ) ),
		new DestFinfo( "child", Ftype1< int >::global(),
			RFCAST( &SmoldynHub::childFunc ),
			"override the Neutral::childFunc here, so that when this is deleted all the zombies are "
			"reanimated." ),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "hub", hubShared, 
			      sizeof( hubShared ) / sizeof( Finfo* ),
					"Handles reaction structure info from the Stoich object" ),
		new SharedFinfo( "molSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ),
					"Takes over the process message to each of the kinetic objects.Replaces the "
					"original message usually sent by the clock Ticks." ),
		new SharedFinfo( "reacSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ),
					"Takes over the process message to each of the kinetic objects.Replaces the "
					"original message usually sent by the clock Ticks." ),
		new SharedFinfo( "enzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ),
					"Takes over the process message to each of the kinetic objects.Replaces the "
					"original message usually sent by the clock Ticks." ),
		new SharedFinfo( "mmEnzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ),
					"Takes over the process message to each of the kinetic objects.Replaces the "
					"original message usually sent by the clock Ticks." ),
		new SharedFinfo( "flux", fluxShared, 
			      sizeof( fluxShared ) / sizeof( Finfo* ),
					"This is used to handle fluxes between sets of molecules solved in this "
					"SmoldynHub and solved by other Hubs. It is implemented as a reciprocal "
					"vector of influx and efflux.The influx during each timestep is added "
					"directly to the molecule number in S_. The efflux is computed by the Hub, "
					"and subtracted from S_, and sent on to the target Hub.Its main purpose, as "
					"the name implies, is for diffusive flux across an interface.Typically used "
					"for mixed simulations where the molecules in different spatial domains are "
					"solved differently." ),
	};

	// Schedule smoldynHubs for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "SmoldynHub",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SmoldynHub: Interface object between Smoldyn (by Steven Andrews) and MOOSE.",
	};

	static Cinfo smoldynHubCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		smoldynHubFinfos,
		sizeof( smoldynHubFinfos )/sizeof(Finfo *),
		ValueFtype1< SmoldynHub >::global(),
			schedInfo, 1
	);

	return &smoldynHubCinfo;
}

static const Cinfo* smoldynHubCinfo = initSmoldynHubCinfo();

/*
const Finfo* SmoldynHub::particleFinfo = 
	initSmoldynHubCinfo()->findFinfo( "particleFinfo" );
	*/

const Finfo* SmoldynHub::molSolveFinfo =
	initSmoldynHubCinfo()->findFinfo( "molSolve" );

static const Finfo* molSumFinfo =
	initSmoldynHubCinfo()->findFinfo( "molSum" );

static const Finfo* reacSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "reacSolve" );
static const Finfo* enzSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "enzSolve" );
static const Finfo* mmEnzSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "mmEnzSolve" );

static const Slot molSumSlot = initSmoldynHubCinfo()->getSlot( "molSum" );
static const Slot fluxSlot =
	initSmoldynHubCinfo()->getSlot( "flux.efflux" );

void redirectDestMessages(
	Element* hub, Element* e, const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex, vector< unsigned int >& map,
	vector< Element* >* elist, bool retain = 0 );

void redirectDynamicMessages( Element* e );
void unzombify( const Conn* c );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SmoldynHub::SmoldynHub()
	: simptr_( 0 ), dt_( 0.01 ), seed_( 0 )
{
		;
}

void SmoldynHub::destroy( const Conn* c )
{
	clearFunc( c );
	Neutral::destroy( c );
}

void SmoldynHub::childFunc( const Conn* c, int stage )
{
	if ( stage == 1 ) // Clean out zombies before deleting messages.
		clearFunc( c );
	Neutral::destroy( c );
}

/**
 * Here we add external inputs to a molecule. This message replaces
 * the SumTot input to a molecule when it is used for controlling its
 * number. It is not meant as a substitute for SumTot between molecules.
 * It is a bit tricky in Smoldyn, because we cannot just add numbers, but
 * have to give them a position.
 */
void SmoldynHub::molSum( const Conn* c, double val )
{
	Element* hub = c->targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, molSumSlot );
	SmoldynHub* sh = static_cast< SmoldynHub* >( hub->data() );

	assert( index < sh->molSumMap_.size() );
	// Do something intelligent here with the molSum input.
}

///////////////////////////////////////////////////
// Hub utility function definitions
///////////////////////////////////////////////////

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
SmoldynHub* SmoldynHub::getHubFromZombie( 
	const Element* e, const Finfo* srcFinfo, unsigned int& index )
{
	unsigned int procSlot = e->cinfo()->getSlotIndex( "process" );
	assert( procSlot != 0 );
	const Conn& c = *e->connDestBegin( procSlot );
	unsigned int slot;
	bool ret = srcFinfo->getSlotIndex( srcFinfo->name(), slot );
	assert( ret );
	Element* hub = c.targetElement();
	index = hub->connSrcRelativeIndex( c, slot );
	return static_cast< SmoldynHub* >( hub->data() );
}

void SmoldynHub::setPos( unsigned int molIndex, double value, 
			unsigned int num, unsigned int dim )
{
	assert( dim < 3 );
	unsigned int count = 0;
	int smolIndex = molIndex + 1; // Add in the extra null molecule index

	// i = 0 is fixed molecules, i = 1 is mobile.
	for ( unsigned int i = 0; i < 2; i++ ) {
		moleculestruct** live = simptr_->mols->live[ i ];
		moleculestruct** live_end = live + simptr_->mols->nl[ i ];
		while ( live < live_end ) {
			if ( ( *live )->ident == smolIndex ) {
				if ( count == num ) {
					( *live )->pos[dim] = value;
					return;
				}
				++count;
			}
			++live;
		}
	}
}

/**
 * This gets the position of the specified molecule species molIndex,
 * for particle #num of that species, on dimension dim.
 * It is spectacularly inefficient, because it has to run through all 
 * the 'live' molecules to find the one you want. If there is a 
 * bigger plotting job one should use the getPosVector function instead.
 */
double SmoldynHub::getPos( unsigned int molIndex, unsigned int num, 
			unsigned int dim )
{
	assert( dim < 3 );
	unsigned int count = 0;
	int smolIndex = molIndex + 1; // Add in the extra null molecule index

	// i = 0 is fixed molecules, i = 1 is mobile.
	for ( unsigned int i = 0; i < 2; i++ ) {
		moleculestruct** live = simptr_->mols->live[ i ];
		moleculestruct** live_end = live + simptr_->mols->nl[ i ];
		while ( live < live_end ) {
			if ( ( *live )->ident == smolIndex ) {
				if ( count == num ) {
					return ( *live )->pos[dim];
				}
				++count;
			}
			++live;
		}
	}
	return 0.0;
}

void SmoldynHub::setPosVector( unsigned int molIndex, 
			const vector< double >& value, unsigned int dim )
{
	assert( dim < 3 );
	int smolIndex = molIndex + 1; // Add in the extra null molecule index

	// i = 0 is fixed molecules, i = 1 is mobile.
	vector< double >::const_iterator v = value.begin();
	for ( unsigned int i = 0; i < 2; i++ ) {
		moleculestruct** live = simptr_->mols->live[ i ];
		moleculestruct** live_end = live + simptr_->mols->nl[ i ];
		while ( live < live_end && v != value.end() ) {
			if ( ( *live )->ident == smolIndex ) {
				( *live )->pos[dim] = *v++;
			}
			++live;
		}
	}
}

/**
 * This gets the position of all molecules of the specified molecule 
 * species molIndex, on dimension dim.
 */
void SmoldynHub::getPosVector( unsigned int molIndex,
			vector< double >& value, unsigned int dim )
{
	assert( dim < 3 );
	int smolIndex = molIndex + 1; // Add in the extra null molecule index
	value.resize( 0 );

	// i = 0 is fixed molecules, i = 1 is mobile.
	for ( unsigned int i = 0; i < 2; i++ ) {
		moleculestruct** live = simptr_->mols->live[ i ];
		moleculestruct** live_end = live + simptr_->mols->nl[ i ];
		while ( live < live_end ) {
			if ( ( *live )->ident == smolIndex ) {
				value.push_back( ( *live )->pos[dim] );
			}
			++live;
		}
	}
}

void SmoldynHub::setNinit( unsigned int molIndex, unsigned int value )
{
	cout << "void SmoldynHub::setNinit( unsigned int molIndex = " <<
		molIndex << ", unsigned int value = " << value << " )\n";
}

unsigned int SmoldynHub::getNinit( unsigned int molIndex )
{
	return 0;
}

// Sets the number of particles of the mol species specified by molIndex
void SmoldynHub::setNparticles( unsigned int molIndex, unsigned int value )
{
	cout << "void SmoldynHub::setNparticles( unsigned int molIndex = " <<
		molIndex << ", unsigned int value = " << value << " )\n";
}

// Returns the number of particles of the mol species specified by molIndex
unsigned int SmoldynHub::getNparticles( unsigned int molIndex )
{
	// Ideally I would use a Smoldyn routine here, but the closest approx
	// is cmdmolcount which dumps the data to a file. So here I'll just
	// reimplement it.
	
	unsigned int count = 0;
	int smolIndex = molIndex + 1; // Add in the extra null molecule index

	// i = 0 is fixed molecules, i = 1 is mobile.
	for ( unsigned int i = 0; i < 2; i++ ) {
		moleculestruct** live = simptr_->mols->live[ i ];
		moleculestruct** live_end = live + simptr_->mols->nl[ i ];
		while ( live < live_end )
			count += ( *(live++) )->ident == smolIndex;
	}
	return count;
}

void SmoldynHub::setD( unsigned int molIndex, double value )
{
}

double SmoldynHub::getD( unsigned int molIndex )
{
	return 0.0314;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
unsigned int SmoldynHub::getNspecies( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numSpecies();
}

unsigned int SmoldynHub::numSpecies() const
{
	return nMol_;
}

unsigned int SmoldynHub::getNreac( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numReac();
}

unsigned int SmoldynHub::numReac() const
{
	return reacMap_.size();
}

unsigned int SmoldynHub::getNenz( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numEnz();
}

unsigned int SmoldynHub::numEnz() const
{
	return enzMap_.size();
}

string SmoldynHub::getPath( const Element* e )
{
	return static_cast< const SmoldynHub* >( e->data() )->path_;
}

void SmoldynHub::setPath( const Conn* c, string value )
{
	Element* e = c->targetElement();
	static_cast< SmoldynHub* >( c->data() )->localSetPath( e, value );
}

double SmoldynHub::getDt( const Element* e )
{
	return static_cast< const SmoldynHub* >( e->data() )->dt_;
}

void SmoldynHub::setDt( const Conn* c, double value )
{
	if ( value > SmoldynHub::MINIMUM_DT )
		static_cast< SmoldynHub* >( c->data() )->dt_ = value;
	else
		cout << "Warning: Assigning dt to SmoldynHub '" <<
			c->targetElement()->name() << "' : requested value = " << 
			value << " out of range, ignored.\nUsing old dt = " <<
			static_cast< const SmoldynHub* >( c->data() )->dt_ << endl;
}

unsigned int SmoldynHub::getSeed( const Element* e )
{
	return static_cast< const SmoldynHub* >( e->data() )->seed_;
}

void SmoldynHub::setSeed( const Conn* c, unsigned int value )
{
	static_cast< SmoldynHub* >( c->data() )->seed_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SmoldynHub::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< SmoldynHub* >( c->data() )->reinitFuncLocal( 
					c->targetElement(), info );
}

void SmoldynHub::reinitFuncLocal( Element* e, ProcInfo info )
{
	simptr_->dt = info->dt_;
}

void SmoldynHub::completeReacSetupLocal( const string& path )
{
	simptr_->boxs->mpbox = 5.0;
	simptr_->wlist[0]->pos = -2.0e-5;
	simptr_->wlist[1]->pos = 2.0e-5;
	simptr_->wlist[2]->pos = -2.0e-5;
	simptr_->wlist[3]->pos = 2.0e-5;
	simptr_->wlist[4]->pos = -2.0e-5;
	simptr_->wlist[5]->pos = 2.0e-5;

	////////////////////////////////////////////////////////////////
	//  On now to reactions.
	////////////////////////////////////////////////////////////////

	// First, fill in the arrays.
	unsigned int maxident = nMol_ + 1;
	
	// Now define reaction entries.
	vector< unsigned int > molIndex; 
	vector< RateTerm* >::iterator ri;
	int num0Order = 0;
	int num1Order = 0;
	int num2Order = 0;
	int reacIndex = 0;
	unsigned int numForward = 0;

	for ( ri = rates_->begin(); ri != rates_->end(); ++ri ) {
		numForward = (*ri)->getReactants( molIndex, *S_ );
		if ( numForward == 0 ) {
			++num0Order;
		} else if ( numForward == 1 ) {
			++num1Order;
		} else if ( numForward == 2 ) {
			++num2Order;
		}
	}
	simptr_->rxn[0] = rxnalloc( 0, maxident, num0Order );
	simptr_->rxn[1] = rxnalloc( 1, maxident, num1Order );
	simptr_->rxn[2] = rxnalloc( 2, maxident, num2Order );
	num0Order = 0;
	num1Order = 0;
	num2Order = 0;
	reacIndex = 0;

	/// \todo: Figure out how to associate reaction names with rates_ so I can add those in as well.
	for ( ri = rates_->begin(); ri != rates_->end(); ++ri ) {
		numForward = ( *ri )->getReactants( molIndex, *S_ );
		if ( numForward == molIndex.size() ) { // need to fill in products
			findProducts( molIndex, ri - rates_->begin() );
		}
		unsigned int numProds = molIndex.size() - numForward;
		if ( numForward == 0 ) {
			reacIndex = num0Order++;
			// AddRxn2Struct( ? ) Don't know what to do with zero-order
			// I need to add a the reaction products and rates etc.
			// last arg here is dimensionality.
			AddRxnProds2Struct( simptr_->rxn[0], reacIndex, numProds, 3 );
			for ( unsigned int j = 0; j < numProds; j++ ) {
				simptr_->rxn[0]->prod[reacIndex][j]->ident = 1 + molIndex[ numForward + j ];
				simptr_->rxn[0]->prod[reacIndex][j]->mstate = MSsoln;
			}

			simptr_->rxn[0]->rate[reacIndex] = ( *ri )->getR1();
			sprintf( simptr_->rxn[0]->rname[ reacIndex ],
				"r0_%d", reacIndex );
		} else if ( numForward == 1 ) {
			reacIndex = num1Order++;

			// The 1 is the # of reactions added.
			AddRxns2Struct( simptr_->rxn[1], 
				1 + molIndex[0], MSsoln ,
				0, MSsoln,
				1, maxident ); 
			
			AddRxnProds2Struct( simptr_->rxn[1], reacIndex, numProds, 3 );
			for ( unsigned int j = 0; j < numProds; j++ ) {
				simptr_->rxn[1]->prod[reacIndex][j]->ident = 1 + molIndex[ numForward + j ];
				simptr_->rxn[1]->prod[reacIndex][j]->mstate = MSsoln;
			}
			simptr_->rxn[1]->rate[reacIndex] = ( *ri )->getR1();
			int j = simptr_->rxn[1]->nrxn[ 1 + molIndex[0] ] - 1;
			simptr_->rxn[1]->table[ 1 + molIndex[0] ][ j ] = reacIndex;
			// 3 is the system dimensionality
			sprintf( simptr_->rxn[1]->rname[ reacIndex ],
				"r1_%d", reacIndex );
		} else if ( numForward == 2 ) {
			reacIndex = num2Order++;
			int i = ( 1 + molIndex[0] ) * maxident + molIndex[1] + 1;
			AddRxns2Struct( simptr_->rxn[2], 
				1 + molIndex[0], MSsoln,	// Identifies first substrate
				1 + molIndex[1], MSsoln,	// Identifies second substrate
				1, maxident ); // 1 is the # of reactions added
			AddRxnProds2Struct( simptr_->rxn[2], reacIndex, numProds, 3 );
			for ( unsigned int j = 0; j < numProds; j++ ) {
				simptr_->rxn[2]->prod[reacIndex][j]->ident = 1 + molIndex[ numForward + j ];
				simptr_->rxn[2]->prod[reacIndex][j]->mstate = MSsoln;
			}

	// These rates are in #*Vol/time.
	// I need to extract volume at some point to scale this.
	// For now, hack in a volume term.

			simptr_->rxn[2]->rate[reacIndex] = ( *ri )->getR1() * 8.0e-18;
			int j = simptr_->rxn[2]->nrxn[ i ] - 1;
			simptr_->rxn[2]->table[ i ][ j ] = reacIndex;
			sprintf( simptr_->rxn[2]->rname[ reacIndex ],
				"r2_%d", reacIndex );
		}
	}
	

	// For reactions, use the function:
	// AddRxns2struct
	// and
	// AddRxnPrds2Struct
	//
	// I still need to fill in the following fields:
	// - Rates
	// - Names
	// - Reversible stuff: set rtype to x
	// - Permit: for surface bound molecules. (for later)
	// 	 ( if A is on surface, and B in bulk, then turn off certain reacs.)
	// 	 Would be possible to handle if I insist that such molecules are
	// 	 different species.
	//
	simptr_->time = 0.0;
	simptr_->tmin = 0.0;
	simptr_->tmax = 100.0;

	// The SmoldynHub has to be informed about a preferred dt. This is
	// automatically done in the KineticManager::smoldynSetup function,
	// using an estimate based on Euler accuracy for dt. Perhaps this
	// can be fine-tuned for Smoldyn numerical methods.
	simptr_->dt = dt_;
	simptr_->randseed = seed_;
	srand( seed_ ); 
	
	/*
	// Ooops. Smoldyn currently needs dt to be set _before_ calling 
	// setupsim. So I need to bung it in somehow.
	// Steve tells me that he'll put in this flexibility some time soon,
	// so for now I'm going to kludge it.
	Id t0( "/sched/cj/t0" );
	assert( t0.good() );
	double dt = 0.1;
	get< double >( t0(), "dt", dt );
	simptr_->dt = dt;
	*/

	// Here we fill up the surface info.
	setSurfaces( path );

	scmdsetfnames( simptr_->cmds, "smoldyn.out smoldyn.coords" );
	// scmdsetfnames( simptr_->cmds, "smoldyn.coords" );
	scmdstr2cmd( simptr_->cmds, "e molcount smoldyn.out", simptr_->tmin, simptr_->tmax, simptr_->dt );

	scmdstr2cmd( simptr_->cmds, "n 1 listmols3 P smoldyn.coords", simptr_->tmin, simptr_->tmax, simptr_->dt );
	if ( setupsim( NULL, NULL, &simptr_, NULL ) ) {
		cout << "Warning: SmoldynHub::reinitFuncLocal: Setupsim failed\n";
	}
	scmdopenfiles( simptr_->cmds, 0 );
}

void SmoldynHub::processFunc( const Conn* c, ProcInfo info )
{
	Element* e = c*targetElement();
	static_cast< SmoldynHub* >( e->data() )->processFuncLocal( e, info );
}

void SmoldynHub::processFuncLocal( Element* e, ProcInfo info )
{
	handleInflux();
	int ret = simulatetimestep( simptr_ );
	if ( ret > 1 ) { // 0 is OK, 1 is out of time. 2 and up are other errors
		cout << "Bad things happened to Smoldyn\n";
	} 
	handleEfflux( e, info );
}

/**
 * flux:
 * This function handles molecule transfer into the solver.
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
 * Internally the solver will have to stuff the flux values into 
 * the Smoldyn list, and do so at the correct spatial locations.
 */

void SmoldynHub::flux( const Conn* c, vector< double > influx )
{
	assert( influx.size() == nMol_ );
	Element* hub = c->targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, fluxSlot );
	SmoldynHub* sh = static_cast< SmoldynHub* >( hub->data() );

	for ( i = 0; i < nMol_; i++ ) {
		int in = influx[i];
		int ret = molputimport( sh->simptr_, in, i + 1, 
			1,  // molece state, what does it mean?
			panelMap_[ molMap_[ i ] ], // Looking up comput.
			0, // panel face
			0 );
	}

	// Here we do some stuff to put the incoming molecules into
	// Smoldyn. The key thing will be to map the correct identities
	// between the two solvers.
}


/**
 * The processFunc handles the efflux from the hub.
 * Smoldyn provides an exportList of particles emerging from the
 * surface at the junction between the domains of the solvers.
 * Here we examine this export list and work out where to send them.
 * This should really be done indepenednetly 
 */
void SmoldynHub::handleEfflux( Element* hub, ProcInfo info )
{
	molputimport( sh_->shiptr, int nmol, int ident, 
	MolState ms, parallelptr prl, enum panelFunc	
}


///////////////////////////////////////////////////
// Geometry assignment utility functions.
///////////////////////////////////////////////////
void assignPoints( Element* e, panelptr pptr )
{
	const unsigned int nDim = 3; // should really refer to simptr.
	// The points array has been allocated by panelsalloc, so we 
	// just need to fill it in.
	// Ask Steven : The dimension is not specified in any of the 
	// panel, surface, or surfacesuper struct. Only the simptr has it.
	vector< double > coords;
	bool ret = get< vector< double > >( e, "coords", coords );
	assert( ret );
	assert ( pptr->npts * nDim + nDim == coords.size() );
	for ( int i = 0; i < pptr->npts; i++ ) {
		for ( unsigned int j = 0; j < nDim; j++ ) {
			pptr->point[i][j] = coords[ i * nDim + j ];
			cout << pptr->point[i][j] << "	";
		}
		cout << endl;
	}
	for ( unsigned int j = 0; j < nDim; j++ ) {
		pptr->front[j] = coords[ pptr->npts * nDim + j ];
		cout << pptr->front[j] << "	";
	}
	cout << endl;
}

void assignNeighbors( Element* e, panelptr pptr,
	map< Element*, panelptr >& panelMap )
{
	static const Cinfo* panelCinfo = Cinfo::find( "Panel" );
	static const Finfo* nSrcFinfo = panelCinfo->findFinfo( "neighborSrc" );
	static const Finfo* nDestFinfo = panelCinfo->findFinfo( "neighbor" );

	// the neigh array has NOT been allocated. need to do so.
	vector< Conn > neighborsIn;
	vector< Conn > neighborsOut;
	unsigned int numIncoming = nDestFinfo->incomingConns( e, neighborsIn );
	unsigned int numOutgoing = nSrcFinfo->outgoingConns( e, neighborsOut );
	pptr->nneigh = numIncoming + numOutgoing;
	pptr->neigh = ( panelptr* ) calloc( pptr->nneigh, sizeof( panelptr* ));
	panelptr* temp = pptr->neigh;
	vector< Conn >::iterator c;
	map< Element*, panelptr >::iterator i;
	for ( c = neighborsIn.begin(); c != neighborsIn.end(); c++ ) {
		i = panelMap.find( c->targetElement() );
		assert( i != panelMap.end() );
		*temp++ = i->second;
	}

	for ( c = neighborsOut.begin(); c != neighborsOut.end(); c++ ) {
		i = panelMap.find( c->targetElement() );
		assert( i != panelMap.end() );
		*temp++ = i->second;
	}
}

///////////////////////////////////////////////////
// This goes through the path and finds all the geometry info.
// The surface and panel instances hold this info.
///////////////////////////////////////////////////

// This is deprecated. I should only use the setSurfaces.
void SmoldynHub::localSetPath( Element* stoich, const string& value )
{
}

void SmoldynHub::setSurfaces( const string& value )
{
	static const Cinfo* panelCinfo = Cinfo::find( "Panel" );
	static const Cinfo* surfaceCinfo = Cinfo::find( "Surface" );
	static const int dim = 3;

	path_ = value;
	vector< Element* > ret;
	// vector< Element* > panels;
	vector< Element* > surfaces;
	vector< vector< Id > > panels;
	vector< Element* >::iterator i;
	vector< Id > children;
	wildcardFind( path_, ret );
	unsigned int numPanels = 0;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
	//	if ( (*i)->cinfo()->isA( panelCinfo ) )
	//		panels.push_back( *i );
		if ( (*i)->cinfo()->isA( surfaceCinfo ) ) {
			surfaces.push_back( *i );
			panels.push_back( Neutral::getChildList( *i ) );
			numPanels += panels.back().size();
		}
	}

	// Check with Steven about nMol_ + 1.
	// Assume 3D.
	simptr_->srfss = surfacessalloc( surfaces.size(), nMol_ + 1, 3 );
	simptr_->srfss->nsrf = surfaces.size();
	for ( unsigned int j = 0; j < surfaces.size(); j++ ) {
		map< Element*, panelptr > panelMap;

		strncpy( simptr_->srfss->snames[j], 
			surfaces[j]->name().c_str(), 199 );
		vector< vector< Id > > typedPanels( PSMAX );
		vector< Id >& pan = panels[j];

		// Here we organize panels into a vector indexed by the shapeId.
		for ( unsigned int k = 0; k < pan.size(); k++ ) {
			unsigned int shapeId;
			if ( pan[k]()->cinfo()->isA( panelCinfo ) ) {
				bool temp = get< unsigned int >( pan[k](), "shapeId", shapeId );
				assert( temp );
				assert ( shapeId < PSMAX );
				typedPanels[ shapeId ].push_back( pan[k] );
			}
		}

		// Now we set up the panels, in order of ShapeId.
		for ( unsigned int k = 0; k < typedPanels.size(); k++ ) {
			if ( typedPanels[k].size() > 0 ) {
				panelsalloc( simptr_->srfss->srflist[j], dim, 
					typedPanels[k].size(), 
					static_cast< PanelShape >( k ) );
				simptr_->srfss->srflist[j]->npanel[k] = typedPanels[k].size();
			}
			for ( unsigned int q = 0; q < typedPanels[k].size(); q++ ) {
				Element* pe = typedPanels[k][q]();
				panelptr pptr = simptr_->srfss->srflist[j]->panels[k][q];
				panelMap[ pe ] = pptr;
			}
		}
		
		for ( unsigned int k = 0; k < typedPanels.size(); k++ ) {
			for ( unsigned int q = 0; q < typedPanels[k].size(); q++ ) {
				Element* pe = typedPanels[k][q]();
				panelptr pptr = simptr_->srfss->srflist[j]->panels[k][q];
				strncpy( pptr->pname, pe->name().c_str(), 199 );
				// This extracts the points from pe and puts into pptr.
				assignPoints( pe, pptr );

				// Traverse neighbours and set them up.
				assignNeighbors( pe, pptr, panelMap );

				// Check if panel is an interface to other solvers
				setupTrade( pe );
			}
		}
	}

	cout << "found " << numPanels << " panels; ";
	cout << surfaces.size() << " surfaces, \n";
}

/**
 * This function sets up import/export of molecules from smoldyn and
 * ties it to a specific panel.
 * At this point Smoldyn assumes a single interface for traffic, and any
 * panel(s) can be part of this interface. There may be improbable cases
 * where we may have the Smoldyn zone as a sandwich between two others,
 * in which case we need more complicated interfacing. Later.
 */
void SmoldynHub::setupTrade( Element* pe )
{
	bool tradeFlag;
	bool ret = get< bool >( pe, "tradeFlag", tradeFlag );
	tradePanel_.resize( 0 );
	if ( tradeFlag ) {
		tradePanel_.push_back( pe );
	}
}

///////////////////////////////////////////////////
// This is where the reaction system from MOOSE is zombified and 
// incorporated into the solution engine.
///////////////////////////////////////////////////

/**
 * This sets up the local version of the rate term array, used to 
 * figure out reaction structures.
 */
void SmoldynHub::rateTermFunc( const Conn* c,
	vector< RateTerm* >* rates, KinSparseMatrix* N, bool useHalfReacs )
{
	// the useHalfReacs flag is irrelevant here, since Smoldyn always 
	// considers irreversible reactions
	
	static_cast< SmoldynHub* >( c->data() )->localRateTermFunc( rates, N );
}

void SmoldynHub::localRateTermFunc( vector< RateTerm* >* rates,
	KinSparseMatrix* N )
{
	rates_ = rates;
	N_ = N;
}

/**
 * This sets up the groundwork for setting up the molecule connections.
 * We need to know how they are subdivided into mol, buf and sumTot to
 * assign the correct messages. The next call should come from the
 * molConnectionFunc which will provide the actual vectors.
 */
void SmoldynHub::molSizeFunc(  const Conn* c,
	unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	static_cast< SmoldynHub* >( c->data() )->molSizeFuncLocal(
			nMol, nBuf, nSumTot );
}
void SmoldynHub::molSizeFuncLocal( 
		unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	nMol_ = nMol;
	nBuf_ = nBuf;
	nSumTot_ = nSumTot;
}

/**
 * This function zombifies the molecules
 */
void SmoldynHub::molConnectionFunc( const Conn* c,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	Element* e = c->targetElement();
	static_cast< SmoldynHub* >( c->data() )->
		molConnectionFuncLocal( e, S, Sinit, elist );
}

void SmoldynHub::findProducts( vector< unsigned int >& molIndex, 
	size_t reacIndex )
{
	unsigned int j = static_cast< unsigned int >( reacIndex );
	for ( unsigned int i = 0; i < N_->nRows(); i++ ) {
		// Handle stoichiometry by pushing back same molecule as often
		// as needed to build up full count.
		for ( int entry = N_->get( i, j ) - 1 ; entry >= 0; entry-- )
			molIndex.push_back( i );
	}
}

void SmoldynHub::molConnectionFuncLocal( Element* hub,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	assert( nMol_ + nBuf_ + nSumTot_ == elist->size() );

	S_ = S;
	Sinit_ = Sinit;

	// cout << "in molConnectionFuncLocal\n";
	vector< Element* >::iterator i;
	// Note that here we have perfect alignment between the
	// order of the S_ and Sinit_ vectors and the elist vector.
	// This is used implicitly in the ordering of the process messages
	// that get set up between the Hub and the objects.
	const Finfo* sumTotFinfo = initParticleCinfo()->findFinfo( "sumTotal" );
	Finfo* particleFinfo = 
		const_cast< Finfo* >( initParticleCinfo()->getThisFinfo() );

	for ( i = elist->begin(); i != elist->end(); i++ ) {
		zombify( hub, *i, molSolveFinfo, particleFinfo );
		redirectDynamicMessages( *i );
	}
	// Here we should really set up a 'set' of mols to check if the
	// sumTotMessage is coming from in or outside the tree.
	// Since I'm hazy about the syntax, here I'm just using the elist.
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		// Here we replace the sumTotMessages from outside the tree.
		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo, 
			i - elist->begin(), molSumMap_, elist, 1 );
	}

	// Steven says: root is the configuration file path. 
	// It is recorded in sim (the simulation
	// structure) so output files will be saved in the proper directory.
	//
	// Put in an extra molecule for the empty one at index 0.
	//
	simptr_ = simalloc( 3, nMol_ + 1, "tmp" );
	double dmax = 0.0;
	for ( vector< double >::iterator i = Sinit->begin(); 
			i != Sinit->end(); ++i )
		dmax += *i;

	// Presumably we won't have much more than dmax molecules
	// Steven cautions that this may need to be bigger. I should
	// check it out for various systems.
	int max = static_cast< int >( dmax * 2.1 );
	if ( max <= 0 ) { // can't assert this error, may happen legally.
		cerr << "Error: SmoldynHub::molConnectionFuncLocal: zero molecules in simulation\n";
		return;
	}
	// The extra molecule species here is for the ident of zero.
	simptr_->mols = molssalloc( 3, max, 1 + nMol_ );
	// dont need simptr_->name = calloc( nMol_, sizeof( char* ) );
	vector< Element* > fixedMols;
	vector< Element* > diffusibleMols;
	vector< int > dim;
	vector< unsigned int > ident;
	unsigned int j = 0;


	// Steven says: I can eliminate this loop if I put all the molecules
	// into the 'dead' list and then call molsort
	//
	// I should fill them in from top (high index) down, and update
	// nd and topd correctly.
#if 0	
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		// do we need to allocate it, or does simalloc do so ?
		simptr_->name[ j++ ] = calloc( 
			( *i )->name().length() + 1, sizeof( char* ) );
		strcpy( simptr_->name[j], ( *i )->name().c_str() );
		double D;
		get< double >( *i, "D", D );
		if ( D > 0.0 ) {
			diffusibleMols.push_back( *i );
		} else {
			fixedMols.push_back( *i );
		}
		dim.push_back( getGeomDim( *i ) );
	}
	simptr_->mols->live[0] = calloc( diffusibleMols.size(), 
		sizeof( moleculeptr ) );
	simptr_->mols->live[1] = calloc( fixedMols.size(), 
		sizeof( moleculeptr ) );
	simptr_->mols->nl[0] = diffusibleMols.size();
	simptr_->mols->nl[1] = fixedMols.size();

	for ( j = 0; j < diffusibleMols.size(); ++j ) {
		moleculeptr mp = molalloc( dim[j] );
		mp->serno = j;
		mp->ident = 
		simptr_->mols->live[ 0 ][ j ] = mp;
	}

	for ( j = 0; j < fixedMols.size(); ++j )
		simptr_->mols->live[ 1 ][ j ] = molalloc( dim[j] );
#endif


	// Here is the alternative suggested by Steven:
	j = 1; // Note indexing starting with 1, to skip empty molecule.
	int k = max - 1;
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		// Here we need to pick out only the variable molecules.
		// Buffered ones have to be handled otherwise.

		
		// All names defined to 256 chars, so we don't have to allocate.
		strncpy( simptr_->name[j], ( *i )->name().c_str(), 255 );
		double D;
		get< double >( *i, "D", D );
		cout << "Diff const of " << (*i)->name() << " = " << D << endl;
		// D = 1.0e-12; // a reasonable number.
		simptr_->mols->difc[j][ MSsoln ] = D;
		// don't need to do: moleculeptr mp = molalloc( getGeomDim( *i ) );
		// don't need to do: mp->serno = j;
		// Need to assign same identity to each instance of that species.
		moleculeptr* d = simptr_->mols->dead;
		unsigned int maxq = static_cast< unsigned int >( ( *Sinit )[j - 1] );
		for ( unsigned int q = 0; q < maxq; q++ ) {
			// Here we want to scatter the molecules in space. Not
			// so easy if we have a funny geometry.
			// It is up to the user to specify a starting position in
			// the script file.
			
			/*
			// As a default I'll put them all at zero.
			// For now let's put them in one corner of the geometry,
			// which is a 3 micron by 1 micron cylinder capped with 
			// hemispheres and with a 0.5 micron nucleus in the middle.
			// So we want them all in a little cube of 0.5 micron side
			// at 0.5 microns to the right of the origin.
			d[ k ]->pos[0] = mtrand() * 0.5e-6 + 5e-6;
			d[ k ]->pos[1] = mtrand() * 0.5e-6 - 0.25e-6;
			d[ k ]->pos[2] = mtrand() * 0.5e-6 - 0.25e-6;
			*/
			d[ k-- ]->ident = j;
		}
		j++;
	}
	// simptr_->mols->nd = max - k;
	simptr_->mols->topd = k + 1; // Check with Steven : right, should be # alloced.
	simptr_->nident = nMol_ + 1;
	// Here is the magic function. Difsort arg should be 0
// 	molsort( simptr_->mols, 0 );
// 	This will be done now with a setup function.
	
}

void SmoldynHub::rateSizeFunc(  const Conn* c,
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	static_cast< SmoldynHub* >( c->data() )->rateSizeFuncLocal(
		c.targetElement(), nReac, nEnz, nMmEnz );
}
void SmoldynHub::rateSizeFuncLocal( Element* hub, 
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	// Ensure we have enough space allocated in each of the maps
	reacMap_.resize( nReac );
	enzMap_.resize( nEnz );
	mmEnzMap_.resize( nMmEnz );

	// Not sure what to do here.
	// cout << "in rateSizeFuncLocal\n";
}

void SmoldynHub::reacConnectionFunc( const Conn* c,
	unsigned int index, Element* reac )
{
	Element* e = c->targetElement();
	static_cast< SmoldynHub* >( c->data() )->
		reacConnectionFuncLocal( e, index, reac );
}

/**
 * This may as well come directly from a base class for hubs. It works
 * the same, with a different local getReacKf type function.
 */
void SmoldynHub::reacConnectionFuncLocal( 
		Element* hub, int rateTermIndex, Element* reac )
{
	// These fields will replace the original reaction fields so that
	// the lookups refer to the solver rather than the molecule.
	static Finfo* reacFields[] =
	{
		new ValueFinfo( "kf",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getReacKf ),
			RFCAST( &SmoldynHub::setReacKf )
		),
		new ValueFinfo( "kb",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getReacKb ),
			RFCAST( &SmoldynHub::setReacKb )
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

void SmoldynHub::enzConnectionFunc( const Conn* c,
	unsigned int index, Element* enz )
{
	Element* e = c->targetElement();
	static_cast< SmoldynHub* >( c->data() )->
		enzConnectionFuncLocal( e, index, enz );
}

/**
 * Zombifies mmEnzs. But these pose an issue for Smoldyn: it is not a
 * good molecular concept. Two issues here. One is that the rate term
 * itself is odd, and we would have to munge the terms in Smoldyn to
 * be equivalent. the other is that the enz-substrate complex must not
 * deplete the originating enzymes.
 */
void SmoldynHub::mmEnzConnectionFunc( const Conn* c,
	unsigned int index, Element* mmEnz )
{
	// cout << "in mmEnzConnectionFunc for " << mmEnz->name() << endl;
	Element* e = c->targetElement();
	static_cast< SmoldynHub* >( c->data() )->
		mmEnzConnectionFuncLocal( e, index, mmEnz );
}

void SmoldynHub::enzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* enz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK1 ),
			RFCAST( &SmoldynHub::setEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK2 ),
			RFCAST( &SmoldynHub::setEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK3 ),
			RFCAST( &SmoldynHub::setEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzKm ),
			RFCAST( &SmoldynHub::setEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK3 ), // Same as k3
			RFCAST( &SmoldynHub::setEnzKcat ) // this is different.
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

void SmoldynHub::mmEnzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* mmEnz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzK1 ),
			RFCAST( &SmoldynHub::setMmEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzK2 ),
			RFCAST( &SmoldynHub::setMmEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKcat ),
			RFCAST( &SmoldynHub::setMmEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKm ),
			RFCAST( &SmoldynHub::setMmEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKcat ),
			RFCAST( &SmoldynHub::setMmEnzKcat )
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

/*
void unzombify( const Conn* c )
{
	Element* e = c->targetElement();
	const Cinfo* ci = c->cinfo();
	bool ret = ci->schedule( e );
	assert( ret );
	e->setThisFinfo( const_cast< Finfo* >( ci->getThisFinfo() ) );
	redirectDynamicMessages( e );
}
*/

/**
 * This operation turns the target element e into a zombie controlled
 * by the hub/solver. It gets rid of any process message coming into 
 * the zombie and replaces it with one from the solver.
 */
void SmoldynHub::zombify( 
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
 * Completes all the setup operations. At this point the Stoich object
 * has called all the individual reaction operations and now is telling
 * the SmoldynHub to wrap it up.
 */
void SmoldynHub::completeReacSetupFunc( const Conn* c, string s )
{
	static_cast< SmoldynHub* >( c->data() )->completeReacSetupLocal( s );
}

/**
 * Clears out all the messages to zombie objects
 */
void SmoldynHub::clearFunc( const Conn* c )
{
	// cout << "Starting clearFunc for " << c.targetElement()->name() << endl;
	Element* e = c->targetElement();

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


///////////////////////////////////////////////////////////////////////
// Zombie functions. Later to be farmed out.
///////////////////////////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'n' field of the 
 * molecule. It first sets the solver location handling this
 * field, then the molecule itself.
 * For the molecule set/get operations, the lookup order is identical
 * to the message order. So we don't need an intermediate table.
 */
/*
 * Not used. Instead we simply define a 'Particle', which is a zombie
 * for a molecule with extra fields that can be accessed when solved
 * with a SmoldynHub.
 * 
void SmoldynHub::setMolN( const Conn& c, double value )
{
	cout << "void SmoldynHub::setMolN( const Conn& c, double value= " <<
		value << ")\n";
}

double SmoldynHub::getMolN( const Element* e )
{
	cout << "double SmoldynHub::getMolN( const Element* e )\n";
	return 0.0;
}

void SmoldynHub::setMolNinit( const Conn& c, double value )
{
	cout << "void SmoldynHub::setMolNinit( const Conn& c, double value= "
		<< value << ")\n";
}

double SmoldynHub::getMolNinit( const Element* e )
{
	cout << "double SmoldynHub::getMolNinit( const Element* e )\n";
	return 0.0;
}
*/

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
void SmoldynHub::setReacKf( const Conn* c, double value )
{
	cout << "void SmoldynHub::setReacKf( const Conn* c, double value= "
		<< value << ")\n";
}

// getReacKf does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double SmoldynHub::getReacKf( const Element* e )
{
	cout << "double SmoldynHub::getReacKf( const Element* e )\n";
	return 0.0;
}

void SmoldynHub::setReacKb( const Conn* c, double value )
{
}

double SmoldynHub::getReacKb( const Element* e )
{
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
void SmoldynHub::setEnzK1( const Conn* c, double value )
{
	cout << "void SmoldynHub::setEnzK1( const Conn* c, double value= "
		<< value << "): Not yet implemented.\n";
}

// getEnzK1 does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double SmoldynHub::getEnzK1( const Element* e )
{
	return Enzyme::getK1( e );
}

void SmoldynHub::setEnzK2( const Conn* c, double value )
{
}

double SmoldynHub::getEnzK2( const Element* e )
{
	return Enzyme::getK2( e );
}

void SmoldynHub::setEnzK3( const Conn* c, double value )
{
}

double SmoldynHub::getEnzK3( const Element* e )
{
	return Enzyme::getK3( e );
}

// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setEnzKcat( const Conn* c, double value )
{
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setEnzKm( const Conn* c, double value )
{
}

double SmoldynHub::getEnzKm( const Element* e )
{
	return Enzyme::getKm( e );
}


//////////////////////////////////////////////////////////////////
// Here we set up stuff for mmEnzymes. It is similar, but not identical,
// to what we did for ordinary enzymes.
//////////////////////////////////////////////////////////////////
void SmoldynHub::setMmEnzK1( const Conn* c, double value )
{
}

// The Kinetic solver has no record of mmEnz::K1, so we simply go back to
// the object here. Should have a way to bypass this.
double SmoldynHub::getMmEnzK1( const Element* e )
{
	return 0.0;
}

// Ugh. Should perhaps ignore this mess.
void SmoldynHub::setMmEnzK2( const Conn* c, double value )
{
}

double SmoldynHub::getMmEnzK2( const Element* e )
{
	return 0.0;
}

// Note that this differs from assigning kcat. k3 leads to changes
// in Km and kcat, whereas kcat only affects kcat.
void SmoldynHub::setMmEnzK3( const Conn* c, double value )
{
}

double SmoldynHub::getMmEnzKcat( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setMmEnzKcat( const Conn* c, double value )
{
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setMmEnzKm( const Conn* c, double value )
{
}

double SmoldynHub::getMmEnzKm( const Element* e )
{
	return 0.0;
}

/////////////////////////////////////////////////////////////////////////
// Redirection functions
/////////////////////////////////////////////////////////////////////////

/**
 * This function redirects messages arriving at zombie elements onto
 * the hub. 
 * e is the zombie element whose messages are being redirected to the hub.
 * eFinfo is the Finfo holding those messages.
 * hubFinfo is the Finfo on the hub which will now handle the messages.
 * eIndex is the index to look up the element.
*/
/*
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
	
	map.push_back( eIndex );
	
	// An issue here: Do I check if the src is on the solved tree?
	// This is a bad iteration: dropping connections is going to affect
	// the list size and position of i in the list.
	// for ( i = 0; i != max; i++ ) {
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
*/

/**
 * Here we replace the existing DynamicFinfos and their messages with
 * new ones for the updated access functions
 */
/*
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
*/
