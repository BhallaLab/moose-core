/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "../ksolve/VoxelPoolsBase.h"
#include "../ksolve/VoxelPools.h"
#include "../ksolve/KsolveBase.h"
#include "../ksolve/Ksolve.h"
#include "lookupVolumeFromMesh.h"
#include "PoolBase.h"

#define EPSILON 1e-15

const SpeciesId DefaultSpeciesId = 0;

const Cinfo* PoolBase::initPoolBaseCinfo()
{
    //////////////////////////////////////////////////////////////
    // Field Definitions
    //////////////////////////////////////////////////////////////
    static ElementValueFinfo< PoolBase, double > n(
        "n",
        "Number of molecules in pool",
        &PoolBase::setN,
        &PoolBase::getN
    );

    static ElementValueFinfo< PoolBase, double > nInit(
        "nInit",
        "Initial value of number of molecules in pool",
        &PoolBase::setNinit,
        &PoolBase::getNinit
    );

    static ElementValueFinfo< PoolBase, double > diffConst(
        "diffConst",
        "Diffusion constant of molecule",
        &PoolBase::setDiffConst,
        &PoolBase::getDiffConst
    );

    static ElementValueFinfo< PoolBase, double > motorConst(
        "motorConst",
        "Motor transport rate molecule. + is away from soma, - is "
        "towards soma. Only relevant for ZombiePool subclasses.",
        &PoolBase::setMotorConst,
        &PoolBase::getMotorConst
    );

    static ElementValueFinfo< PoolBase, double > conc(
        "conc",
        "Concentration of molecules in this pool",
        &PoolBase::setConc,
        &PoolBase::getConc
    );

    static ElementValueFinfo< PoolBase, double > concInit(
        "concInit",
        "Initial value of molecular concentration in pool",
        &PoolBase::setConcInit,
        &PoolBase::getConcInit
    );

    static ElementValueFinfo< PoolBase, double > volume(
        "volume",
        "Volume of compartment. Units are SI. "
        "Utility field, the actual volume info is "
        "stored on a volume mesh entry in the parent compartment."
        "This mapping is implicit: the parent compartment must be "
        "somewhere up the element tree, and must have matching mesh "
        "entries. If the compartment isn't"
        "available the volume is just taken as 1",
        &PoolBase::setVolume,
        &PoolBase::getVolume
    );

    static ElementValueFinfo< PoolBase, unsigned int > speciesId(
        "speciesId",
        "Species identifier for this mol pool. Eventually link to ontology.",
        &PoolBase::setSpecies,
        &PoolBase::getSpecies
    );

    static ElementValueFinfo< PoolBase, bool > isBuffered(
        "isBuffered",
        "Flag: True if Pool is buffered. "
        "In the case of Pool and BufPool the field can be assigned, to "
        "change the type of the Pool object to BufPool, or vice versa. "
        "None of the messages are affected. "
        "This object class flip can only be done in the non-zombified "
        "form of the Pool/BufPool. In Zombies it is read-only.",
        &PoolBase::setIsBuffered,
        &PoolBase::getIsBuffered
    );

    //////////////////////////////////////////////////////////////
    // MsgDest Definitions
    //////////////////////////////////////////////////////////////

    static DestFinfo reacDest( "reacDest",
                               "Handles reaction input",
                               new OpFunc2< PoolBase, double, double >( &PoolBase::reac )
                             );

    static DestFinfo handleMolWt( "handleMolWt",
                                  "Separate finfo to assign molWt, and consequently diffusion const."
                                  "Should only be used in SharedMsg with species.",
                                  new EpFunc1< PoolBase, double >( &PoolBase::handleMolWt )
                                );
    static DestFinfo notifyCreate("notifyCreate", 
				"Called when object is created. Arg is parent.", 
				new EpFunc1<PoolBase, ObjId >(&PoolBase::notifyCreate));
    static DestFinfo notifyCopy("notifyCopy", 
				"Called when object is created. Arg is original.", 
				new EpFunc1<PoolBase, ObjId >(&PoolBase::notifyCopy));
    static DestFinfo notifyDestroy("notifyDestroy", 
				"Called when object is destroyed.", 
				new EpFunc0<PoolBase>(&PoolBase::notifyDestroy));
    static DestFinfo notifyMove("notifyMove", 
				"Called when object is moved. Arg is new parent.", 
				new EpFunc1<PoolBase, ObjId>(&PoolBase::notifyMove));
    static DestFinfo notifyAddMsgSrc("notifyAddMsgSrc", 
				"Called when a message is created, current object is src. Arg is msgId.", 
				new EpFunc1<PoolBase, ObjId>(&PoolBase::notifyAddMsgSrc));
    static DestFinfo notifyAddMsgDest("notifyAddMsgSrc", 
				"Called when a message is created, current object is dest. Arg is msgId.", 
				new EpFunc1<PoolBase, ObjId>(&PoolBase::notifyAddMsgDest));

    static DestFinfo setSolvers( "setSolvers",
                "Assigns solvers to Pool. Args: ksolve, dsolve",
                new EpFunc2< PoolBase, ObjId, ObjId >(&PoolBase::setSolvers)
    );
    //////////////////////////////////////////////////////////////
    // MsgDest Definitions: These three are used for non-reaction
    // calculations involving algebraically defined rate terms.
    //////////////////////////////////////////////////////////////
    static DestFinfo increment( "increment",
                                "Increments mol numbers by specified amount. Can be +ve or -ve",
                                new OpFunc1< PoolBase, double >( &PoolBase::increment )
                              );

    static DestFinfo decrement( "decrement",
                                "Decrements mol numbers by specified amount. Can be +ve or -ve",
                                new OpFunc1< PoolBase, double >( &PoolBase::decrement )
                              );

    static DestFinfo nIn( "nIn",
                          "Assigns the number of molecules in Pool to specified value",
                          new OpFunc1< PoolBase, double >( &PoolBase::nIn )
                        );

    //////////////////////////////////////////////////////////////
    // SrcFinfo Definitions
    //////////////////////////////////////////////////////////////

    static SrcFinfo1< double > nOut(
        "nOut",
        "Sends out # of molecules in pool on each timestep"
    );

    static SrcFinfo0 requestMolWt(
        "requestMolWt",
        "Requests Species object for mol wt"
    );

    //////////////////////////////////////////////////////////////
    // SharedMsg Definitions
    //////////////////////////////////////////////////////////////
    static Finfo* reacShared[] =
    {
        &reacDest, &nOut
    };
    static SharedFinfo reac( "reac",
                             "Connects to reaction",
                             reacShared, sizeof( reacShared ) / sizeof( const Finfo* )
                           );

    static Finfo* speciesShared[] =
    {
        &requestMolWt, &handleMolWt
    };

    static SharedFinfo species( "species",
                                "Shared message for connecting to species objects",
                                speciesShared, sizeof( speciesShared ) / sizeof ( const Finfo* )
                              );

    static Finfo* poolFinfos[] =
    {
        &n,			// Value
        &nInit,		// Value
        &diffConst,	// Value
        &motorConst,	// Value
        &conc,		// Value
        &concInit,	// Value
        &volume,	// Readonly Value
        &speciesId,	// Value
        &isBuffered,	// Value
        &increment,			// DestFinfo
        &decrement,			// DestFinfo
        &nIn,				// DestFinfo
        &reac,				// SharedFinfo
        &species,			// SharedFinfo
		&notifyCreate,		 // DestFinfo
		&notifyCopy,		 // DestFinfo
		&notifyDestroy,	 	// DestFinfo
		&notifyMove,		 // DestFinfo
		&notifyAddMsgSrc,	 // DestFinfo
		&notifyAddMsgDest,	 // DestFinfo
		&setSolvers,	 // DestFinfo
    };

    static string doc[] =
    {
        "Name", "PoolBase",
        "Author", "Upi Bhalla",
        "Description", "Base class for pools."
    };
    // static ZeroSizeDinfo< int > dinfo;
	static Dinfo< PoolBase > dinfo;
    static Cinfo poolCinfo (
        "PoolBase",
        Neutral::initCinfo(),
        poolFinfos,
        sizeof( poolFinfos ) / sizeof ( Finfo* ),
        &dinfo,
        doc,
        sizeof( doc )/sizeof( string ),
        true // Ban creation as this is an abstract base class.
    );

    return &poolCinfo;
}

const Cinfo* PoolBase::initPoolCinfo()
{
    static string doc[] =
    {
        "Name", "Pool",
        "Author", "Upi Bhalla",
        "Description", "Pool of molecules of a given species."
    };
	static Dinfo< PoolBase > dinfo;
	static Cinfo poolCinfo (
		"Pool",
		PoolBase::initPoolBaseCinfo(),
		0,
		0,
		&dinfo,
		doc, sizeof( doc )/sizeof( string )
	);
	return &poolCinfo;
}

const Cinfo* PoolBase::initBufPoolCinfo()
{
    static string doc[] =
    {
        "Name", "BufPool",
        "Author", "Upi Bhalla",
        "Description", "Buffered Pool of molecules of a given species."
    };
	static Dinfo< PoolBase > dinfo;
	static Cinfo poolCinfo (
		"BufPool",
		PoolBase::initPoolBaseCinfo(),
		0,
		0,
		&dinfo,
		doc, sizeof( doc )/sizeof( string )
	);
	return &poolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* poolBaseCinfo = PoolBase::initPoolBaseCinfo();
static const Cinfo* poolCinfo = PoolBase::initPoolCinfo();
static const Cinfo* bufPoolCinfo = PoolBase::initBufPoolCinfo();

//////////////////////////////////////////////////////////////
/*
KsolveBase* defaultKsolve()
{
	static Ksolve defaultKsolve_;
	return &defaultKsolve_;
}
*/

PoolBase::PoolBase()
	:
		dsolve_( 0 ),
		ksolve_( 0 ),
		concInit_( 0.0 ),
		diffConst_( 0.0 ),
		motorConst_( 0.0 )
{;}

PoolBase::~PoolBase()
{
	// notify the ksolve and dsolve that we are going.
	;
}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void PoolBase::increment( double val )
{
}

void PoolBase::decrement( double val )
{
}

void PoolBase::nIn( double val)
{
}

void PoolBase::reac( double A, double B )
{
}

void PoolBase::handleMolWt( const Eref& e, double v )
{
}

//////////////////////////////////////////////////////////////
/// notification functions
//////////////////////////////////////////////////////////////

void PoolBase::notifyDestroy(const Eref& e)
{
	if ( ksolve_ )
		ksolve_->notifyRemovePool( e );
	if ( dsolve_ )
		dsolve_->notifyRemovePool( e );
}

void PoolBase::notifyCreate(const Eref& e, ObjId parent)
{
	// cout << "Creating poolBase " << e.id().path() << endl;
	if ( ksolve_ )
		ksolve_->notifyAddPool( e );
	if ( dsolve_ )
		dsolve_->notifyAddPool( e );
}

void PoolBase::notifyCopy(const Eref& e, ObjId old)
{
	if ( ksolve_ )
		ksolve_->notifyAddPool( e );
	if ( dsolve_ )
		dsolve_->notifyAddPool( e );

	// setNinit( e, Field< double >::get( old, "nInit" ) );
	// setN( e, Field< double >::get( old, "n" ) );
	double c = Field< double >::get( old, "concInit" );
	setConcInit( e, c );
	c = Field< double >::get( old, "conc" );
	setConc( e, c );
}

void PoolBase::notifyMove(const Eref& e, ObjId newParent)
{
	// cout << "Moving poolBase " << e.id().path() << " onto " << newParent.path() << endl;
	if ( ksolve_ ) {
		ksolve_->notifyRemovePool( e );
	}
	if ( dsolve_ != 0 ) {
		dsolve_->notifyRemovePool( e );
	}
	// Assign to new ksolve if newParent is a solver.
}

void PoolBase::notifyAddMsgSrc(const Eref& e, ObjId msgId)
{
	if ( ksolve_ )
		ksolve_->notifyAddMsgSrcPool( e, msgId );
}

void PoolBase::notifyAddMsgDest(const Eref& e, ObjId msgId)
{
	if ( ksolve_ )
		ksolve_->notifyAddMsgDestPool( e, msgId );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void PoolBase::innerSetN( const Eref& e, double v ) 
{
	if ( v < 0.0 )
		v = 0.0;
	if ( ksolve_ )
    	ksolve_->setN(e, v);
	if ( dsolve_ )
    	dsolve_->setN(e, v);
}

void PoolBase::innerSetConcInit( const Eref& e, double conc )
{
	if ( conc < 0.0 )
		conc = 0.0;
	concInit_ = conc;
	if ( ksolve_ )
		ksolve_->setConcInit( e, conc );
	if ( dsolve_ )
    	dsolve_->setConcInit( e, conc );
}

void PoolBase::setN( const Eref& e, double v )
{
	innerSetN( e, v );
	if ( getIsBuffered( e ) ) {
		innerSetConcInit( e, v / ( NA * getVolume( e ) ) );
	}
}

double PoolBase::getN( const Eref& e ) const
{
	if ( ksolve_ )
    	return ksolve_->getN( e );
	else
    	return ( NA * getVolume( e ) ) * concInit_;
}

void PoolBase::setNinit( const Eref& e, double v )
{
    double c = v / ( NA * getVolume( e ) );
	setConcInit( e, c );
}

double PoolBase::getNinit( const Eref& e ) const
{
	return concInit_ * NA * getVolume( e );
}

// Conc is given in millimolar. Volume is in m^3
void PoolBase::setConc( const Eref& e, double conc )
{
    double n = NA * conc * getVolume( e );
    setN( e, n );
}

// Returns conc in millimolar.
double PoolBase::getConc( const Eref& e ) const
{
    return getN( e ) / ( NA * getVolume( e ) );
}

void PoolBase::setConcInit( const Eref& e, double conc )
{
	innerSetConcInit( e, conc );
	if ( getIsBuffered( e ) )
		innerSetN( e, NA * getVolume( e ) * conc );
}

double PoolBase::getConcInit( const Eref& e ) const
{
	return concInit_;
}

void PoolBase::setDiffConst( const Eref& e, double v )
{
	if ( v < 0.0 )
		v = 0.0;
	diffConst_ = v;
	if ( dsolve_ )
    	dsolve_->setDiffConst( e, v );
}

double PoolBase::getDiffConst(const Eref& e ) const
{
	return diffConst_;
}

void PoolBase::setMotorConst( const Eref& e, double v )
{
	motorConst_ = v;
	if ( dsolve_ )
    	dsolve_->setMotorConst( e, v );
}

double PoolBase::getMotorConst(const Eref& e ) const
{
	return motorConst_;
}

void PoolBase::setVolume( const Eref& e, double v )
{
	; // illegal op
}

double PoolBase::getVolume( const Eref& e ) const
{
    return lookupVolumeFromMesh( e );
}

void PoolBase::setSpecies( const Eref& e, unsigned int v )
{
    ;
}

unsigned int PoolBase::getSpecies( const Eref& e ) const
{
    return 0;
}

void PoolBase::setSolvers( const Eref& e, ObjId ks, ObjId ds )
{
	if ( ks == ObjId() ) {
		if ( ksolve_ )
			ksolve_->notifyRemovePool( e );
		ksolve_ = 0;
	} else if ( ! ks.bad() ) {
		string nm = ks.element()->cinfo()->name();
		if ( nm == "Ksolve" || nm == "Gsolve" ) {
			KsolveBase* k = reinterpret_cast< KsolveBase *>(ks.data() );
			if ( k && k != ksolve_ ) {
				if ( ksolve_ )
					ksolve_->notifyRemovePool( e );
				ksolve_ = k;
				k->notifyAddPool( e );
			}
		}
	}
	if ( ds == ObjId() ) {
		if ( dsolve_ ) {
			dsolve_->notifyRemovePool( e );
		}
		dsolve_ = 0;
	} else if ( ! ds.bad() ) {
		string nm = ds.element()->cinfo()->name();
		if ( nm == "Dsolve" ) {
			KsolveBase* d = reinterpret_cast< KsolveBase *>(ds.data() );
			if ( d && d != dsolve_ ) {
				if ( dsolve_ )
					dsolve_->notifyRemovePool( e );
				dsolve_ = d;
				d->notifyAddPool( e );
			}
		}
	}
}

/**
 * Changing the buffering flag changes the class between Pool and BufPool.
 */
void PoolBase::setIsBuffered( const Eref& e, bool v )
{
	Element* elm = e.element();
	bool isBuf = ( elm->cinfo() == bufPoolCinfo );
	if ( v == isBuf ) return;

	if ( v ) {
		elm->replaceCinfo( bufPoolCinfo );
	} else {
		elm->replaceCinfo( poolCinfo );
	}
	
	if ( ksolve_ )
		ksolve_->setIsBuffered( e, v );
	if ( dsolve_ )
		dsolve_->setIsBuffered( e, v );
}

bool PoolBase::getIsBuffered( const Eref& e ) const
{
	// return e.element()->cinfo()->name() == "BufPool";
	return e.element()->cinfo() == bufPoolCinfo;
}
