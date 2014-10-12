/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#endif

#include "OdeSystem.h"
#include "VoxelPoolsBase.h"
#include "VoxelPools.h"
#include "ZombiePoolInterface.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "../shell/Shell.h"

#include "../mesh/VoxelJunction.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"
#include "Ksolve.h"

const unsigned int OFFNODE = ~0;

// static function
SrcFinfo2< Id, vector< double > >* Ksolve::xComptOut() {
	static SrcFinfo2< Id, vector< double > > xComptOut( "xComptOut",
		"Sends 'n' of all molecules participating in cross-compartment "
		"reactions between any juxtaposed voxels between current compt "
		"and another compartment. This includes molecules local to this "
		"compartment, as well as proxy molecules belonging elsewhere. "
		"A(t+1) = (Alocal(t+1) + AremoteProxy(t+1)) - Alocal(t) "
		"A(t+1) = (Aremote(t+1) + Aproxy(t+1)) - Aproxy(t) "
		"Then we update A on the respective solvers with: "
		"Alocal(t+1) = Aproxy(t+1) = A(t+1) "
		"This is equivalent to sending dA over on each timestep. "
   	);
	return &xComptOut;
}

const Cinfo* Ksolve::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		
		static ValueFinfo< Ksolve, string > method (
			"method",
			"Integration method, using GSL. So far only explict. Options are:"
			"rk5: The default Runge-Kutta-Fehlberg 5th order adaptive dt method"
			"gsl: alias for the above"
			"rk4: The Runge-Kutta 4th order fixed dt method"
			"rk2: The Runge-Kutta 2,3 embedded fixed dt method"
			"rkck: The Runge-Kutta Cash-Karp (4,5) method"
			"rk8: The Runge-Kutta Prince-Dormand (8,9) method" ,
			&Ksolve::setMethod,
			&Ksolve::getMethod
		);
		
		static ValueFinfo< Ksolve, double > epsAbs (
			"epsAbs",
			"Absolute permissible integration error range.",
			&Ksolve::setEpsAbs,
			&Ksolve::getEpsAbs
		);
		
		static ValueFinfo< Ksolve, double > epsRel (
			"epsRel",
			"Relative permissible integration error range.",
			&Ksolve::setEpsRel,
			&Ksolve::getEpsRel
		);
		
		static ValueFinfo< Ksolve, Id > compartment(
			"compartment",
			"Compartment in which the Ksolve reaction system lives.",
			&Ksolve::setCompartment,
			&Ksolve::getCompartment
		);

		static ReadOnlyValueFinfo< Ksolve, unsigned int > numLocalVoxels(
			"numLocalVoxels",
			"Number of voxels in the core reac-diff system, on the "
			"current solver. ",
			&Ksolve::getNumLocalVoxels
		);
		static LookupValueFinfo< 
				Ksolve, unsigned int, vector< double > > nVec(
			"nVec",
			"vector of pool counts. Index specifies which voxel.",
			&Ksolve::setNvec,
			&Ksolve::getNvec
		);
		static ValueFinfo< Ksolve, unsigned int > numAllVoxels(
			"numAllVoxels",
			"Number of voxels in the entire reac-diff system, "
			"including proxy voxels to represent abutting compartments.",
			&Ksolve::setNumAllVoxels,
			&Ksolve::getNumAllVoxels
		);

		static ValueFinfo< Ksolve, unsigned int > numPools(
			"numPools",
			"Number of molecular pools in the entire reac-diff system, "
			"including variable, function and buffered.",
			&Ksolve::setNumPools,
			&Ksolve::getNumPools
		);

		static ReadOnlyValueFinfo< Ksolve, double > estimatedDt(
			"estimatedDt",
			"Estimated timestep for reac system based on Euler error",
			&Ksolve::getEstimatedDt
		);
		static ReadOnlyValueFinfo< Ksolve, Id > stoich(
			"stoich",
			"Id for stoichiometry object tied to this Ksolve",
			&Ksolve::getStoich
		);


		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////

		static DestFinfo process( "process",
			"Handles process call from Clock",
			new ProcOpFunc< Ksolve >( &Ksolve::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call from Clock",
			new ProcOpFunc< Ksolve >( &Ksolve::reinit ) );

		static DestFinfo initProc( "initProc",
			"Handles initProc call from Clock",
			new ProcOpFunc< Ksolve >( &Ksolve::initProc ) );
		static DestFinfo initReinit( "initReinit",
			"Handles initReinit call from Clock",
			new ProcOpFunc< Ksolve >( &Ksolve::initReinit ) );
		
		static DestFinfo voxelVol( "voxelVol",
			"Handles updates to all voxels. Comes from parent "
			"ChemCompt object.",
			new OpFunc1< Ksolve, vector< double > >( 
					&Ksolve::updateVoxelVol )
		);
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit. These are used for "
			"all regular Ksolve calculations including interfacing with "
			"the diffusion calculations by a Dsolve.",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		static Finfo* initShared[] = {
			&initProc, &initReinit
		};
		static SharedFinfo init( "init",
			"Shared message for initProc and initReinit. This is used"
		    " when the system has cross-compartment reactions. ",
			initShared, sizeof( initShared ) / sizeof( const Finfo* )
		);

		static DestFinfo xComptIn( "xComptIn",
			"Handles arriving pool 'n' values used in cross-compartment "
			"reactions.",
			new EpFunc2< Ksolve, Id, vector< double > >( &Ksolve::xComptIn )
		);
		static Finfo* xComptShared[] = {
			xComptOut(), &xComptIn
		};
		static SharedFinfo xCompt( "xCompt",
			"Shared message for pool exchange for cross-compartment "
			"reactions. Exchanges latest values of all pools that "
			"participate in such reactions.",
			xComptShared, sizeof( xComptShared ) / sizeof( const Finfo* )
		);

	static Finfo* ksolveFinfos[] =
	{
		&method,			// Value
		&epsAbs,			// Value
		&epsRel,			// Value
		&compartment,		// Value
		&numLocalVoxels,	// ReadOnlyValue
		&nVec,				// LookupValue
		&numAllVoxels,		// ReadOnlyValue
		&numPools,			// Value
		&estimatedDt,		// ReadOnlyValue
		&stoich,			// ReadOnlyValue
		&voxelVol,			// DestFinfo
		&xCompt,			// SharedFinfo
		&proc,				// SharedFinfo
		&init,				// SharedFinfo
	};
	
	static Dinfo< Ksolve > dinfo;
	static  Cinfo ksolveCinfo(
		"Ksolve",
		Neutral::initCinfo(),
		ksolveFinfos,
		sizeof(ksolveFinfos)/sizeof(Finfo *),
		&dinfo
	);

	return &ksolveCinfo;
}

static const Cinfo* ksolveCinfo = Ksolve::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

Ksolve::Ksolve()
	: 
		method_( "rk5" ),
		epsAbs_( 1e-4 ),
		epsRel_( 1e-6 ),
		pools_( 1 ),
		startVoxel_( 0 ),
		stoich_(),
		stoichPtr_( 0 ),
		dsolve_(),
		compartment_(),
		dsolvePtr_( 0 ),
		isBuilt_( false )
{;}

Ksolve::~Ksolve()
{;}

//////////////////////////////////////////////////////////////
// Field Access functions
//////////////////////////////////////////////////////////////

string Ksolve::getMethod() const
{
	return method_;
}

void Ksolve::setMethod( string method )
{
	if ( method == "rk5" || method == "gsl" ) {
		method_ = "rk5";
	} else if ( method == "rk4"  || method == "rk2" || 
					method == "rk8" || method == "rkck" ) {
		method_ = method;
	} else {
		cout << "Warning: Ksolve::setMethod: '" << method << 
				"' not known, using rk5\n";
		method_ = "rk5";
	}
}

double Ksolve::getEpsAbs() const
{
	return epsAbs_;
}

void Ksolve::setEpsAbs( double epsAbs )
{
	if ( epsAbs < 0 ) {
			epsAbs_ = 1.0e-4;
	} else {
		epsAbs_ = epsAbs;
	}
}


double Ksolve::getEpsRel() const
{
	return epsRel_;
}

void Ksolve::setEpsRel( double epsRel )
{
	if ( epsRel < 0 ) {
			epsRel_ = 1.0e-6;
	} else {
		epsRel_ = epsRel;
	}
}

Id Ksolve::getStoich() const
{
	return stoich_;
}

#ifdef USE_GSL
void innerSetMethod( OdeSystem& ode, const string& method )
{
	ode.method = method;
	if ( method == "rk5" ) {
		ode.gslStep = gsl_odeiv2_step_rkf45;
	} else if ( method == "rk4" ) {
		ode.gslStep = gsl_odeiv2_step_rk4;
	} else if ( method == "rk2" ) {
		ode.gslStep = gsl_odeiv2_step_rk2;
	} else if ( method == "rkck" ) {
		ode.gslStep = gsl_odeiv2_step_rkck;
	} else if ( method == "rk8" ) {
		ode.gslStep = gsl_odeiv2_step_rk8pd;
	} else {
		ode.gslStep = gsl_odeiv2_step_rkf45;
	}
}
#endif

void Ksolve::setStoich( Id stoich )
{
	assert( stoich.element()->cinfo()->isA( "Stoich" ) );
	stoich_ = stoich;
	stoichPtr_ = reinterpret_cast< Stoich* >( stoich.eref().data() );
	if ( !isBuilt_ ) {
		OdeSystem ode;
		ode.epsAbs = epsAbs_;
		ode.epsRel = epsRel_;
		// ode.initStepSize = getEstimatedDt();
		ode.initStepSize = 0.01; // This will be overridden at reinit.
#ifdef USE_GSL
		innerSetMethod( ode, method_ );
		ode.gslSys.function = &VoxelPools::gslFunc;
   		ode.gslSys.jacobian = 0;
		ode.gslSys.dimension = stoichPtr_->getNumAllPools() + stoichPtr_->getNumProxyPools();
		innerSetMethod( ode, method_ );
		unsigned int numVoxels = pools_.size();
		for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
   			ode.gslSys.params = &pools_[i];
			pools_[i].setStoich( stoichPtr_, &ode );
			// pools_[i].setIntDt( ode.initStepSize ); // We're setting it up anyway
		}
		isBuilt_ = true;
#endif
	}
}

Id Ksolve::getDsolve() const
{
	return dsolve_;
}

void Ksolve::setDsolve( Id dsolve )
{
	if ( dsolve == Id () ) {
		dsolvePtr_ = 0;
		dsolve_ = Id();
	} else if ( dsolve.element()->cinfo()->isA( "Dsolve" ) ) {
		dsolve_ = dsolve;
		dsolvePtr_ = reinterpret_cast< ZombiePoolInterface* >( 
						dsolve.eref().data() );
	} else {
		cout << "Warning: Ksolve::setDsolve: Object '" << dsolve.path() <<
				"' should be class Dsolve, is: " << 
				dsolve.element()->cinfo()->name() << endl;
	}
}

Id Ksolve::getCompartment() const
{
	return compartment_;
}

void Ksolve::setCompartment( Id compt )
{
	isBuilt_ = false; // We will have to now rebuild the whole thing.
	if ( compt.element()->cinfo()->isA( "ChemCompt" ) ) {
		compartment_ = compt;
		vector< double > vols = 
			Field< vector < double > >::get( compt, "voxelVolume" );
		if ( vols.size() > 0 ) {
			pools_.resize( vols.size() );
			for ( unsigned int i = 0; i < vols.size(); ++i ) {
				pools_[i].setVolume( vols[i] );
			}
		}
	}
}

unsigned int Ksolve::getNumLocalVoxels() const
{
	return pools_.size();
}

unsigned int Ksolve::getNumAllVoxels() const
{
	return pools_.size(); // Need to redo.
}

// If we're going to do this, should be done before the zombification.
void Ksolve::setNumAllVoxels( unsigned int numVoxels )
{
	if ( numVoxels == 0 ) {
		return;
	}
	pools_.resize( numVoxels );
}

vector< double > Ksolve::getNvec( unsigned int voxel) const
{
	static vector< double > dummy;
	if ( voxel < pools_.size() ) {
		return const_cast< VoxelPools* >( &( pools_[ voxel ] ) )->Svec();
	}
	return dummy;
}

void Ksolve::setNvec( unsigned int voxel, vector< double > nVec )
{
	if ( voxel < pools_.size() ) {
		if ( nVec.size() != pools_[voxel].size() ) {
			cout << "Warning: Ksolve::setNvec: size mismatch ( " <<
				nVec.size() << ", " << pools_[voxel].size() << ")\n";
			return;
		}
		double* s = pools_[voxel].varS();
		for ( unsigned int i = 0; i < nVec.size(); ++i )
			s[i] = nVec[i];
	}
}


double Ksolve::getEstimatedDt() const
{
	static const double EPSILON = 1e-15;
	vector< double > s( stoichPtr_->getNumAllPools() + 
					stoichPtr_->getNumProxyPools(), 1.0 );
	vector< double > v( stoichPtr_->getNumRates(), 0.0 );
	double maxVel = 0.0;
	if ( pools_.size() > 0.0 ) {
		pools_[0].updateReacVelocities( &s[0], v );
		for ( vector< double >::iterator 
						i = v.begin(); i != v.end(); ++i )
				if ( maxVel < *i )
					maxVel = *i;
	}
	if ( maxVel < EPSILON )
	 	return 0.1; // Based on typical sig pathway reac rates.
	// Heuristic: the largest velocity times dt should be 10% of mol conc.
	return 0.1 / maxVel; 
}

//////////////////////////////////////////////////////////////
// Process operations.
//////////////////////////////////////////////////////////////
void Ksolve::process( const Eref& e, ProcPtr p )
{
	// Second, handle incoming diffusion values, update S with those.
	if ( dsolvePtr_ ) {
		vector< double > dvalues( 4 );
		dvalues[0] = 0;
		dvalues[1] = getNumLocalVoxels();
		dvalues[2] = 0;
		dvalues[3] = stoichPtr_->getNumVarPools();
		dsolvePtr_->getBlock( dvalues );
		/*
		getBlock( kvalues );
		vector< double >::iterator d = dvalues.begin() + 4;
		for ( vector< double >::iterator 
				k = kvalues.begin() + 4; k != kvalues.end(); ++k )
				*k++ = ( *k + *d )/2.0
		setBlock( kvalues );
		*/
		setBlock( dvalues );
	}
	// First, take the arrived xCompt reac values and update S with them.
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		const XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			pools_[xf.xferVoxel[j]].xferIn( 
					xf.xferPoolIdx, xf.values, xf.lastValues, j );
		}
	}
	// Third, record the current value of pools as the reference for the
	// next cycle.
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			pools_[xf.xferVoxel[j]].xferOut( j, xf.lastValues, xf.xferPoolIdx );
		}
	}

	// Fourth, do the numerical integration for all reactions.
	for ( vector< VoxelPools >::iterator 
				i = pools_.begin(); i != pools_.end(); ++i ) {
		i->advance( p );
	}
	// Finally, assemble and send the integrated values off for the Dsolve.
	if ( dsolvePtr_ ) {
		vector< double > kvalues( 4 );
		kvalues[0] = 0;
		kvalues[1] = getNumLocalVoxels();
		kvalues[2] = 0;
		kvalues[3] = stoichPtr_->getNumVarPools();
		getBlock( kvalues );
		dsolvePtr_->setBlock( kvalues );
	}
}

void Ksolve::reinit( const Eref& e, ProcPtr p )
{
	assert( stoichPtr_ );
	if ( isBuilt_ ) {
		for ( unsigned int i = 0 ; i < pools_.size(); ++i )
			pools_[i].reinit( p->dt );
	} else {
		cout << "Warning:Ksolve::reinit: Reaction system not initialized\n";
		return;
	}
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		const XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			pools_[xf.xferVoxel[j]].xferInOnlyProxies( 
					xf.xferPoolIdx, xf.values, 
					stoichPtr_->getNumProxyPools(),
					j );
		}
	}
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			pools_[xf.xferVoxel[j]].xferOut( 
					j, xf.lastValues, xf.xferPoolIdx );
		}
	}
}
//////////////////////////////////////////////////////////////
// init operations.
//////////////////////////////////////////////////////////////
void Ksolve::initProc( const Eref& e, ProcPtr p )
{
	// vector< vector< double > > values( xfer_.size() );
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		XferInfo& xf = xfer_[i];
		unsigned int size = xf.xferPoolIdx.size() * xf.xferVoxel.size();
		// values[i].resize( size, 0.0 );
		vector< double > values( size, 0.0 );
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			unsigned int vox = xf.xferVoxel[j];
			pools_[vox].xferOut( j, values, xf.xferPoolIdx );
		}
		xComptOut()->sendTo( e, xf.ksolve, e.id(), values );
	}
	// xComptOut()->sendVec( e, values );
}

void Ksolve::initReinit( const Eref& e, ProcPtr p )
{
	for ( unsigned int i = 0 ; i < pools_.size(); ++i ) {
		pools_[i].reinit( p->dt );
	}
	// vector< vector< double > > values( xfer_.size() );
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		XferInfo& xf = xfer_[i];
		unsigned int size = xf.xferPoolIdx.size() * xf.xferVoxel.size();
		xf.lastValues.assign( size, 0.0 );
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) {
			unsigned int vox = xf.xferVoxel[j];
			pools_[ vox ].xferOut( j, xf.lastValues, xf.xferPoolIdx );
			// values[i] = xf.lastValues;
		}
		xComptOut()->sendTo( e, xf.ksolve, e.id(), xf.lastValues );
	}
	// xComptOut()->sendVec( e, values );
}

/**
 * updateRateTerms obtains the latest parameters for the rates_ vector,
 * and has each of the pools update its parameters including rescaling
 * for volumes.
 */
void Ksolve::updateRateTerms( unsigned int index )
{
	if ( index == ~0U ) {
		// unsigned int numCrossRates = stoichPtr_->getNumRates() - stoichPtr_->getNumCoreRates();
		for ( unsigned int i = 0 ; i < pools_.size(); ++i ) {
			// pools_[i].resetXreacScale( numCrossRates );
			pools_[i].updateAllRateTerms( stoichPtr_->getRateTerms(),
						   stoichPtr_->getNumCoreRates() );
		}
	} else if ( index < stoichPtr_->getNumRates() ) {
		for ( unsigned int i = 0 ; i < pools_.size(); ++i )
			pools_[i].updateRateTerms( stoichPtr_->getRateTerms(),
							stoichPtr_->getNumCoreRates(), index );
	}
}


//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////

unsigned int Ksolve::getPoolIndex( const Eref& e ) const
{
	return stoichPtr_->convertIdToPoolIndex( e.id() );
}

unsigned int Ksolve::getVoxelIndex( const Eref& e ) const
{
	unsigned int ret = e.dataIndex();
	if ( ret < startVoxel_  || ret >= startVoxel_ + pools_.size() ) 
		return OFFNODE;
	return ret - startVoxel_;
}

//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void Ksolve::setN( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setN( getPoolIndex( e ), v );
}

double Ksolve::getN( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getN( getPoolIndex( e ) );
	return 0.0;
}

void Ksolve::setNinit( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setNinit( getPoolIndex( e ), v );
}

double Ksolve::getNinit( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getNinit( getPoolIndex( e ) );
	return 0.0;
}

void Ksolve::setDiffConst( const Eref& e, double v )
{
		; // Do nothing.
}

double Ksolve::getDiffConst( const Eref& e ) const
{
		return 0;
}

void Ksolve::setNumPools( unsigned int numPoolSpecies )
{
	unsigned int numVoxels = pools_.size();
	for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
		pools_[i].resizeArrays( numPoolSpecies );
	}
}

unsigned int Ksolve::getNumPools() const
{
	if ( pools_.size() > 0 )
		return pools_[0].size();
	return 0;
}

void Ksolve::getBlock( vector< double >& values ) const
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );
	values.resize( 4 + numVoxels * numPools );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		const double* v = pools_[ startVoxel + i ].S();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			values[ 4 + j * numVoxels + i]  = v[ j + startPool ];
		}
	}
}

void Ksolve::setBlock( const vector< double >& values )
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );
	assert( values.size() == 4 + numVoxels * numPools );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		double* v = pools_[ startVoxel + i ].varS();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			v[ j + startPool ] = values[ 4 + j * numVoxels + i ];
		}
	}
}

//////////////////////////////////////////////////////////////////////////
void Ksolve::updateVoxelVol( vector< double > vols )
{
	// For now we assume identical numbers of voxels. Also assume
	// identical voxel junctions. But it should not be too hard to
	// update those too.
	if ( vols.size() == pools_.size() ) {
		for ( unsigned int i = 0; i < vols.size(); ++i ) {
			pools_[i].setVolumeAndDependencies( vols[i] );
		}
		stoichPtr_->setupCrossSolverReacVols();
		updateRateTerms( ~0U );
	}
}

//////////////////////////////////////////////////////////////////////////
// cross-compartment reaction stuff.
//////////////////////////////////////////////////////////////////////////
// void Ksolve::xComptIn( const Eref& e, const ObjId& src, 
// vector< double > values )
void Ksolve::xComptIn( const Eref& e, Id srcKsolve,
	vector< double > values )
{
		/*
	assert( values.size() == xComptData_.size() );
	for ( vector< VoxelPools >::iterator
			i = pools_.begin(); i != pools_.end(); ++i )
		i->mergeProxy( values, xComptData_ );
		*/
	// Identify the xfer_ that maps to the srcKsolve. Assume only a small
	// number of them, otherwise we should use a map.
	unsigned int comptIdx ;
	for ( comptIdx = 0 ; comptIdx < xfer_.size(); ++comptIdx ) {
		if ( xfer_[comptIdx].ksolve == srcKsolve ) break;
	}
	assert( comptIdx != xfer_.size() );
	XferInfo& xf = xfer_[comptIdx];
	// assert( values.size() == xf.values.size() );
	xf.values = values;
//	xfer_[comptIdx].lastValues = values;
}

void Ksolve::xComptOut( const Eref& e )
{
	for ( vector< XferInfo >::const_iterator i = 
			xfer_.begin(); i != xfer_.end(); ++i ) {
		vector< double > values( i->lastValues.size(), 0.0 );
		for ( unsigned int j = 0; j < i->xferVoxel.size(); ++j ) {
			pools_[ i->xferVoxel[j] ].xferOut( j, values, i->xferPoolIdx );
		}
		// Use sendTo or sendVec to send to specific ksolves.
		xComptOut()->sendTo( e, i->ksolve, e.id(), values );
	}
}

/////////////////////////////////////////////////////////////////////
// Functions for setup of cross-compartment transfer.
/////////////////////////////////////////////////////////////////////
/**
 * Figures out which voxels are involved in cross-compt reactions. Stores
 * in the appropriate xfer_ entry.
 */
void Ksolve::assignXferVoxels( unsigned int xferCompt )
{
	assert( xferCompt < xfer_.size() );
	XferInfo& xf = xfer_[xferCompt];
	for ( unsigned int i = 0; i < pools_.size(); ++i ) {
		if ( pools_[i].hasXfer( xferCompt ) )
			xf.xferVoxel.push_back( i );
	}
	xf.values.resize( xf.xferVoxel.size() & xf.xferPoolIdx.size(), 0 );
	xf.lastValues.resize( xf.xferVoxel.size() & xf.xferPoolIdx.size(), 0 );
}

/**
 * Figures out indexing of the array of transferred pool n's used to fill
 * out proxies on each timestep.
 */
void Ksolve::assignXferIndex( unsigned int numProxyMols, 
		unsigned int xferCompt,
		const vector< vector< unsigned int > >& voxy )
{
	unsigned int idx = 0;
	for ( unsigned int i = 0; i < voxy.size(); ++i ) {
		const vector< unsigned int >& rpv = voxy[i];
		if ( rpv.size()  > 0) { // There would be a transfer here
			for ( vector< unsigned int >::const_iterator
					j = rpv.begin(); j != rpv.end(); ++j ) {
				pools_[*j].addProxyTransferIndex( xferCompt, idx );
			}
			idx += numProxyMols;
		}
	}
}

/**
 * This function sets up the information about the pool transfer for
 * cross-compartment reactions. It consolidates the transfer into a
 * distinct vector for each direction of the transfer between each coupled 
 * pair of Ksolves.
 * This one call sets up the information about transfer on both sides
 * of the junction(s) between current Ksolve and otherKsolve.
 *
 * VoxelJunction refers to a specific junction between compartments.
 * It has the following relevant fields:
 * 	first, second: VoxelIndex for the first and second compartments.
 * 	firstVol, secondVol: VoxelVolume for the first and second compartments.
 */
void Ksolve::setupXfer( Id myKsolve, Id otherKsolve, 
	unsigned int numProxyMols, const vector< VoxelJunction >& vj )
{
	const ChemCompt *myCompt = reinterpret_cast< const ChemCompt* >(
			compartment_.eref().data() );
	Ksolve* otherKsolvePtr = reinterpret_cast< Ksolve* >( 
					otherKsolve.eref().data() );
	const ChemCompt *otherCompt = reinterpret_cast< const ChemCompt* >(
			otherKsolvePtr->compartment_.eref().data() );
	// Use this so we can figure out what the other side will send.
	vector< vector< unsigned int > > proxyVoxy( myCompt->getNumEntries() );
	vector< vector< unsigned int > > reverseProxyVoxy( otherCompt->getNumEntries() );
	assert( xfer_.size() > 0 && otherKsolvePtr->xfer_.size() > 0 );
	unsigned int myKsolveIndex = xfer_.size() -1;
	unsigned int otherKsolveIndex = otherKsolvePtr->xfer_.size() -1;
	for ( unsigned int i = 0; i < vj.size(); ++i ) {
		unsigned int j = vj[i].first;
		assert( j < pools_.size() ); // Check voxel indices.
		proxyVoxy[j].push_back( vj[i].second );
		pools_[j].addProxyVoxy( myKsolveIndex, 
						otherKsolvePtr->compartment_, vj[i].second);
		unsigned int k = vj[i].second;
		assert( k < otherCompt->getNumEntries() );
		reverseProxyVoxy[k].push_back( vj[i].first );
		otherKsolvePtr->pools_[k].addProxyVoxy( 
			otherKsolveIndex, compartment_, vj[i].first );
	}

	// Build the indexing for the data values to transfer on each timestep
	assignXferIndex( numProxyMols, myKsolveIndex, reverseProxyVoxy );
	otherKsolvePtr->assignXferIndex( 
			numProxyMols, otherKsolveIndex, proxyVoxy );
	// Figure out which voxels participate in data transfer.
	assignXferVoxels( myKsolveIndex );
	otherKsolvePtr->assignXferVoxels( otherKsolveIndex );
}

/**
 *
 * At every junction, all the reacs that inhabit the two compartments need
 * to have their cross-terms set up. Go through each VJ and set up the
 * xReacScaling for each rate Term that applies.
void Ksolve::buildCrossReacVolScaling( Id otherKsolve,
				const vector< VoxelJunction >& vj )
{
	Ksolve* otherKsolvePtr = reinterpret_cast< Ksolve* >( 
					otherKsolve.eref().data() );

	vector< pair< Id, Id > > xr; 
	// This returns cross-reac compartments associated with each RateTerm
	stoichPtr_->comptsOnCrossReacTerms( xr );
	for ( unsigned int i = 0; i < xr.size(); ++i ) {
		if ( xr[i].first == otherKsolvePtr->compartment_ ||
			xr[i].second == otherKsolvePtr->compartment_ )
		{
			for ( unsigned int k = 0; k < vj.size(); ++k )
				pools_[vj[k].first].multiplyXreacScale( i, vj[k].secondVol);
		}
	}

	// Now do this for the otherKsolve.
	xr.clear(); 
	otherKsolvePtr->stoichPtr_->comptsOnCrossReacTerms( xr );
	for ( unsigned int i = 0; i < xr.size(); ++i ) {
		if ( xr[i].first == compartment_ ||
			xr[i].second == compartment_ )
		{
			for ( unsigned int k = 0; k < vj.size(); ++k )
				otherKsolvePtr->pools_[vj[k].second].multiplyXreacScale( i, vj[k].firstVol);
		}
	}
}
 */


/**
 * Builds up the list of proxy pools on either side of the junction,
 * and assigns to the XferInfo data structures for use during runtime.
 */
unsigned int Ksolve::assignProxyPools( const map< Id, vector< Id > >& xr,
				Id myKsolve, Id otherKsolve, Id otherComptId )
{
	map< Id, vector< Id > >::const_iterator i = xr.find( otherComptId );
	vector< Id > proxyMols;
	if ( i != xr.end() ) 
		proxyMols = i->second;
	Ksolve* otherKsolvePtr = reinterpret_cast< Ksolve* >( 
					otherKsolve.eref().data() );
		
	vector< Id > otherProxies = LookupField< Id, vector< Id > >::get( 
			otherKsolvePtr->stoich_, "proxyPools", stoich_ );

	proxyMols.insert( proxyMols.end(), 
					otherProxies.begin(), otherProxies.end() );
	// if ( proxyMols.size() == 0 )
		// return 0;
	sort( proxyMols.begin(), proxyMols.end() );
	xfer_.push_back( XferInfo( otherKsolve ) );

	otherKsolvePtr->xfer_.push_back( XferInfo( myKsolve ) );
	vector< unsigned int >& xfi = xfer_.back().xferPoolIdx;
	vector< unsigned int >& oxfi = otherKsolvePtr->xfer_.back().xferPoolIdx;
	xfi.resize( proxyMols.size() );
	oxfi.resize( proxyMols.size() );
	for ( unsigned int i = 0; i < xfi.size(); ++i ) {
		xfi[i] = stoichPtr_->convertIdToPoolIndex( proxyMols[i] );
		oxfi[i] = otherKsolvePtr->stoichPtr_->convertIdToPoolIndex( 
						proxyMols[i] );
	}
	return proxyMols.size();
}


// This function cleans out the RateTerms of cross reactions that 
// don't have anything to connect to.
// It should be called after all cross reacs have been assigned.
void Ksolve::filterCrossRateTerms( const vector< pair< Id, Id > >& xrt )
{
	for ( vector< VoxelPools >::iterator
			i = pools_.begin(); i != pools_.end(); ++i )
			i->filterCrossRateTerms( xrt );
}

/**
 * This function builds cross-solver reaction calculations. For the 
 * specified pair of stoichs (this->stoich_, otherStoich) it identifies
 * interacting molecules, finds where the junctions are, sets up the
 * info to build the data transfer vector, and sets up the transfer
 * itself.
 */
void Ksolve::setupCrossSolverReacs( const map< Id, vector< Id > >& xr,
	   Id otherStoich )
{
	const ChemCompt *myCompt = reinterpret_cast< const ChemCompt* >(
			compartment_.eref().data() );
	Id otherComptId = Field< Id >::get( otherStoich, "compartment" );
	Id myKsolve = Field< Id >::get( stoich_, "ksolve" );
	if ( myKsolve == Id() )
		return;
	Id otherKsolve = Field< Id >::get( otherStoich, "ksolve" );
	if ( otherKsolve == Id() ) 
		return;

	// Establish which molecules will be exchanged.
	unsigned int numPools = assignProxyPools( xr, myKsolve, otherKsolve, 
					otherComptId );
	if ( numPools == 0 ) return;

	// Then, figure out which voxels do the exchange.
	// Note that vj has a list of pairs of voxels on either side of a 
	// junction. If one voxel on self touches 5 voxels on other, then
	// there will be five entries in vj for this contact. 
	// If one voxel on self touches two different compartments, then
	// a distinct vj vector must be built for those contacts.
	const ChemCompt *otherCompt = reinterpret_cast< const ChemCompt* >(
			otherComptId.eref().data() );
	vector< VoxelJunction > vj;
	myCompt->matchMeshEntries( otherCompt, vj );
	if ( vj.size() == 0 )
		return;

	// This function sets up the information about the pool transfer on
	// both sides.
	setupXfer( myKsolve, otherKsolve, numPools, vj );

	/// This sets up the volume scaling from cross reac terms
	// Deprecated. Handled by setupCrossSolverReacVols. 
	// buildCrossReacVolScaling( otherKsolve, vj );

	// Here we set up the messaging.
	Shell *shell = reinterpret_cast< Shell* >( Id().eref().data() );
	shell->doAddMsg( "Single", myKsolve, "xCompt", otherKsolve, "xCompt" );
}

/** 
 * This fills the vols vector with the volume of the abutting 
 * voxel on compt. If there are no abutting voxels on a given
 * voxel then that entry of the vols vector is filled with a zero.
 * There is exactly one vols entry for each voxel of the local compt.
 */
void Ksolve::matchJunctionVols( vector< double >& vols, Id otherComptId ) 
		const
{
	vols.resize( pools_.size() );
	for ( unsigned int i = 0; i < vols.size(); ++i )
		vols[i] = pools_[i].getVolume();
	if ( otherComptId == compartment_ ) {
		// This may legitimately happen if the substrate or product is
		// on the local compartment.
		// cout << "Warning: Ksolve::matchJunctionVols: self compt.\n";
		return;
	}
	const ChemCompt *myCompt = reinterpret_cast< const ChemCompt* >(
			compartment_.eref().data() );
	const ChemCompt *otherCompt = reinterpret_cast< const ChemCompt* >(
			otherComptId.eref().data() );
	vector< VoxelJunction > vj;
	myCompt->matchMeshEntries( otherCompt, vj );
	if ( vj.size() == 0 )
		return;
	for ( vector< VoxelJunction >::const_iterator 
			i = vj.begin(); i != vj.end(); ++i ) {
		assert( i->first < vols.size() );
		/*
		if ( !doubleEq( vols[ i->first ], 0.0 ) )
			cout << "Warning: Ksolve::matchJuntionVols: repeated voxel\n";
			*/
		vols[ i->first ] = i->secondVol;
	}
}

/**
 * This function builds cross-solver reaction volume scaling. 
 */
void Ksolve::setupCrossSolverReacVols( 
	const vector< vector< Id > >& subCompts, 
	const vector< vector< Id > >& prdCompts )
{
	map< Id, vector< double > > comptVolMap;
	unsigned int numOffSolverReacs = 
			stoichPtr_->getNumRates() - stoichPtr_->getNumCoreRates();
	assert( subCompts.size() == numOffSolverReacs );
	assert( prdCompts.size() == numOffSolverReacs );
	for ( unsigned int i = 0 ; i < pools_.size(); ++i )
		pools_[i].resetXreacScale( numOffSolverReacs );
	for( unsigned int i = 0; i < numOffSolverReacs; ++i ) {
		for ( unsigned int j = 0; j < subCompts[i].size(); ++j ) {
			map< Id, vector< double > >::iterator cvi;
			vector< double > vols;
			cvi = comptVolMap.find( subCompts[i][j] );
			if ( cvi == comptVolMap.end() ) {
				matchJunctionVols( vols, subCompts[i][j] );
				comptVolMap[subCompts[i][j]] = vols;
			} else {
				vols = cvi->second;
			}
			assert( vols.size() == pools_.size() );
			for ( unsigned int k = 0; k < vols.size(); ++k )
				pools_[k].forwardReacVolumeFactor( i, vols[k] );
		}

		for ( unsigned int j = 0; j < prdCompts[i].size(); ++j ) {
			map< Id, vector< double > >::iterator cvi;
			vector< double > vols;
			cvi = comptVolMap.find( prdCompts[i][j] );
			if ( cvi == comptVolMap.end() ) {
				matchJunctionVols( vols, prdCompts[i][j] );
				comptVolMap[prdCompts[i][j]] = vols;
			} else {
				vols = cvi->second;
			}
			assert( vols.size() == pools_.size() );
			for ( unsigned int k = 0; k < vols.size(); ++k )
				pools_[k].backwardReacVolumeFactor( i, vols[k] );
		}
	}
}

void Ksolve::print() const
{
	cout << "path = " << stoichPtr_->getKsolve().path() << 
			", numPools = " << pools_.size() << "\n";
	for ( unsigned int i = 0; i < pools_.size(); ++i ) {
		cout << "pools[" << i << "] contents = ";
		pools_[i].print();
	}
	cout << "method = " << method_ << ", stoich=" << stoich_.path() <<endl;
	cout << "dsolve = " << dsolve_.path() << endl;
	cout << "compartment = " << compartment_.path() << endl;
	cout << "xfer summary: numxfer = " << xfer_.size() << "\n";
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		cout << "xfer_[" << i << "] numValues=" << 
			xfer_[i].values.size() <<
			", xferPoolIdx.size = " << xfer_[i].xferPoolIdx.size() <<
			", xferVoxel.size = " << xfer_[i].xferVoxel.size() << endl;
	}
	cout << "xfer details:\n";
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) {
		cout << "xfer_[" << i << "] xferPoolIdx=\n";
		const vector< unsigned int>& xi = xfer_[i].xferPoolIdx;
		for ( unsigned int j = 0; j << xi.size(); ++j )
			cout << "	" << xi[j];
		cout << "\nxfer_[" << i << "] xferVoxel=\n";
		const vector< unsigned int>& xv = xfer_[i].xferVoxel;
		for ( unsigned int j = 0; j << xv.size(); ++j )
			cout << "	" << xv[j];
	}
}
