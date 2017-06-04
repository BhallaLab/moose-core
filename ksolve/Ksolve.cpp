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
#include "mesh/VoxelJunction.h"
#include "XferInfo.h"
#include "ZombiePoolInterface.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "shell/Shell.h"

#include "mesh/MeshEntry.h"
#include "mesh/Boundary.h"
#include "mesh/ChemCompt.h"
#include "utility/strutil.h"
#include "Ksolve.h"

#include <omp.h>
#include <thread>
#include <atomic>
#include <future>

const unsigned int OFFNODE = ~0;

// static function
SrcFinfo2< Id, vector< double > >* Ksolve::xComptOut()
{
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
    static Finfo* procShared[] =
    {
        &process, &reinit
    };
    static SharedFinfo proc( "proc",
                             "Shared message for process and reinit. These are used for "
                             "all regular Ksolve calculations including interfacing with "
                             "the diffusion calculations by a Dsolve.",
                             procShared, sizeof( procShared ) / sizeof( const Finfo* )
                           );
    static Finfo* initShared[] =
    {
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
    static Finfo* xComptShared[] =
    {
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
#if USE_GSL
    method_( "rk5" ),
#elif USE_BOOST
    method_( "rk5a" ),
#endif
    epsAbs_( 1e-7 ),
    epsRel_( 1e-7 ),
    voxelPools_( 1 ),
    startVoxel_( 0 ),
    dsolve_(),
    dsolvePtr_( 0 )
{
    ;
}

Ksolve::~Ksolve()
{
    //for (size_t i = 0; i < num_threads_; i++) 
        //threads_[i]->join( );
}

//////////////////////////////////////////////////////////////
// Field Access functions
//////////////////////////////////////////////////////////////

string Ksolve::getMethod() const
{
    return method_;
}

void Ksolve::setMethod( string method )
{
#if USE_GSL
    if ( method == "rk5" || method == "gsl" )
    {
        method_ = "rk5";
    }
    else if ( method == "rk4"  || method == "rk2" ||
              method == "rk8" || method == "rkck" )
    {
        method_ = method;
    }
    else
    {
        cout << "Warning: Ksolve::setMethod: '" << method <<
             "' not known, using rk5\n";
        method_ = "rk5";
    }
#elif USE_BOOST
    // TODO: Check for boost related methods.
    method_ = method;
#endif
}

double Ksolve::getEpsAbs() const
{
    return epsAbs_;
}

void Ksolve::setEpsAbs( double epsAbs )
{
    if ( epsAbs < 0 )
        epsAbs_ = 1.0e-4;
    else
        epsAbs_ = epsAbs;
}


double Ksolve::getEpsRel() const
{
    return epsRel_;
}

void Ksolve::setEpsRel( double epsRel )
{
    if ( epsRel < 0 )
    {
        epsRel_ = 1.0e-6;
    }
    else
    {
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
    if ( method == "rk5" )
    {
        ode.gslStep = gsl_odeiv2_step_rkf45;
    }
    else if ( method == "rk4" )
    {
        ode.gslStep = gsl_odeiv2_step_rk4;
    }
    else if ( method == "rk2" )
    {
        ode.gslStep = gsl_odeiv2_step_rk2;
    }
    else if ( method == "rkck" )
    {
        ode.gslStep = gsl_odeiv2_step_rkck;
    }
    else if ( method == "rk8" )
    {
        ode.gslStep = gsl_odeiv2_step_rk8pd;
    }
    else
    {
        ode.gslStep = gsl_odeiv2_step_rkf45;
    }
}
#endif

void Ksolve::setStoich( Id stoich )
{
    assert( stoich.element()->cinfo()->isA( "Stoich" ) );
    stoich_ = stoich;
    stoichPtr_ = reinterpret_cast< Stoich* >( stoich.eref().data() );
    if ( !isBuilt_ )
    {
        OdeSystem ode;
        ode.epsAbs = epsAbs_;
        ode.epsRel = epsRel_;
        // ode.initStepSize = getEstimatedDt();
        ode.initStepSize = 0.01; // This will be overridden at reinit.
        ode.method = method_;
#ifdef USE_GSL
        ode.gslSys.dimension = stoichPtr_->getNumAllPools();
        if ( ode.gslSys.dimension == 0 ) {
			stoichPtr_ = 0;
            return; // No pools, so don't bother.
		}
        innerSetMethod( ode, method_ );
        ode.gslSys.function = &VoxelPools::gslFunc;
        ode.gslSys.jacobian = 0;
        innerSetMethod( ode, method_ );
        unsigned int numVoxels = voxelPools_.size();
        for ( unsigned int i = 0 ; i < numVoxels; ++i )
        {
            ode.gslSys.params = &voxelPools_[i];
            voxelPools_[i].setStoich( stoichPtr_, &ode );
            // voxelPools_[i].setIntDt( ode.initStepSize ); // We're setting it up anyway
        }
#elif USE_BOOST
        ode.dimension = stoichPtr_->getNumAllPools();
        ode.boostSys.epsAbs = epsAbs_;
        ode.boostSys.epsRel = epsRel_;
        ode.boostSys.method = method_;
        if ( ode.dimension == 0 )
            return; // No pools, so don't bother.
        unsigned int numVoxels = voxelPools_.size();
        for ( unsigned int i = 0 ; i < numVoxels; ++i )
        {
            ode.boostSys.params = &voxelPools_[i];
            voxelPools_[i].setStoich( stoichPtr_, &ode );
        }
#endif
        isBuilt_ = true;
    }
}

Id Ksolve::getDsolve() const
{
    return dsolve_;
}

void Ksolve::setDsolve( Id dsolve )
{
    if ( dsolve == Id () )
    {
        dsolvePtr_ = 0;
        dsolve_ = Id();
    }
    else if ( dsolve.element()->cinfo()->isA( "Dsolve" ) )
    {
        dsolve_ = dsolve;
        dsolvePtr_ = reinterpret_cast< ZombiePoolInterface* >(
                         dsolve.eref().data() );
    }
    else
    {
        cout << "Warning: Ksolve::setDsolve: Object '" << dsolve.path() <<
             "' should be class Dsolve, is: " <<
             dsolve.element()->cinfo()->name() << endl;
    }
}

unsigned int Ksolve::getNumLocalVoxels() const
{
    return voxelPools_.size();
}

unsigned int Ksolve::getNumAllVoxels() const
{
    return voxelPools_.size(); // Need to redo.
}

// If we're going to do this, should be done before the zombification.
void Ksolve::setNumAllVoxels( unsigned int numVoxels )
{
    if ( numVoxels == 0 )
    {
        return;
    }
    voxelPools_.resize( numVoxels );
}

vector< double > Ksolve::getNvec( unsigned int voxel) const
{
    static vector< double > dummy;
    if ( voxel < voxelPools_.size() )
    {
        return const_cast< VoxelPools* >( &( voxelPools_[ voxel ] ) )->Svec();
    }
    return dummy;
}

void Ksolve::setNvec( unsigned int voxel, vector< double > nVec )
{
    if ( voxel < voxelPools_.size() )
    {
        if ( nVec.size() != voxelPools_[voxel].size() )
        {
            cout << "Warning: Ksolve::setNvec: size mismatch ( " <<
                 nVec.size() << ", " << voxelPools_[voxel].size() << ")\n";
            return;
        }
        double* s = voxelPools_[voxel].varS();
        for ( unsigned int i = 0; i < nVec.size(); ++i )
            s[i] = nVec[i];
    }
}


double Ksolve::getEstimatedDt() const
{
    static const double EPSILON = 1e-15;
    vector< double > s( stoichPtr_->getNumAllPools(), 1.0 );
    vector< double > v( stoichPtr_->getNumRates(), 0.0 );
    double maxVel = 0.0;
    if ( voxelPools_.size() > 0.0 )
    {
        voxelPools_[0].updateReacVelocities( &s[0], v );
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
    if ( isBuilt_ == false )
        return;

    // First, handle incoming diffusion values, update S with those.
    if ( dsolvePtr_ )
    {
        vector< double > dvalues( 4 );
        dvalues[0] = 0;
        dvalues[1] = getNumLocalVoxels();
        dvalues[2] = 0;
        dvalues[3] = stoichPtr_->getNumVarPools();
        dsolvePtr_->getBlock( dvalues );

        /*
        vector< double >::iterator i = dvalues.begin() + 4;
        for ( ; i != dvalues.end(); ++i )
        	cout << *i << "	" << round( *i ) << endl;
        getBlock( kvalues );
        vector< double >::iterator d = dvalues.begin() + 4;
        for ( vector< double >::iterator
        		k = kvalues.begin() + 4; k != kvalues.end(); ++k )
        		*k++ = ( *k + *d )/2.0
        setBlock( kvalues );
        */
        setBlock( dvalues );
    }

    // Second, take the arrived xCompt reac values and update S with them.
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        const XferInfo& xf = xfer_[i];
        // cout << xfer_.size() << "	" << xf.xferVoxel.size() << endl;
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            voxelPools_[xf.xferVoxel[j]].xferIn(
                xf.xferPoolIdx, xf.values, xf.lastValues, j );
        }
    }
    // Third, record the current value of pools as the reference for the
    // next cycle.
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        XferInfo& xf = xfer_[i];
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            voxelPools_[xf.xferVoxel[j]].xferOut( j, xf.lastValues, xf.xferPoolIdx );
        }
    }

    // Fourth, do the numerical integration for all reactions.
    //MICKY: This can be parallelized.

    //int tid  = omp_get_thread_num( );

    size_t nvPools = voxelPools_.size( );

    // Compute the grain size.
    //size_t grainSize = 1 + (nvPools / num_threads_);

#if 1
    //omp part
    size_t thread_no = omp_get_thread_num();

    //std::cout << "Threads are " << thread_no  << std::endl;

    for(size_t k=0; k<4; ++k)
    {

        if(k == thread_no){
            for ( size_t i = thread_no; i < nvPools-4; i = i + 4 )
            {
                //cout << "Total threads " << omp_get_num_threads( )  << endl;
                cout << "Voxel pool id " <<  i << " out of " << nvPools 
                    << " thread id " << thread_no << endl;
                voxelPools_[i].advance( p );
            }
            break;
        }

    }
#else
    for (size_t i = 0; i < num_threads_; i++) 
    {
        par_advance( i * num_threads_, (i+1) * num_threads_, p );
    }


#endif

    // Finally, assemble and send the integrated values off for the Dsolve.
    if ( dsolvePtr_ )
    {
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
    if ( !stoichPtr_ )
        return;

    //cout << "Total voxel pools " << voxelPools_.size( ) << endl;

    if ( isBuilt_ )
    {
        for ( unsigned int i = 0 ; i < voxelPools_.size(); ++i )
            voxelPools_[i].reinit( p->dt );
    }
    else
    {
        cout << "Warning:Ksolve::reinit: Reaction system not initialized\n";
        return;
    }

    // cout << "************************* path = " << e.id().path() << endl;
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        const XferInfo& xf = xfer_[i];
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            voxelPools_[xf.xferVoxel[j]].xferInOnlyProxies(
                xf.xferPoolIdx, xf.values,
                stoichPtr_->getNumProxyPools(),
                j );
        }
    }
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        XferInfo& xf = xfer_[i];
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            voxelPools_[xf.xferVoxel[j]].xferOut(
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
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        XferInfo& xf = xfer_[i];
        unsigned int size = xf.xferPoolIdx.size() * xf.xferVoxel.size();
        // values[i].resize( size, 0.0 );
        vector< double > values( size, 0.0 );
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            unsigned int vox = xf.xferVoxel[j];
            voxelPools_[vox].xferOut( j, values, xf.xferPoolIdx );
        }
        xComptOut()->sendTo( e, xf.ksolve, e.id(), values );
    }
    // xComptOut()->sendVec( e, values );
}

void Ksolve::initReinit( const Eref& e, ProcPtr p )
{
    for ( unsigned int i = 0 ; i < voxelPools_.size(); ++i )
    {
        voxelPools_[i].reinit( p->dt );
    }
    // vector< vector< double > > values( xfer_.size() );
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        XferInfo& xf = xfer_[i];
        unsigned int size = xf.xferPoolIdx.size() * xf.xferVoxel.size();
//		xf.values.assign( size, 0.0 );
        xf.lastValues.assign( size, 0.0 );
        for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j )
        {
            unsigned int vox = xf.xferVoxel[j];
            voxelPools_[ vox ].xferOut( j, xf.lastValues, xf.xferPoolIdx );
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
    if ( index == ~0U )
    {
        // unsigned int numCrossRates = stoichPtr_->getNumRates() - stoichPtr_->getNumCoreRates();
        for ( unsigned int i = 0 ; i < voxelPools_.size(); ++i )
        {
            // voxelPools_[i].resetXreacScale( numCrossRates );
            voxelPools_[i].updateAllRateTerms( stoichPtr_->getRateTerms(),
                                          stoichPtr_->getNumCoreRates() );
        }
    }
    else if ( index < stoichPtr_->getNumRates() )
    {
        for ( unsigned int i = 0 ; i < voxelPools_.size(); ++i )
            voxelPools_[i].updateRateTerms( stoichPtr_->getRateTerms(),
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
    if ( ret < startVoxel_  || ret >= startVoxel_ + voxelPools_.size() )
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
        voxelPools_[vox].setN( getPoolIndex( e ), v );
}

double Ksolve::getN( const Eref& e ) const
{
    unsigned int vox = getVoxelIndex( e );
    if ( vox != OFFNODE )
        return voxelPools_[vox].getN( getPoolIndex( e ) );
    return 0.0;
}

void Ksolve::setNinit( const Eref& e, double v )
{
    unsigned int vox = getVoxelIndex( e );
    if ( vox != OFFNODE )
        voxelPools_[vox].setNinit( getPoolIndex( e ), v );
}

double Ksolve::getNinit( const Eref& e ) const
{
    unsigned int vox = getVoxelIndex( e );
    if ( vox != OFFNODE )
        return voxelPools_[vox].getNinit( getPoolIndex( e ) );
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
    unsigned int numVoxels = voxelPools_.size();
    for ( unsigned int i = 0 ; i < numVoxels; ++i )
    {
        voxelPools_[i].resizeArrays( numPoolSpecies );
    }
}

unsigned int Ksolve::getNumPools() const
{
    if ( voxelPools_.size() > 0 )
        return voxelPools_[0].size();
    return 0;
}

VoxelPoolsBase* Ksolve::pools( unsigned int i )
{
    if ( voxelPools_.size() > i )
        return &voxelPools_[i];
    return 0;
}

void Ksolve::par_advance(int begin, int end, ProcPtr p) 
{
    std::atomic<int> idx( begin );
    int num_cpus = std::thread::hardware_concurrency();
    for (int cpu = 0; cpu != num_cpus; ++cpu) 
    {
        std::async( std::launch::async
                , [this, &idx, end, p]() { 
                    for (;;) 
                    {
                        int i = idx++;
                        if (i >= end) 
                            break;
                        voxelPools_[i].advance( p );
                    }
                }
            );
    }
}

double Ksolve::volume( unsigned int i ) const
{
    if ( voxelPools_.size() > i )
        return voxelPools_[i].getVolume();
    return 0.0;
}

void Ksolve::getBlock( vector< double >& values ) const
{
    unsigned int startVoxel = values[0];
    unsigned int numVoxels = values[1];
    unsigned int startPool = values[2];
    unsigned int numPools = values[3];

    assert( startVoxel >= startVoxel_ );
    assert( numVoxels <= voxelPools_.size() );
    assert( voxelPools_.size() > 0 );
    assert( numPools + startPool <= voxelPools_[0].size() );
    values.resize( 4 + numVoxels * numPools );

    for ( unsigned int i = 0; i < numVoxels; ++i )
    {
        const double* v = voxelPools_[ startVoxel + i ].S();
        for ( unsigned int j = 0; j < numPools; ++j )
        {
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
    assert( numVoxels <= voxelPools_.size() );
    assert( voxelPools_.size() > 0 );
    assert( numPools + startPool <= voxelPools_[0].size() );
    assert( values.size() == 4 + numVoxels * numPools );

    for ( unsigned int i = 0; i < numVoxels; ++i )
    {
        double* v = voxelPools_[ startVoxel + i ].varS();
        for ( unsigned int j = 0; j < numPools; ++j )
        {
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
    if ( vols.size() == voxelPools_.size() )
    {
        for ( unsigned int i = 0; i < vols.size(); ++i )
        {
            voxelPools_[i].setVolumeAndDependencies( vols[i] );
        }
        stoichPtr_->setupCrossSolverReacVols();
        updateRateTerms( ~0U );
    }
}

//////////////////////////////////////////////////////////////////////////
// cross-compartment reaction stuff.
//////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// Functions for setup of cross-compartment transfer.
/////////////////////////////////////////////////////////////////////

void Ksolve::print() const
{
    cout << "path = " << stoichPtr_->getKsolve().path() <<
         ", numPools = " << voxelPools_.size() << "\n";
    for ( unsigned int i = 0; i < voxelPools_.size(); ++i )
    {
        cout << "pools[" << i << "] contents = ";
        voxelPools_[i].print();
    }
    cout << "method = " << method_ << ", stoich=" << stoich_.path() <<endl;
    cout << "dsolve = " << dsolve_.path() << endl;
    cout << "compartment = " << compartment_.path() << endl;
    cout << "xfer summary: numxfer = " << xfer_.size() << "\n";
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
        cout << "xfer_[" << i << "] numValues=" <<
             xfer_[i].values.size() <<
             ", xferPoolIdx.size = " << xfer_[i].xferPoolIdx.size() <<
             ", xferVoxel.size = " << xfer_[i].xferVoxel.size() << endl;
    }
    cout << "xfer details:\n";
    for ( unsigned int i = 0; i < xfer_.size(); ++i )
    {
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


