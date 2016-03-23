/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <omp.h>
#include <sys/time.h>
#include "header.h"
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#endif


#include <unistd.h>

#include "OdeSystem.h"
#include "VoxelPoolsBase.h"
#include "VoxelPools.h"
#include "../mesh/VoxelJunction.h"
#include "XferInfo.h"
#include "ZombiePoolInterface.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "../shell/Shell.h"

#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"
#include "Ksolve.h"

const unsigned int OFFNODE = ~0;

time_t time_taken = 0;
static int zeros = 0;

int advanceProcess(VoxelPools* pool, int blockSize, ProcPtr p)
{
	   int GSLSUCCESS = 0;
	   size_t i;
	
   for(int j =0; j < blockSize; j++)
   {
	   
	   gsl_odeiv2_driver* localDriver_= pool[j].getVoxeldriver();
	   const gsl_odeiv2_system* sys = localDriver_->sys;

	   rkf45_state_t *state = (rkf45_state_t *) localDriver_->s->state;
	   size_t dim = localDriver_->s->dimension;
	   double t = p->currTime - p->dt;
	   double h = localDriver_->h;
	   double* y = pool[j].varS();
	   double *yerr = localDriver_->e->yerr;
	   const double* dydt_in = localDriver_->e->dydt_in;
	   double* dydt_out = localDriver_->e->dydt_out;

	   double *const k1 = state->k1;
	   double *const k2 = state->k2;
	   double *const k3 = state->k3;
	   double *const k4 = state->k4;
	   double *const k5 = state->k5;
	   double *const k6 = state->k6;
	   double *const ytmp = state->ytmp;
	   double *const y0 = state->y0;
    
	   memcpy (y0, y, dim); // memcpy in case of failure...

    /*K1 step */
    {
	    if (dydt_in == NULL)
	    {
			  int s = RKF45_ODEIV_FN_EVAL (sys, t, y, k1);
			  if (s != GSLSUCCESS)
					return s;
	    }
	    else
			  memcpy (k1, dydt_in, dim);
    }
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] + ah[0] * h * k1[i];

    /*k2 step */
    {
	    int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[0] * h, ytmp, k2);
	    if (s != GSLSUCCESS)
			  return s;
    }
	    for (i = 0; i < dim; i++)
			  ytmp[i] = y[i] + h * (b3[0] * k1[i] + b3[1] * k2[i]);

    /*k3 step */
    {
	     int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[1] * h, ytmp, k3);
		if (s != GSLSUCCESS)
			   return s;
    }

		for (i = 0; i < dim; i++)
			   ytmp[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i]);

	 /*k4 step*/ 
    {
		 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[2] * h, ytmp, k4);
		 if (s != GSLSUCCESS)
			    return s;
    }

		 for (i = 0; i < dim; i++)
			    ytmp[i] =  y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i]);

	 /*k5 step */
    {
		 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[3] * h, ytmp, k5);
		 if (s != GSLSUCCESS)
			    return s;
    }

		 for (i = 0; i < dim; i++)
			    ytmp[i] = y[i] + h * (b6[0] * k1[i] + b6[1] * k2[i] + b6[2] * k3[i] + b6[3] * k4[i] + b6[4] * k5[i]);

	 /*k6 and final sum */
    {
		 int s = RKF45_ODEIV_FN_EVAL (sys, t + ah[4] * h, ytmp, k6);
		 if (s != GSLSUCCESS)
			    return s;
    }

		 for (i = 0; i < dim; i++)
		 {
			    const double d_i = c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i];
			    y[i] += h * d_i;
		 }

		 /* Derivatives at output */
				   
		 if (dydt_out != NULL)
		 {
			    int s = RKF45_ODEIV_FN_EVAL (sys, t + h, y, dydt_out);
			    if (s != GSLSUCCESS)
			    {
					  /* Restore initial values */
					  memcpy (y, y0, dim);
					  return s;
			    }
		 }
		 /* difference between 4th and 5th order */
		   for (i = 0; i < dim; i++)
				 yerr[i] = h * (ec[1] * k1[i] + ec[3] * k3[i] + ec[4] * k4[i] + ec[5] * k5[i] + ec[6] * k6[i]);

   }


   return GSLSUCCESS;
}



#if _KSOLVE_PTHREADS

extern "C" void* call_func( void* f )
{
	   std::auto_ptr< pthreadWrap > w( static_cast< pthreadWrap* >( f ) );
	   int localId = w->tid;
	   bool* destroySignal = w->destroySig;


	   while(!*destroySignal) 
	   {
			 sem_wait(w->sMain); // Wait for a signal from main thread

	   //Assigning the addresses as set by the process function
			 ProcPtr p = *(w->p);
			 VoxelPools *lpoolarray_ = *w->poolsArr_;
			 int blz = *(w->pthreadBlock);
			 int startIndex = localId * blz;
			 int endIndex = startIndex + blz;

		//Perform the advance function 
		advanceProcess(&lpoolarray_[startIndex], blz, p);

		   sem_post(w->sThread); // Send the signal to the main thread. 
	   }

	   pthread_exit(NULL);
	   return NULL;

}
#endif // _KSOLVE_PTHREADS



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
		dsolve_(),
		dsolvePtr_( 0 )
{
#if _KSOLVE_PTHREADS
	   pthread_attr_t attr;
	   pthread_attr_init(&attr);
	   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	   destroySignal = new bool;
	   *destroySignal = false;

	   pthreadP = new ProcPtr;
	   poolArray_ = new VoxelPools*;
	   pthreadBlock = new int;
	   
	   for(long i = 0; i < NTHREADS; i++)
	   {
			sem_init(&threadSemaphor[i],0,0);
			sem_init(&mainSemaphor[i],0,0);

			 pthreadWrap* w = new pthreadWrap(i, &threadSemaphor[i], &mainSemaphor[i], destroySignal, pthreadP, poolArray_, pthreadBlock);

			 int rc = pthread_create(&threads[i], &attr, call_func, (void*) w);
			 if(rc)
				    printf("thread creation problem\n");
	   }
#endif
;
}

Ksolve::~Ksolve()
{
#if _KSOLVE_PTHREADS
	   *destroySignal = true;

	   for(int i = 0; i < NTHREADS; i++)
			 sem_post(&mainSemaphor[i]);

	   for(int i = 0; i < NTHREADS; i++)
			 pthread_join(threads[i], NULL);
//			 sem_destroy(&threadSemaphor[i]);
//			 sem_destroy(&mainSemaphor[i]);
#endif
	   ;
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
		ode.initStepSize = 0.01; // This will be overridden at reinit.

#ifdef USE_GSL
		ode.gslSys.dimension = stoichPtr_->getNumAllPools();
		if ( ode.gslSys.dimension == 0 )
			return; // No pools, so don't bother.
		innerSetMethod( ode, method_ );
		ode.gslSys.function = &VoxelPools::gslFunc;
   		ode.gslSys.jacobian = 0;
		innerSetMethod( ode, method_ );
		unsigned int numVoxels = pools_.size();
		for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
   			ode.gslSys.params = &pools_[i];
			pools_[i].setStoich( stoichPtr_, &ode );
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
	vector< double > s( stoichPtr_->getNumAllPools(), 1.0 );
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


        static unsigned int usedThreads = 0;


	if ( isBuilt_ == false )
		return;
	// First, handle incoming diffusion values, update S with those.
	if ( dsolvePtr_ ) {
		vector< double > dvalues( 4 );
		dvalues[0] = 0;
		dvalues[1] = getNumLocalVoxels();
		dvalues[2] = 0;
		dvalues[3] = stoichPtr_->getNumVarPools();
		dsolvePtr_->getBlock( dvalues );
		
		setBlock( dvalues );
	}

//	 omp_set_num_threads(4);
	// Second, take the arrived xCompt reac values and update S with them.

	for ( unsigned int i = 0; i < xfer_.size(); ++i ) 
	{
		const XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) 
		{
			pools_[xf.xferVoxel[j]].xferIn(xf.xferPoolIdx, xf.values, xf.lastValues, j );
		}
	}

	// Third, record the current value of pools as the reference for the
	// next cycle.
	for ( unsigned int i = 0; i < xfer_.size(); ++i ) 
	{
		XferInfo& xf = xfer_[i];
		for ( unsigned int j = 0; j < xf.xferVoxel.size(); ++j ) 
		{
			pools_[xf.xferVoxel[j]].xferOut( j, xf.lastValues, xf.xferPoolIdx );
		}
	}

#if _KSOLVE_SEQ
	static int seqVersion = 0;
	 VoxelPools* poolArray_ = &pools_[0]; //Call the vector as an array
	 int poolSize = pools_.size(); //Find out the size of the vector

	 if(!seqVersion)
	 {
		    seqVersion = 1;
		    cout << endl << "Sequential Version " << endl;
	 }

	for(int j = 0; j < poolSize ; j++)
		   advanceProcess(&poolArray_[j], 1, p);
//		   poolArray_[j].advance(p);


#endif     

/***************************************************************KSOLVE OPENMP Parallel-for Parallelization ***************************************************************************/
#if _KSOLVE_OPENMP_FOR
		   
	 int numThreads = 8; //Define the number of threads
	 int poolSize = pools_.size(); //Find out the size of the vector
	 static int cellsPerThread = 0; // Used for printing...
	 VoxelPools* poolArray_ = &pools_[0]; //Call the vector as an array
	 int j = 0;
//	 int blockSize = poolSize/numThreads;

	 if(!cellsPerThread)
	 {
		    cellsPerThread = 2;
		    cout << endl << "OpenMP parallelism: Using parallel-for " << endl;
		    cout << "NUMBER OF CELLS PER THREAD = " << cellsPerThread << " And THREADS USED = " << numThreads << endl;
	 }

//	   struct timeval stop, start;
//	   time_t time_taken = 0;
//	   gettimeofday(&start, NULL);

#pragma omp parallel for schedule(guided, cellsPerThread) num_threads(numThreads) shared(poolArray_,p, poolSize)
	for(int j = 0; j < poolSize ; j++)
		   advanceProcess(&poolArray_[j], 1, p);


	//   gettimeofday(&stop, NULL);
	//   time_taken = stop.tv_usec - start.tv_usec;

	//cout << "PARALLEL Time Taken from rungekutta = " << time_taken << " With PoolSize = " << poolSize << "  varS() = " << poolSize*(poolArray_[0].getVoxeldriver()->s->dimension) << endl;


#endif //_KSOLVE_OPENMP_FOR
/*************************************************************************************************************************************************************************************/

/***************************************************************KSOLVE OPENMP Task-based Parallelization ***************************************************************************/
#if _KSOLVE_OPENMP_TASK
	 static int usedThreads = 0;
	 int numThreads = 2; //Each block will be executed by one thread

	 int poolSize = pools_.size();
	 int blockSize = poolSize/numThreads; //Number of cells in each block
	 int remainder = poolSize % blockSize;

#pragma omp parallel num_threads(numThreads)
#pragma omp single 
        {
	        if( usedThreads == 0)
	        {
	        	usedThreads = numThreads;
	        	cout << "Info: OpenMP tasking: Threads used: " << usedThreads << endl;
	        }

		   int iterator = 0, j = 0;
		   vector<VoxelPools>::iterator i = pools_.begin();

            while(iterator < numThreads)
            {
                vector<VoxelPools>::iterator threadIterator = i;
#pragma omp task private(j) firstprivate(threadIterator, blockSize,p)
                {
                    for(j = 0; j < blockSize ; threadIterator++,j++)
                        threadIterator->advance( p );
                }

                for(j = 0; j < blockSize ;j++) i++;
                iterator++;
            }

#pragma omp parallel for schedule (guided, 1) firstprivate(p) 
		  for ( vector< VoxelPools >::iterator k = i; k < pools_.end(); ++k ) 
				k->advance( p );

#pragma omp taskwait
        }

#endif //_KSOLVE_OPENMP_TASK	   
/*************************************************************************************************************************************************************************************/

#if _KSOLVE_PTHREADS

	 static int usedThreads = 0;

	 clock_t startPthreadTimer = clock();

	 if(!usedThreads)
	 {
		    usedThreads = NTHREADS;
		    cout << endl << "Pthread Parallelism " << endl;
		    cout << "NUMBER OF THREADS USED = " << usedThreads << endl;
	 }

	*poolArray_ = &pools_[0];
	int poolSize = pools_.size(); //Find out the size of the vector
	*pthreadBlock = poolSize/NTHREADS;
	*pthreadP = p;

	for(int i = 0; i < NTHREADS; i++)
		   sem_post(&mainSemaphor[i]); //Send signal to the threads to start

	advanceProcess(&pools_[NTHREADS*(*pthreadBlock)], poolSize-NTHREADS*(*pthreadBlock), p);

	for(int i = 0; i < NTHREADS; i++)
		   sem_wait(&threadSemaphor[i]); // Wait for threads to finish their work

#endif //_KSOLVE_PTHREADS

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
    // cout << "************************* path = " << e.id().path() << endl;
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
        //		xf.values.assign( size, 0.0 );
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

VoxelPoolsBase* Ksolve::pools( unsigned int i )
{
    if ( pools_.size() > i )
        return &pools_[i];
    return 0;
}

double Ksolve::volume( unsigned int i ) const
{
    if ( pools_.size() > i )
        return pools_[i].getVolume();
    return 0.0;
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

/////////////////////////////////////////////////////////////////////
// Functions for setup of cross-compartment transfer.
/////////////////////////////////////////////////////////////////////

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

