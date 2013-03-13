/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "VoxelPools.h"
/*
#include "../shell/Shell.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"
*/
#include "OdeSystem.h"
#include "GslStoich.h"

const Cinfo* GslStoich::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< GslStoich, unsigned int > numVarPools(
			"numVarPools",
			"Number of variable molecule pools in the core reac system",
			&GslStoich::getNumVarPools
		);
		static ReadOnlyValueFinfo< GslStoich, unsigned int > numAllPools(
			"numAllPools",
			"Number of variable molecule pools in the core reac system",
			&GslStoich::getNumAllPools
		);

		static ReadOnlyValueFinfo< GslStoich, unsigned int > numLocalVoxels(
			"numLocalVoxels",
			"Number of voxels in the core reac-diff system, on the current "
			"solver. ",
			&GslStoich::getNumLocalVoxels
		);
		static ReadOnlyValueFinfo< GslStoich, unsigned int > numAllVoxels(
			"numAllVoxels",
			"Number of voxels in the entire reac-diff system, "
			"including proxy voxels to represent abutting compartments.",
			&GslStoich::getNumLocalVoxels
		);

		static ElementValueFinfo< GslStoich, string > path(
			"path",
			"Path of reaction system to take over",
			&GslStoich::setPath,
			&GslStoich::getPath
		);

		static ReadOnlyValueFinfo< GslStoich, double > estimatedDt(
			"estimatedDt",
			"Estimate of fastest (smallest) timescale in system."
			"This is fallible because it depends on instantaneous concs,"
			"which of course change over the course of the simulation.",
			&GslStoich::getEstimatedDt
		);

		static ReadOnlyValueFinfo< GslStoich, bool > isInitialized( 
			"isInitialized", 
			"True if the Stoich message has come in to set parms",
			&GslStoich::getIsInitialized
		);
		static ValueFinfo< GslStoich, string > method( "method", 
			"Numerical method to use.",
			&GslStoich::setMethod,
			&GslStoich::getMethod 
		);
		static ValueFinfo< GslStoich, double > relativeAccuracy( 
			"relativeAccuracy", 
			"Accuracy criterion",
			&GslStoich::setRelativeAccuracy,
			&GslStoich::getRelativeAccuracy
		);
		static ValueFinfo< GslStoich, double > absoluteAccuracy( 
			"absoluteAccuracy", 
			"Another accuracy criterion",
			&GslStoich::setAbsoluteAccuracy,
			&GslStoich::getAbsoluteAccuracy
		);
		static ValueFinfo< GslStoich, double > internalDt( 
			"internalDt", 
			"internal timestep to use.",
			&GslStoich::setInternalDt,
			&GslStoich::getInternalDt
		);

		static ValueFinfo< GslStoich, Id > compartment( 
			"compartment",
			"This is the Id of the compartment, which must be derived from"
			"the ChemCompt baseclass. The GslStoich needs"
			"the ChemCompt Id only for diffusion, "
			" and one can pass in Id() instead if there is no diffusion,"
			" or just leave it unset.",
			&GslStoich::setCompartment,
			&GslStoich::getCompartment
		);
		static ReadOnlyValueFinfo< GslStoich, vector< Id > > coupledCompartments( 
			"coupledCompartments",
			"This is the Id of all compartment coupled to the one in which"
			"the GslStoich resides. This is found by checking for reactions"
			"which span compartment boundaries.",
			&GslStoich::getCoupledCompartments
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< GslStoich >( &GslStoich::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< GslStoich >( &GslStoich::reinit ) );

		static DestFinfo initProc( "initProc",
			"Handles init call",
			new ProcOpFunc< GslStoich >( &GslStoich::init ) );
		static DestFinfo initReinit( "initReinit",
			"Handles initReinit call",
			new ProcOpFunc< GslStoich >( &GslStoich::initReinit ) );

		static DestFinfo setElist( "elist", 
			"Assign the list of Elements that this solver handles. "
			"These are normally all children of the compartment on which "
			"the current GslStoich object resides.",
			new EpFunc1< GslStoich, vector< Id > >( &GslStoich::setElist ));


		static DestFinfo remesh( "remesh",
			"Handle commands to remesh the pool. This may involve changing "
			"the number of pool entries, as well as changing their volumes",
			new EpFunc5< GslStoich, 
			double,
			unsigned int, unsigned int, 
			vector< unsigned int >, vector< double > >( 
					&GslStoich::remesh )
		);
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		
		static Finfo* initShared[] = {
			&initProc, &initReinit
		};
		static SharedFinfo init( "init",
			"Shared message for init and initReinit",
			initShared, sizeof( initShared ) / sizeof( const Finfo* )
		);

	static Finfo* gslStoichFinfos[] =
	{
		&numVarPools,		// ReadOnlyValue
		&numAllPools,		// ReadOnlyValue
		&numLocalVoxels,	// ReadOnlyValue
		&numAllVoxels,		// ReadOnlyValue
		&path,				// Value
		&estimatedDt,		// ReadOnlyValue
		&isInitialized,		// Value
		&method,			// Value
		&relativeAccuracy,	// Value
		&absoluteAccuracy,	// Value
		&compartment,		// Value
		&coupledCompartments,	// ReadOnlyValue
		&setElist,			// DestFinfo
		&remesh,			// DestFinfo
		&proc,				// SharedFinfo
		&init,				// SharedFinfo
	};
	
	static  Cinfo gslStoichCinfo(
		"GslStoich",
		SolverBase::initCinfo(),
		gslStoichFinfos,
		sizeof(gslStoichFinfos)/sizeof(Finfo *),
		new Dinfo< GslStoich >
	);

	return &gslStoichCinfo;
}

static const Cinfo* gslStoichCinfo = GslStoich::initCinfo();

///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

GslStoich::GslStoich()
	: 
	isInitialized_( false ),
	junctionsNotReady_( false ),
	method_( "rk5" ),
	path_( "" ),
	absAccuracy_( 1.0e-9 ),
	relAccuracy_( 1.0e-6 ),
	internalStepSize_( 0.1 ),
	y_( 0 ),
	coreStoich_( true ), // Set isMaster flag, so it handles dellocation.
	ode_(0), // 
	pools_(0), // Don't need to set up compts or pools, setPath does it..
	compartmentId_( 0 ),
	diffusionMesh_( 0 )
{
		;
}

/**
 * Needed for the Dinfo::assign function, to ensure we initialize the 
 * gsl pointers correctly. Instead of trying to guess the Element indices,
 * we zero out the pointers (not free them) so that the system has to
 * do the initialization in a separate call to GslStoich::setPath().
 */
GslStoich& GslStoich::operator=( const GslStoich& other )
{
	isInitialized_ = 0;
	method_ = other.method_;
	path_ = other.path_;
	absAccuracy_ = other.absAccuracy_;
	relAccuracy_ = other.relAccuracy_;
	internalStepSize_ = other.internalStepSize_;
	y_.clear();
	ode_.clear();
	pools_.clear();
	return *this;
}

GslStoich::~GslStoich()
{
	// Need to preempt this, do it before all the pools are deleted.
	coreStoich_.unZombifyModel(); 
	for( unsigned int i = 0; i < ode_.size(); ++i )
		ode_[i].reallyFreeOdeSystem();
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
unsigned int GslStoich::getNumVarPools() const
{
	if ( coreStoich() )
		return coreStoich()->getNumVarPools();
	else
		return 0;
}

unsigned int GslStoich::getNumAllPools() const
{
	if ( coreStoich() )
		return coreStoich()->getNumAllPools();
	else
		return 0;
}

unsigned int GslStoich::getNumLocalVoxels() const
{
	return localMeshEntries_.size();
}

unsigned int GslStoich::getNumAllVoxels() const
{
	return pools_.size();
}

string GslStoich::getPath( const Eref& e, const Qinfo* q ) const
{
	return path_;
}

double GslStoich::getEstimatedDt() const
{
	if ( coreStoich() )
		return coreStoich()->getEstimatedDt();
	else
		return 0;
}

// In GslStoichSetup.cpp
// void GslStoich::setPath( string path )

bool GslStoich::getIsInitialized() const
{
	return isInitialized_;
}

string GslStoich::getMethod() const
{
	return method_;
}

void GslStoich::setMethod( string method )
{
	for ( vector < OdeSystem >::iterator 
					i = ode_.begin(); i != ode_.end(); ++i )
			method_ = i->setMethod( method );
}

double GslStoich::getRelativeAccuracy() const
{
	return relAccuracy_;
}
void GslStoich::setRelativeAccuracy( double value )
{
	relAccuracy_ = value;
}

double GslStoich::getAbsoluteAccuracy() const
{
	return absAccuracy_;
}
void GslStoich::setAbsoluteAccuracy( double value )
{
	absAccuracy_ = value;
}

double GslStoich::getInternalDt() const
{
	return internalStepSize_;
}
void GslStoich::setInternalDt( double value )
{
	internalStepSize_ = value;
}

Id GslStoich::getCompartment() const
{
	return compartmentId_;
}
void GslStoich::setCompartment( Id value )
{
	if ( value == Id() || !value.element()->cinfo()->isA( "ChemCompt" ) )
   	{
		cout << "Warning: GslStoich::setCompartment: "
				"Value must be a ChemCompt subclass\n";
		compartmentId_ = Id();
		diffusionMesh_ = 0;
	} else {
		compartmentId_ = value;
		diffusionMesh_ = reinterpret_cast< ChemCompt* >(
				compartmentId_.eref().data() );
	}
}

// Checks coreStoich for coupled compartments
vector< Id > GslStoich::getCoupledCompartments() const
{
	return coreStoich_.getOffSolverCompts();
}

const double* GslStoich::S( unsigned int meshIndex ) const
{
	assert( pools_.size() > meshIndex );
	return pools_[meshIndex].S();
}

/////////////////////////////////////////////////////////////////////////
// Informaiton functions
/////////////////////////////////////////////////////////////////////////
const vector< VoxelPools >& GslStoich::pools()
{
	return pools_;
}

const vector< OdeSystem >& GslStoich::ode()
{
	return ode_;
}
