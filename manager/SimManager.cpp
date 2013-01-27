/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "ElementValueFinfo.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
#include "SimManager.h"

/*
static SrcFinfo1< Id >* plugin()
{
	static SrcFinfo1< Id > ret(
		"plugin", 
		"Sends out Stoich Id so that plugins can directly access fields and functions"
	);
	return &ret;
}
*/
static SrcFinfo0* requestMeshStats()
{
	static SrcFinfo0 requestMeshStats(
		"requestMeshStats", 
		"Asks for basic stats for mesh:"
		"Total # of entries, and a vector of unique volumes of voxels"
	);
	return &requestMeshStats;
}

static SrcFinfo2< unsigned int, unsigned int >* nodeInfo()
{
	static SrcFinfo2< unsigned int, unsigned int > nodeInfo(
		"nodeInfo", 
		"Sends out # of nodes to use for meshing, and # of threads to "
		"use on each node, to the ChemMesh. These numbers sometimes"
		"differ from the total # of nodes and threads, because the "
		"SimManager may have other portions of the model to allocate."
	);
	return &nodeInfo;
}

const Cinfo* SimManager::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< SimManager, double > syncTime(
			"syncTime",
			"SyncTime is the interval between synchronizing solvers"
			"5 msec is a typical value",
			&SimManager::setSyncTime,
			&SimManager::getSyncTime
		);

		static ValueFinfo< SimManager, bool > autoPlot(
			"autoPlot",
			"When the autoPlot flag is true, the simManager guesses which"
			"plots are of interest, and builds them.",
			&SimManager::setAutoPlot,
			&SimManager::getAutoPlot
		);

		static ValueFinfo< SimManager, double > plotDt(
			"plotDt",
			"plotDt is the timestep for plotting variables. As most will be"
			"chemical, a default of 1 sec is reasonable",
			&SimManager::setPlotDt,
			&SimManager::getPlotDt
		);

		static ValueFinfo< SimManager, double > runTime(
			"runTime",
			"runTime is the requested duration of the simulation that is "
			"stored in some kinds of model definition files.",
			&SimManager::setRunTime,
			&SimManager::getRunTime
		);

		static ElementValueFinfo< SimManager, string > method(
			"method",
			"method is the numerical method used for the calculations."
			"This will set up or even replace the solver with one able"
			"to use the specified method. "
			"Currently works only with two solvers: GSL and GSSA."
			"The GSL solver has a variety of ODE methods, by default"
			"Runge-Kutta-Fehlberg."
			"The GSSA solver currently uses the Gillespie Stochastic"
			"Systems Algorithm, somewhat optimized over the original"
			"method.",
			&SimManager::setMethod,
			&SimManager::getMethod
		);

		static ValueFinfo< SimManager, unsigned int > version(
			"version",
			"Numerical version number. Used by kkit",
			&SimManager::setVersion,
			&SimManager::getVersion
		);

		static ReadOnlyElementValueFinfo< SimManager, string > modelFamily(
			"modelFamily",
			"Family classification of model: *kinetic, and *neuron "
			"are the options so far. In due course expect to see things"
			"like detailedNetwork, intFireNetwork, sigNeur and so on.",
			&SimManager::getModelFamily
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo build( "build",
			"Sets up model, with the specified method. The method may be"
			"empty if the intention is that methods be set up through "
			"hints in the ChemMesh compartments.",
			new EpFunc1< SimManager, string >( &SimManager::build ) );

		static DestFinfo makeStandardElements( "makeStandardElements",
			"Sets up the usual infrastructure for a model, with the"
			"ChemMesh, Stoich, solver and suitable messaging."
			"The argument is the MeshClass to use.",
			new EpFunc1< SimManager, string >( &SimManager::makeStandardElements ) );

		static DestFinfo meshSplit( "meshSplit",
			"Handles message from ChemMesh that defines how"
			"meshEntries communicate between nodes."
			"First arg is oldvol, next is list of other nodes, third arg is list number of"
			"meshEntries to be transferred for each of these nodes, "
			"fourth arg is catenated list of meshEntries indices on"
			"my node going to each of the other connected nodes, and"
			"last arg is matching list of meshEntries on other nodes",
			new EpFunc5< SimManager, double, vector< unsigned int >, 
			vector< unsigned int>, vector< unsigned int >, 
			vector< unsigned int > >( &SimManager::meshSplit )
		);

		static DestFinfo meshStats( "meshStats",
			 "Basic statistics for mesh: Total # of entries, and a vector"
			 "of unique volumes of voxels",
			 new EpFunc2< SimManager, unsigned int, vector< double > >( 
			 	&SimManager::meshStats )
		);

		//////////////////////////////////////////////////////////////
		// Shared Finfos
		//////////////////////////////////////////////////////////////
		static Finfo* nodeMeshingShared[] = {
			&meshSplit, &meshStats, requestMeshStats(), nodeInfo()
		};  
		
		static SharedFinfo nodeMeshing( "nodeMeshing",
			"Connects to ChemMesh to coordinate meshing with parallel"
			"decomposition and with the Stoich",
			nodeMeshingShared, 
			sizeof( nodeMeshingShared ) / sizeof( const Finfo* )
		);

		//////////////////////////////////////////////////////////////

	static Finfo* simManagerFinfos[] = {
		&syncTime,		// Value
		&autoPlot,		// Value
		&plotDt,		// Value
		&runTime,		// Value
		&method,		// Value
		&version,		// Value
		&modelFamily,	// Value
		&build,			// DestFinfo
		&makeStandardElements,			// DestFinfo
		&nodeMeshing,	// SharedFinfo
	};

	static Cinfo simManagerCinfo (
		"SimManager",
		Neutral::initCinfo(),
		simManagerFinfos,
		sizeof( simManagerFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SimManager >()
	);

	return &simManagerCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* simManagerCinfo = SimManager::initCinfo();

SimManager::SimManager()
	: 
		syncTime_( 0.005 ),
		autoPlot_( 1 ),
		plotdt_( 1 ),
		simdt_( 1 ),
		version_( 0 )
{;}

SimManager::~SimManager()
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SimManager::setAutoPlot( bool v )
{
	autoPlot_ = v;
}

bool SimManager::getAutoPlot() const
{
	return autoPlot_;
}

void SimManager::setSyncTime( double v )
{
	syncTime_ = v;
}

double SimManager::getSyncTime() const
{
	return syncTime_;
}

void SimManager::setPlotDt( double v )
{
	plotdt_ = v;
}

double SimManager::getPlotDt() const
{
	return plotdt_;
}

void SimManager::setSimDt( double v )
{
	simdt_ = v;
}

double SimManager::getSimDt() const
{
	return simdt_;
}

void SimManager::setRunTime( double v )
{
	runTime_ = v;
}

double SimManager::getRunTime() const
{
	return runTime_;
}

void SimManager::setVersion( unsigned int v )
{
	version_ = v;
}

unsigned int SimManager::getVersion() const
{
	return version_;
}

void SimManager::setMethod( const Eref& e, const Qinfo* q, string v )
{
	if ( q->addToStructuralQ() )
		return;

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	if ( stoich_ != Id() && stoich_.element() != 0 )
		shell->doDelete( stoich_ );

	if ( v == "GSSA" || v == "gssa" || v == "Gillespie" || v == "gillespie")
	{
		buildGssa( e, q, shell );
	} else if ( v == "rk5" || v == "gsl" || v == "GSL" ) {
		buildGsl( e, q, shell, v );
		// setupRK5();
	} else if ( v == "ee" || v == "EE" || v == "ExpEuler" ) {
		// shell->doDelete( stoich_ );
		;
	} else {
		cout << "SimManager::setMethod(" << v << "): Not yet implemented.";
		cout << "Falling back to EE method\n";
	}
	method_ = v;
}

string SimManager::getMethod( const Eref& e, const Qinfo* q ) const
{
	return method_;
}

string SimManager::getModelFamily( const Eref& e, const Qinfo* q ) const
{
	Id mesh = findChemMesh();
	if ( mesh != Id() )
			return "kinetic";
	return "unknown";
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

Id SimManager::findChemMesh() const
{
	vector< Id > ret;
	string basePath = baseId_.path();

	int num = simpleWildcardFind( basePath + "/##[ISA=ChemMesh]", ret );
	if ( num == 0 )
		return Id();
	return ret[0];
}

double estimateChemLoad( Id mesh, Id stoich )
{

	unsigned int numMeshEntries = 
	 	Field< unsigned int >::get( mesh, "num_mesh" );
	unsigned int numPools = 
	 	Field< unsigned int >::get( stoich, "numVarPools" );
	double dt = Field< double >::get( stoich, "estimatedDt" );
	double load = numMeshEntries * numPools / dt;
	return load;
}

double estimateHsolveLoad( Id hsolver )
{
	// First check if solver exists.
	/*
	double dt = Field< double >::get( hsolver, "estimatedDt" );
	double mathDt = Field< double >::get( hsolver, "numHHChans" );
	*/
	return 0;
}

/**
 * This needs to be called only on the master node, and in shell
 * thread mode.
 */
void SimManager::build( const Eref& e, const Qinfo* q, string method )
{
	// First, check if the tree has a compartment/ChemMesh as the base
	// of the chemical system. If not, put in a single-voxel ChemMesh.
	baseId_ = e.id();
	Id mesh = findChemMesh();

	if ( mesh == Id() ) {
		 cout << "SimManager::build: No chem mesh found, still need to sort this out\n";
		 return;
	}
	buildFromKkitTree( e, q, method );

	// Apply heuristic for threads and nodes
	// Replicate pools as per node decomp. Shell::handleReMesh
	// Make the stoich, set up its path
	// Make the GslIntegrator
	// Set up GslIntegrator in the usual way.
	// ? Apply boundary conditions
	// Set up internode messages to and from stoichs
	// Set up and assign the clocks
	// Create the plots.
}

void SimManager::makeStandardElements( const Eref& e, const Qinfo* q, 
	string meshClass )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id baseId_ = e.id();
	Id kinetics = 
		shell->doCreate( meshClass, baseId_, "kinetics", dims, true );
		SetGet2< double, unsigned int >::set( kinetics, "buildDefaultMesh", 1e-15, 1 );
	assert( kinetics != Id() );
	assert( kinetics.eref().element()->getName() == "kinetics" );

	Id graphs = Neutral::child( baseId_.eref(), "graphs" );
	if ( graphs == Id() ) {
		graphs = 
		shell->doCreate( "Neutral", baseId_, "graphs", dims, true );
	}
	assert( graphs != Id() );

	Id geometry = Neutral::child( baseId_.eref(), "geometry" );
	if ( geometry == Id() ) {

		geometry = 
		shell->doCreate( "Geometry", baseId_, "geometry", dims, true );
		// MsgId ret = shell->doAddMsg( "Single", geometry, "compt", kinetics, "reac" );
		// assert( ret != Msg::bad );
	}
	assert( geometry != Id() );

	Id groups = Neutral::child( baseId_.eref(), "groups" );
	if ( groups == Id() ) {
		groups = 
		shell->doCreate( "Neutral", baseId_, "groups", dims, true );
	}
	assert( groups != Id() );
}

void SimManager::meshSplit( const Eref& e, const Qinfo* q,
	double oldVol,
	vector< unsigned int > nodeList, 
	vector< unsigned int > numEntriesPerNode, 
	vector< unsigned int > outgoingEntries, 
	vector< unsigned int > incomingEntries
	)
{
	cout << "in SimManager::meshSplit"	;
	// buildFromKkitTree( "gsl" );
}

void SimManager::meshStats( const Eref& e, const Qinfo* q,
	unsigned int numMeshEntries, vector< double > voxelVols )
{
	cout << "in SimManager::meshStats"	;
}

//////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////

 // Don't make any solvers.
void SimManager::buildEE( Shell* shell )
{
	string basePath = baseId_.path();
	shell->doUseClock( basePath + "/kinetics/##[TYPE=Pool]", "process", 4);
		// Normally we would simply say [ISA!=Pool] here. But that puts
		// a Process operation on the mesh, which should not be done in
		// this mode as diffusion isn't supported.
	shell->doUseClock( basePath + "/kinetics/##[ISA!=Pool]", "process", 5);
}

void SimManager::buildGssa( const Eref& e, const Qinfo* q, Shell* shell )
{
	vector< int > dims( 1, 1 );
	 // This is a placeholder for more sophisticated node-balancing info
	 // May also need to put in volscales here.
	stoich_ = shell->doCreate( "GssaStoich", baseId_, "stoich", dims );

	string basePath = baseId_.path();
	Id compt( basePath + "/kinetics" );
	assert( compt != Id() );

	Field< string >::set( stoich_, "path", basePath + "/kinetics/##");

	MsgId mid = shell->doAddMsg( "Single", compt, "meshSplit", 
		stoich_, "meshSplit" );
	assert( mid != Msg::bad );

	double chemLoad = estimateChemLoad( compt, stoich_ );
	// Here we would also estimate cell load
	Id hsolver;
	double hsolveLoad = estimateHsolveLoad( hsolver );

	numChemNodes_ = Shell::numNodes() * chemLoad / ( chemLoad + hsolveLoad);
	
	nodeInfo()->send( e, q->threadNum(), numChemNodes_,
		Shell::numProcessThreads() ); 
	Qinfo::waitProcCycles( 2 );

	string path0 = basePath + "/kinetics/mesh," + 
		basePath + "/kinetics/##[ISA=StimulusTable]";
	shell->doUseClock( path0, "process", 4);
	shell->doUseClock( basePath + "/stoich", "process", 5);
	/*
	Id meshEntry = Neutral::child( mesh.eref(), "mesh" );
	assert( meshEntry != Id() );
	*/
}

void SimManager::buildSmoldyn( Shell* shell )
{
}

void SimManager::buildGsl( const Eref& e, const Qinfo* q, 
	Shell* shell, const string& method )
{
	vector< int > dims( 1, 1 );

	string basePath = baseId_.path();
	Id compt( basePath + "/kinetics" );
	assert( compt != Id() );
	Id mesh( basePath + "/kinetics/mesh" );
	assert( compt != Id() );

	stoich_ = shell->doCreate( "GslStoich", compt, "stoich", dims );
	Field< string >::set( stoich_, "path", 
					baseId_.path() + "/kinetics/##");

	MsgId mid = shell->doAddMsg( "OneToAll", mesh, "remesh", 
		stoich_, "remesh" );
	assert( mid != Msg::bad );

	double chemLoad = estimateChemLoad( compt, stoich_ );
	// Here we would also estimate cell load
	Id hsolver;
	double hsolveLoad = estimateHsolveLoad( hsolver );

	numChemNodes_ = Shell::numNodes() * chemLoad / ( chemLoad + hsolveLoad);
	
//	bool ret = Field< Id >::set( stoich_, "compartment", compt );
	// assert( ret );
	bool ret = Field< string >::set( stoich_, "method", method );
	assert( ret );
	// The GSL does some massaging of the method string, so we ask it back.
	method_ = Field< string >::get( stoich_, "method" );
	string path0 = basePath + "/kinetics/mesh," + 
		basePath + "/kinetics/##[ISA=StimulusTable]";
	shell->doUseClock( path0, "process", 4);
	shell->doUseClock( basePath + "/kinetics/stoich", "process", 5);
}

void SimManager::buildFromKkitTree( const Eref& e, const Qinfo* q,
	const string& method )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	autoPlot_ = 0;
	vector< int > dims( 1, 1 );

	shell->doSetClock( 0, 0 );
	shell->doSetClock( 1, 0 );
	shell->doSetClock( 2, 0 );
	shell->doSetClock( 3, 0 );
	shell->doSetClock( 4, plotdt_ );
	shell->doSetClock( 5, plotdt_ );
	shell->doSetClock( 6, simdt_ );
	shell->doSetClock( 8, plotdt_ );
	shell->doSetClock( 9, 0 );

	Field< double >::set( Id( 1 ), "runTime", runTime_ );

	string basePath = baseId_.path();
	if ( method == "Gillespie" || method == "gillespie" || 
		method == "GSSA" || method == "gssa" || method == "Gssa" ) {
		method_ = "gssa";
		buildGssa( e, q, shell );
	} else if ( method == "Neutral" || method == "ee" || method == "EE" ) {
		buildEE( shell );
		method_ = "ee";
	} else if ( method == "Smoldyn" || method == "smoldyn" ) {
		buildSmoldyn( shell );
		method_ = "smoldyn";
	} else {
		buildGsl( e, q, shell, method );
	}

	string plotpath = basePath + "/graphs/##[TYPE=Table]," + 
		basePath + "/moregraphs/##[TYPE=Table]";
	vector< Id > list;
	if ( wildcardFind( plotpath, list ) > 0 )
		shell->doUseClock( plotpath, "process", 8 );
	string stimpath = basePath + "/kinetics/##[TYPE=PulseGen]";
	if ( wildcardFind( stimpath, list ) > 0 )
		shell->doUseClock( stimpath, "process", 6 );
	
	// shell->doReinit(); // Cannot use unless process is running.
}

