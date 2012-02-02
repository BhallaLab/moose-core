/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
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
			"SyncTime is the interval between synchornizing solvers"
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
			"First arg is list of other nodes, second arg is list number of"
			"meshEntries to be transferred for each of these nodes, "
			"third arg is catenated list of meshEntries indices on"
			"my node going to each of the other connected nodes, and"
			"fourth arg is matching list of meshEntries on other nodes",
			new EpFunc4< SimManager, vector< unsigned int >, vector< unsigned int>, vector<     unsigned int >, vector< unsigned int > >( &SimManager::meshSplit )
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
		plotdt_( 1 )
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
	 	Field< unsigned int >::get( stoich, "nVarPools" );
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
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	// First, check if the tree has a compartment/ChemMesh as the base
	// of the chemical system. If not, put in a single-voxel ChemMesh.
	baseId_ = e.id();
	Id mesh = findChemMesh();

	if ( mesh == Id() ) {
		 cout << "SimManager::build: No chem mesh found, still need to sort this out\n";
		 return;
	}

	// Then, do the setup. The setup function is capable of doing the
	// multi-voxel simulation but is fine if there is just a single voxel.
	// Get # of voxels from ChemMesh
	// Get list of local node voxels.
	/* This shared message should allow the mesh to force an update, and
	 * the SimManager to request an update. 
	 * Unfortunately messages do not (yet) work well with setup calls.
	 */
	MsgId mid = shell->doAddMsg( "OneToOne", mesh, "nodeMeshing", 
		baseId_, "nodeMeshing" );
	assert( mid != Msg::bad );

	vector< int > dims( 1, 1 );
	 // This is a placeholder for more sophisticated node-balancing info
	 // May also need to put in volscales here.
	stoich_ = shell->doCreate( "Stoich", baseId_, "stoich", dims );
	Field< string >::set( stoich_, "path", baseId_.path() + "/kinetics/##");
	double chemLoad = estimateChemLoad( mesh, stoich_ );
	// Here we would also estimate cell load
	/*
	Id hsolver = Neutral::child( e, "solve" );
	*/
	Id hsolver;
	double hsolveLoad = estimateHsolveLoad( hsolver );

	numChemNodes_ = Shell::numNodes() * chemLoad / ( chemLoad + hsolveLoad);
	
	/*
		*/
	nodeInfo()->send( e, q->threadNum(), numChemNodes_,
		Shell::numProcessThreads() ); 
	Qinfo::waitProcCycles( 2 );

	cout << "Waited 2 cycles\n";
	buildFromKkitTree( "gsl" );
	// send numChemNodes off to the ChemMesh to come back with the 
	// partitioning rules. This in due course leads to the return message
	// with the mesh partitioning info. The return message in turn triggers
	// the requests for node allocation and the construction of the 
	// GSL or GSSA integrators. The current function does not do these
	// subsequent steps as the outgoing 'send' call is non-blocking.


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

void SimManager::buildFromBareKineticTree( const string& method )
{
	;
}

void SimManager::buildFromKkitTree( const string& method )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	autoPlot_ = 0;
	vector< int > dims( 1, 1 );

	shell->doSetClock( 0, plotdt_ );
	shell->doSetClock( 1, plotdt_ );
	shell->doSetClock( 2, plotdt_ );
	shell->doSetClock( 3, 0 );

	string basePath = baseId_.path();
	Id mesh( basePath + "/kinetics/mesh" );
	assert( mesh != Id() );
	if ( method == "Gillespie" || method == "gillespie" || 
		method == "GSSA" || method == "gssa" || method == "Gssa" ) {
		// Id stoich = shell->doCreate( "Stoich", baseId_, "stoich", dims );
		// Field< string >::set( stoich, "path", basePath + "/##" );
		cout << "SimManager::buildFromKkitTree: Not yet got GSSA working here.\n";
	} else if ( method == "Neutral" || method == "ee" || method == "EE" ) {
		// cout << "SimManager::buildFromKkitTree: No solvers built\n";
		; // Don't make any solvers.
	} else {
		// Id stoich = shell->doCreate( "Stoich", baseId_, "stoich", dims );
		// Field< string >::set( stoich, "path", basePath + "/##" );
		Id gsl = shell->doCreate( "GslIntegrator", stoich_, "gsl", dims );
		bool ret = SetGet1< Id >::set( gsl, "stoich", stoich_ );
		assert( ret );
		ret = Field< bool >::get( gsl, "isInitialized" );
		assert( ret );
		ret = Field< string >::set( gsl, "method", method );
		assert( ret );
		shell->doUseClock( basePath + "/stoich/gsl", "process", 0);
	}

	string plotpath = basePath + "/graphs/##[TYPE=Table]," + 
		basePath + "/moregraphs/##[TYPE=Table]";
	shell->doUseClock( plotpath, "process", 2 );
	shell->doReinit();
}

