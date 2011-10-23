/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SingleMsg.h"
#include "DiagonalMsg.h"
#include "OneToOneMsg.h"
#include "OneToAllMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "ReduceMsg.h"
#include "ReduceFinfo.h"
#include "Shell.h"
#include "Dinfo.h"
#include "Wildcard.h"

// Want to separate out this search path into the Makefile options
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

#include "../mesh/MeshEntry.h"
// #include "../mesh/Boundary.h"
// #include "../mesh/ChemMesh.h"

const unsigned int Shell::OkStatus = ~0;
const unsigned int Shell::ErrorStatus = ~1;

bool Shell::isBlockedOnParser_ = 0;
bool Shell::keepLooping_ = 0;
unsigned int Shell::numCores_;
unsigned int Shell::numProcessThreads_;
unsigned int Shell::numNodes_;
unsigned int Shell::myNode_;
ProcInfo Shell::p_;
vector< ProcInfo> Shell::threadProcs_(1);
unsigned int Shell::numAcks_ = 0;
vector< unsigned int > Shell::acked_( 1, 0 );
pthread_t* Shell::threads_( 0 );
pthread_mutex_t* Shell::parserMutex_( 0 );
pthread_cond_t* Shell::parserBlockCond_( 0 );
pthread_attr_t* Shell::attr_( 0 );
FuncBarrier* Shell::barrier1_( 0 );
FuncBarrier* Shell::barrier2_( 0 );
FuncBarrier* Shell::barrier3_( 0 );
bool Shell::doReinit_( 0 );
bool Shell::isParserIdle_( 0 );
double Shell::runtime_( 0.0 );

const ThreadId ScriptThreadNum = 0;

static SrcFinfo5< string, Id, Id, string, vector< int > > requestCreate( "requestCreate",
			"requestCreate( class, parent, newElm, name, dimensions ): "
			"creates a new Element on all nodes with the specified Id. "
			"Initiates a callback to indicate completion of operation. "
			"Goes to all nodes including self."
			);

SrcFinfo2< unsigned int, unsigned int >* ack()
{
	static SrcFinfo2< unsigned int, unsigned int > temp( "ack",
			"ack( unsigned int node#, unsigned int status ):"
			"Acknowledges receipt and completion of a command on a worker node."
			"Goes back only to master node."
			);
	return &temp;
}

static SrcFinfo1< Id  > requestDelete( "requestDelete",
			"requestDelete( doomedElement ):"
			"Deletes specified Element on all nodes."
			"Initiates a callback to indicate completion of operation."
			"Goes to all nodes including self." ); 

static SrcFinfo0 requestQuit( "requestQuit",
			"requestQuit():"
			"Emerges from the inner loop, and wraps up. No return value." );
static SrcFinfo1< double > requestStart( "requestStart",
			"requestStart( runtime ):"
			"Starts a simulation. Goes to all nodes including self."
			"Initiates a callback to indicate completion of run."
			);
static SrcFinfo1< unsigned int > requestStep( "requestStep",
			"requestStep():"
			"Advances a simulation for the specified # of steps."
			"Goes to all nodes including self."
			);
static SrcFinfo0 requestStop( "requestStop",
			"requestStop():"
			"Gently stops a simulation after completing current ops."
			"After this op it is save to do 'start' again, and it will"
			"resume where it left off"
			"Goes to all nodes including self."
			);
static SrcFinfo2< unsigned int, double > requestSetupTick( 
			"requestSetupTick",
			"requestSetupTick():"
			"Asks the Clock to coordinate the assignment of a specific"
			"clock tick. Args: Tick#, dt."
			"Goes to all nodes including self."
			);
static SrcFinfo0 requestReinit( "requestReinit",
			"requestReinit():"
			"Reinits a simulation: sets to time 0."
			"If simulation is running it stops it first."
			"Goes to all nodes including self."
			);

static SrcFinfo0 requestTerminate( "requestTerminate",
			"requestTerminate():"
			"Violently stops a simulation, possibly leaving things half-done."
			"Goes to all nodes including self."
			);
static SrcFinfo6< string, MsgId, ObjId, string, ObjId, string > 
		requestAddMsg( 
			"requestAddMsg",
			"requestAddMsg( type, src, srcField, dest, destField );"
			"Creates specified Msg between specified Element on all nodes."
			"Initiates a callback to indicate completion of operation."
			"Goes to all nodes including self."
			); 

SrcFinfo4< Id, DataId, FuncId, PrepackedBuffer >* requestSet()
{
	static SrcFinfo4< Id, DataId, FuncId, PrepackedBuffer > temp(
			"requestSet",
			"requestSet( tgtId, tgtDataId, tgtFieldId, value ):"
			"Assigns a value on target field."
			);
	return &temp;
}

static SrcFinfo2< Id, Id > requestMove(
			"move",
			"move( origId, newParent);"
			"Moves origId to become a child of newParent"
			);
static SrcFinfo5< vector< Id >, string, unsigned int, bool, bool > requestCopy(
			"copy",
			"copy( origId, newParent, numRepeats, toGlobal, copyExtMsg );"
			"Copies origId to become a child of newParent"
			);
static SrcFinfo3< string, string, unsigned int > requestUseClock(
			"useClock",
			"useClock( path, field, tick# );"
			"Specifies which clock tick to use for all elements in Path."
			"The 'field' is typically process, but some cases need to send"
			"updates to the 'init' field."
			"Tick # specifies which tick to be attached to the objects."
			);

static DestFinfo handleUseClock( "handleUseClock", 
			"Deals with assignment of path to a given clock.",
			new EpFunc3< Shell, string, string, unsigned int >( 
				&Shell::handleUseClock )
			);

static DestFinfo handleCreate( "create", 
			"create( class, parent, newElm, name, dimensions )",
			new EpFunc5< Shell, string, Id, Id, string, vector< int > >( &Shell::handleCreate ) );

static DestFinfo handleDelete( "delete", 
			"Destroys Element, all its messages, and all its children. Args: Id",
			new EpFunc1< Shell, Id >( & Shell::destroy ) );

DestFinfo* handleAck()
{
	static DestFinfo temp( "handleAck", 
			"Keeps track of # of acks to a blocking shell command. Arg: Source node num.",
			new OpFunc2< Shell, unsigned int, unsigned int >( 
				& Shell::handleAck ) );
	return &temp;
}

static DestFinfo handleAddMsg( "handleAddMsg", 
			"Makes a msg",
			new EpFunc6< Shell, string, MsgId, ObjId, string, ObjId, string >
				( & Shell::handleAddMsg ) );

static DestFinfo handleQuit( "handleQuit", 
			"Stops simulation running and quits the simulator",
			new OpFunc0< Shell >( & Shell::handleQuit ) );

/*
static DestFinfo handleSet( "handleSet", 
			"Deals with request, to set specified field on any node to a value.",
			new EpFunc4< Shell, Id, DataId, FuncId, PrepackedBuffer >( 
				&Shell::handleSet )
			);
			*/


/**
 * Sequence is:
 * innerDispatchGet->requestGet->handleGet->lowLevelGet->get_field->
 * 	receiveGet->completeGet
 */

/*
SrcFinfo4< Id, DataId, FuncId, unsigned int >* requestGet()
{
	static SrcFinfo4< Id, DataId, FuncId, unsigned int > temp( 
			"requestGet",
			"Function to request another Element for a value."
			"Args: Id of target, DataId of target, "
			"FuncId identifying field, int to specify # of entries to get."
			);
	return &temp;
}

static DestFinfo handleGet( "handleGet", 
			"handleGet( Id elementId, DataId index, FuncId fid )"
			"Deals with requestGet, to get specified field from any node.",
			new EpFunc4< Shell, Id, DataId, FuncId, unsigned int >( 
				&Shell::handleGet )
			);

SrcFinfo1< PrepackedBuffer >* lowLevelSetGet() {
	static SrcFinfo1< PrepackedBuffer > temp(
			"lowLevelSetGet",
			"lowlevelSetGet():"
			"Low-level SrcFinfo. Not for external use, internally used as"
			"a handle to set or get a single or vector value from "
			" target field."
	);
	return &temp;
}
*/

DestFinfo* receiveGet() {
	static DestFinfo temp( "receiveGet", 
		"receiveGet( Uint node#, Uint status, PrepackedBuffer data )"
		"Function on master shell that handles the value relayed from worker.",
		new EpFunc1< Shell, PrepackedBuffer >( &Shell::recvGet )
	);
	return &temp;
}

/** Deprecated?
static SrcFinfo3< unsigned int, unsigned int, PrepackedBuffer > relayGet(
	"relayGet",
	"relayGet( node, status, data ): Passes 'get' data back to master node"
);
*/

static DestFinfo handleMove( "move", 
		"handleMove( Id orig, Id newParent ): "
		"moves an Element to a new parent",
	new EpFunc2< Shell, Id, Id >( & Shell::handleMove ) );

static DestFinfo handleCopy( "handleCopy", 
		"handleCopy( vector< Id > args, string newName, unsigned int nCopies, bool toGlobal, bool copyExtMsgs ): "
		" The vector< Id > has Id orig, Id newParent, Id newElm. " 
		"This function copies an Element and all its children to a new parent."
		" May also expand out the original into nCopies copies."
		" Normally all messages within the copy tree are also copied. "
		" If the flag copyExtMsgs is true, then all msgs going out are also copied.",
			new EpFunc5< Shell, vector< Id >, string, unsigned int, bool, bool >( 
				& Shell::handleCopy ) );

static SrcFinfo2< Id, FuncId > requestSync(
			"sync",
			"sync( ElementId, FuncId );"
			"Synchronizes Element data indexing across all nodes."
			"Used when distributed ops like message setup might set up"
			"different #s of data entries on Elements on different nodes."
			"The ElementId is the element being synchronized."
			"The FuncId is the 'get' function for the synchronized field."
			);
static DestFinfo handleSync( "handleSync", 
		"handleSync( Id Element): "
		"Synchronizes DataHandler indexing across nodes"
		"The ElementId is the element being synchronized."
		"The FuncId is the 'get' function for the synchronized field.",
	new EpFunc2< Shell, Id, FuncId >( & Shell::handleSync ) );

static SrcFinfo1< Id > requestReMesh(
			"requestReMesh",
			"requestReMesh( meshId );"
			"Chops up specified mesh."
			);
static DestFinfo handleReMesh( "handleReMesh", 
		"handleReMesh( Id BaseMesh): "
		"Deals with outcome of resizing the meshing in a cellular"
		"compartment (the ChemMesh class). The mesh change has to"
		"propagate down to the molecules and reactions managed by this."
		"Mesh. The ElementId is the mesh being synchronized.",
	new OpFunc1< Shell, Id >( & Shell::handleReMesh ) );

static SrcFinfo1< bool > requestSetParserIdleFlag(
			"requestSetParserIdleFlag",
			"SetParserIdleFlag( bool isParserIdle );"
			"When True, the main ProcessLoop waits a little each cycle"
			"so as to avoid pounding on the CPU."
			);
static DestFinfo handleSetParserIdleFlag( "handleSetParserIdleFlag", 
		"handleSetParserIdleFlag( bool isParserIdle ): "
		"When True, tells the ProcessLoop to wait as the Parser is idle.",
	new OpFunc1< Shell, bool >( & Shell::handleSetParserIdleFlag ) );


static Finfo* shellMaster[] = {
	&requestCreate, &requestDelete,
	&requestAddMsg, 
	&requestQuit,
	&requestMove, &requestCopy, &requestUseClock,
	&requestSync, &requestReMesh, &requestSetParserIdleFlag,
	handleAck() };
static Finfo* shellWorker[] = {
	&handleCreate, &handleDelete,
		&handleAddMsg,
		&handleQuit,
		&handleMove, &handleCopy, &handleUseClock,
		&handleSync, &handleReMesh, &handleSetParserIdleFlag,
	ack() };

static Finfo* clockControlFinfos[] = 
{
	&requestStart, &requestStep, &requestStop, &requestSetupTick,
	&requestReinit, handleAck()
};

ReduceFinfoBase* reduceArraySizeFinfo()
{
	static ReduceFinfo< Shell, unsigned int, ReduceFieldDimension >
		reduceArraySize(
		"reduceArraySize",
		"Look up maximum value of an index, here ragged array size,"
		"across many nodes, and assign uniformly to all nodes. Normally"
		"followed by an operation to assign the size to the object that"
		"was resized.",
		&Shell::digestReduceFieldDimension
	);
	return &reduceArraySize;
}

const Cinfo* Shell::initCinfo()
{
////////////////////////////////////////////////////////////////
// Value Finfos
////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< Shell, bool > isRunning( 
			"isRunning",
			"Flag: Checks if simulation is in progress",
			&Shell::isRunning );

	static ValueFinfo< Shell, Id > cwe( 
			"cwe",
			"Current working Element",
			&Shell::setCwe,
			&Shell::getCwe );

////////////////////////////////////////////////////////////////
// Dest Finfos: Functions handled by Shell
////////////////////////////////////////////////////////////////
		static DestFinfo setclock( "setclock", 
			"Assigns clock ticks. Args: tick#, dt",
			new OpFunc2< Shell, unsigned int, double >( & Shell::doSetClock ) );
		static SharedFinfo master( "master",
			"Issues commands from master shell to worker shells located "
			"on different nodes. Also handles acknowledgements from them.",
			shellMaster, sizeof( shellMaster ) / sizeof( const Finfo* )
		);
		static SharedFinfo worker( "worker",
			"Handles commands arriving from master shell on node 0."
			"Sends out acknowledgements from them.",
			shellWorker, sizeof( shellWorker ) / sizeof( const Finfo* )

		);
		static SharedFinfo clockControl( "clockControl",
			"Controls the system Clock",
			clockControlFinfos, 
			sizeof( clockControlFinfos ) / sizeof( const Finfo* )
		);
	
	static Finfo* shellFinfos[] = {
		receiveGet(),
		&setclock,

////////////////////////////////////////////////////////////////
//  Predefined Msg Src and MsgDests.
////////////////////////////////////////////////////////////////
		reduceArraySizeFinfo(),
////////////////////////////////////////////////////////////////
//  Shared msg
////////////////////////////////////////////////////////////////
		&master,
		&worker,
		&clockControl,
	};

	static Cinfo shellCinfo (
		"Shell",
		Neutral::initCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		new Dinfo< Shell >()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = Shell::initCinfo();


Shell::Shell()
	: 
		gettingVector_( 0 ),
		numGetVecReturns_( 0 ),
		cwe_( Id() ),
		reduceMsg_( Msg::bad )
{
	// cout << myNode() << ": fids\n";
	// shellCinfo->reportFids();
	getBuf_.resize( 1, 0 );
}

Shell::~Shell()
{
	clearGetBuf();
}

void Shell::setShellElement( Element* shelle )
{
	shelle_ = shelle;
}
////////////////////////////////////////////////////////////////
// Parser functions.
////////////////////////////////////////////////////////////////

/**
 * This is the version used by the parser. Acts as a blocking,
 * serial-like interface to a potentially multithread, multinode call.
 * Returns the new Id index.
 * The data of the new Element is not necessarily allocated at this point,
 * that can be deferred till the global Instantiate or Reset calls.
 * Idea is that the model should be fully defined before load balancing.
 *
 */
Id Shell::doCreate( string type, Id parent, string name, vector< int > dimensions, bool isGlobal )
{
	Id ret = Id::nextId();
	vector< int > dims( dimensions );
	dims.push_back( isGlobal );
	initAck(); // Nasty thread stuff happens here for multithread mode.
		requestCreate.send( Id().eref(), ScriptThreadNum, type, parent, ret, name, dims );
	waitForAck();
	return ret;
}

bool Shell::doDelete( Id i )
{
	initAck();
		requestDelete.send( Id().eref(), ScriptThreadNum, i );
	waitForAck();
	return 1;
}

MsgId Shell::doAddMsg( const string& msgType, 
	ObjId src, const string& srcField, 
	ObjId dest, const string& destField )
{
	if ( !src.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: src not found\n";
		return Msg::bad;
	}
	if ( !dest.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: dest not found\n";
		return Msg::bad;
	}
	const Finfo* f1 = src.id()->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src: " << src.id()->getName() << "\n";
		return Msg::bad;
	}
	const Finfo* f2 = dest.id()->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << destField << 
			" on dest: " << dest.id()->getName() << "\n";
		return Msg::bad;
	}
	if ( ! f1->checkTarget( f2 ) ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Src/Dest Msg type mismatch: " << srcField << "/" << destField << endl;
		return Msg::bad;
	}
	initAck();
	MsgId mid = Msg::nextMsgId();
	requestAddMsg.send( Eref( shelle_, 0 ), ScriptThreadNum, 
		msgType, mid, src, srcField, dest, destField );
	//	Qinfo::clearQ( &p_ );
	waitForAck();
	return latestMsgId_;
}

/**
 * Static function, sets up the master message that connects
 * all shells on all nodes to each other. Uses low-level calls to
 * do so.
 */
void Shell::connectMasterMsg()
{
	Id shellId( 0 );
	Element* shelle = shellId();
	const Finfo* f1 = shelle->cinfo()->findFinfo( "master" );
	if ( !f1 ) {
		cout << "Error: Shell::connectMasterMsg: failed to find 'master' msg\n";
		exit( 0 ); // Bad!
	}
	const Finfo* f2 = shelle->cinfo()->findFinfo( "worker" );
	if ( !f2 ) {
		cout << "Error: Shell::connectMasterMsg: failed to find 'worker' msg\n";
		exit( 0 ); // Bad!
	}

	Msg* m = 0;
		m = new OneToOneMsg( Msg::nextMsgId(), shelle, shelle );
	if ( m ) {
		if ( !f1->addMsg( f2, m->mid(), shelle ) ) {
			cout << "Error: failed in Shell::connectMasterMsg()\n";
			delete m; // Nasty, but rare.
		}
	} else {
		cout << Shell::myNode() << ": Error: failed in Shell::connectMasterMsg()\n";
		exit( 0 );
	}
	// cout << Shell::myNode() << ": Shell::connectMasterMsg gave id: " << m->mid() << "\n";

	Id clockId( 1 );
	Shell* s = reinterpret_cast< Shell* >( shellId.eref().data() );
	bool ret = s->innerAddMsg( "Single", Msg::nextMsgId(), 
		ObjId( shellId, 0 ), "clockControl", 
		ObjId( clockId, 0 ), "clockControl" );
	assert( ret );
	// innerAddMsg( string msgType, ObjId src, string srcField, ObjId dest, string destField )
}

void Shell::doQuit( )
{
	// No acks needed: the next call from parser should be to 
	// exit parser itself.
	requestQuit.send( Id().eref(), ScriptThreadNum );
}

void Shell::doStart( double runtime )
{
	Eref sheller( shelle_, 0 );
	// Check if sim not yet initialized. Do it if needed.

	// Then actually run simulation.
	initAck();
		requestStart.send( sheller, ScriptThreadNum, runtime );
	waitForAck();
	// cout << Shell::myNode() << ": Shell::doStart(" << runtime << ")" << endl;
}

void Shell::doNonBlockingStart( double runtime )
{
	Eref sheller( shelle_, 0 );
	// Check if sim not yet initialized. Do it if needed.

	requestStart.send( sheller, ScriptThreadNum, runtime );
}

void Shell::doReinit()
{
	Eref sheller( shelle_, 0 );
	initAck();
		requestReinit.send( sheller, ScriptThreadNum );
	waitForAck();
}

void Shell::doStop()
{
	Eref sheller( shelle_, 0 );
	initAck();
		requestStop.send( sheller, ScriptThreadNum );
	waitForAck();
}
////////////////////////////////////////////////////////////////////////

void Shell::doSetClock( unsigned int tickNum, double dt )
{
	/*
	// if ( q->addToStructuralQ() ) return;
	Eref ce = Id( 1 ).eref();
	assert( ce.element() );
	// We do NOT go through the message queuing here, as the clock is
	// always local and this operation fiddles with scheduling.
	Clock* clock = reinterpret_cast< Clock* >( ce.data() );
	clock->setupTick( tickNum, dt );
	*/
	/*
	Eref sheller( shelle_, 0 );
	initAck(); // This causes a race condition in testShellAddMsg:771
		requestSetupTick.send( sheller, ScriptThreadNum, tickNum, dt );
	waitForAck();
	*/
	SetGet2< unsigned int, double >::set( ObjId( 1 ), "setupTick", tickNum, dt );
}

void Shell::doUseClock( string path, string field, unsigned int tick )
{
	Eref sheller( shelle_, 0 );
	initAck();
		requestUseClock.send( sheller, ScriptThreadNum, path, field, tick );
	waitForAck();
}

////////////////////////////////////////////////////////////////////////

void Shell::doMove( Id orig, Id newParent )
{
	if ( orig == Id() ) {
		cout << "Error: Shell::doMove: Cannot move root Element\n";
		return;
	}

	if ( newParent() == 0 ) {
		cout << "Error: Shell::doMove: Cannot move object to null parent \n";
		return;
	}
	if ( Neutral::isDescendant( newParent, orig ) ) {
		cout << "Error: Shell::doMove: Cannot move object to descendant in tree\n";
		return;
		
	}
	// Put in check here that newParent is not a child of orig.
	Eref sheller( shelle_, 0 );
	initAck();
		requestMove.send( sheller, ScriptThreadNum, orig, newParent );
	waitForAck();
}

bool extractIndices( const string& s, vector< unsigned int >& indices )
{
	vector< unsigned int > open;
	vector< unsigned int > close;

	indices.clear();
	if ( s.length() == 0 ) // a plain slash is OK
		return 1;

	if ( s[0] == '[' ) // Cannot open with a brace
		return 0;

	for ( unsigned int i = 0; i < s.length(); ++i ) {
		if ( s[i] == '[' )
			open.push_back( i+1 );
		else if ( s[i] == ']' )
			close.push_back( i );
	}

	if ( open.size() != close.size() )
		return 0;

	const char* str = s.c_str();
	for ( unsigned int i = 0; i < open.size(); ++i ) {
		if ( open[i] > close[i] ) {
			indices.clear();
			return 0;
		} else if ( open[i] == close[i] ) {
			indices.push_back( ~1U ); // []: Indicate any index.
		} else {
			int j = atoi( str + open[i] );
			if ( j >= 0 ) {
				indices.push_back( j );
			} else {
				indices.clear();
				return 0;
			}
		}
	}
	return 1;
}

/**
 * Static func to subdivide a string at the specified separator.
 */
bool Shell::chopString( const string& path, vector< string >& ret, 
	char separator )
{
	// /foo/bar/zod
	// foo/bar/zod
	// ./foo/bar/zod
	// ../foo/bar/zod
	// .
	// /
	// ..
	ret.resize( 0 );
	if ( path.length() == 0 )
		return 1; // Treat it as an absolute path

	bool isAbsolute = 0;
	string temp = path;
	if ( path[0] == separator ) {
		isAbsolute = 1;
		if ( path.length() == 1 )
			return 1;
		temp = temp.substr( 1 );
	}

	string::size_type pos = temp.find_first_of( separator );
	ret.push_back( temp.substr( 0, pos ) );
	while ( pos != string::npos ) {
		temp = temp.substr( pos + 1 );
		if ( temp.length() == 0 )
			break;
		pos = temp.find_first_of( separator );
		ret.push_back( temp.substr( 0, pos ) );
	}
	return isAbsolute;
}

/**
 * static func.
 *
 * Example: /foo/bar[10]/zod[3][4][5] would return:
 * ret: {"foo", "bar", "zod" }
 * index: { {}, {10}, {3,4,5} }
 */
bool Shell::chopPath( const string& path, vector< string >& ret, 
	vector< vector< unsigned int > >& index, Id cwe )
{
	bool isAbsolute = chopString( path, ret, '/' );
	vector< unsigned int > empty;
	if ( isAbsolute ) {
		index.clear();
		index.resize( 1 ); // The zero index is for /root.
	} else {
		index = cwe.element()->dataHandler()->pathIndices( 0 );
	}
	for ( unsigned int i = 0; i < ret.size(); ++i )
	{
		if ( ret[i] == "." )
			continue;
		if ( ret[i] == ".." ) {
			index.pop_back();
			continue;
		}
		index.push_back( empty );
		if ( !extractIndices( ret[i], index.back() ) ) {
			cout << "Error: Shell::chopPath: Failed to parse indices in path '" <<
				path << "'\n";
		}
		if ( index.back().size() > 0 ) {
			unsigned int pos = ret[i].find_first_of( '[' );
			ret[i] = ret[i].substr( 0, pos );
		}
	}

	return isAbsolute;
}

/// non-static func. Returns the Id found by traversing the specified path.
ObjId Shell::doFind( const string& path ) const
{
	Id curr = Id();
	vector< string > names;
	vector< vector< unsigned int > > indices;
	bool isAbsolute = chopPath( path, names, indices, cwe_ );

	if ( !isAbsolute )
		curr = cwe_;
	
	for ( vector< string >::iterator i = names.begin(); 
		i != names.end(); ++i ) {
		if ( *i == "." ) {
		} else if ( *i == ".." ) {
			curr = Neutral::parent( curr.eref() ).id;
		} else {
			curr = Neutral::child( curr.eref(), *i );
		}
	}
	
	assert( curr.element() );
	assert( curr.element()->dataHandler() );
	DataId di = curr.element()->dataHandler()->pathDataId( indices );
	if ( di == DataId::bad )
		return ObjId::bad;
	return ObjId( curr, di );
}

/// Static func.
void Shell::clearRestructuringQ()
{
	// cout << "o";
}

/**
 * This function synchronizes fieldDimension on the DataHandler across 
 * nodes. Used after function calls that might alter the number of
 * Field entries in the table.
 * The tgt is the FieldElement whose fieldDimension needs updating.
 */
void Shell::doSyncDataHandler( Id tgt )
{
	const Finfo* f = tgt()->cinfo()->findFinfo( "get_localNumField" );
	assert( f );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	assert( df );
	FuncId sizeFid = df->getFid();

	FieldDataHandlerBase* fb = 
		dynamic_cast< FieldDataHandlerBase* >( tgt()->dataHandler() );
	if ( !fb ) {
		cout << myNode_ << ": Shell::doSyncDataHandler:Error: target '" <<
			tgt.path() << "' is not a FieldElement\n";
		return;
	}
	Eref sheller( shelle_, 0 );
	initAck();
		requestSync.send( sheller, ScriptThreadNum, tgt, sizeFid );
	waitForAck();
	// Now the data is back, assign the field.
	// Check: will this function work on all nodes? No, only on master.
	// So we have to do the ugly but reliable 'set' function.
	// fv->setFieldDimension( maxIndex_ );
	Field< unsigned int >::set( ObjId( tgt, 0 ), 
		"fieldDimension", maxIndex_ );
	/*
	*/
}

/**
 * Tell all attached pools and vols to update themselves: set their
 * array sizes and set their new volumes.
 * Dangerous function, if called elsewhere it will cause all sorts of
 * structuralQ issues.
 */
void Shell::handleReMesh( Id baseMesh )
{
	static const Finfo* finfo = MeshEntry::initCinfo()->findFinfo( "get_size" );
	static const DestFinfo* df = dynamic_cast< const DestFinfo* >( finfo );
	assert( df );
	vector< Id > tgts;
	unsigned int numTgts = baseMesh()->getInputs( tgts, df );
	assert( tgts.size() == numTgts );
	unsigned int numMeshEntries = 
		baseMesh()->dataHandler()->localEntries() ;
	for ( vector< Id >::iterator i = tgts.begin(); i != tgts.end(); ++i )
	{
		unsigned short depth = i->element()->dataHandler()->pathDepth();
		// Note that the resize command also copies over the data values.
		// bool ret = i->element()->appendDimension( numMeshEntries );
		bool ret = i->element()->resize( depth, numMeshEntries );
		assert( ret );
		// Now we need to tell each tgt to scale its n, rates etc from vol.
	}
}

/**
 * This function builds a reac-diffusion mesh starting at the
 * specified ChemCompt, which houses MeshEntry FieldElements.
 * Assumes that the dimensions of the baseCompartment have just been
 * redefined, and we now need to go through and update the child 
 * reaction system.
 */

void Shell::doReacDiffMesh( Id baseCompartment )
{
	Eref sheller( shelle_, 0 );
	assert( baseCompartment()->dataHandler()->isGlobal() );
	assert( baseCompartment()->cinfo()->isA( "ChemMesh" ) );
	Id baseMesh( baseCompartment.value() + 1 );
	doSyncDataHandler( baseMesh );

	initAck();
		requestReMesh.send( sheller, ScriptThreadNum, baseMesh );
	waitForAck();

	// Traverse all child compts and do their meshes.
	vector< Id > kids;
	Neutral::children( baseCompartment.eref(), kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i)
	{
		if ( i->operator()()->cinfo()->isA( "ChemMesh" ) ) {
			doReacDiffMesh( *i );
		}
	}

	// Here we need to check on any non-matching Reacs and enzymes.
	
}

// We don't need to do this through acks, as it has no effect on processing
// other than to slow it down or speed it up.
void Shell::doSetParserIdleFlag( bool isParserIdle )
{
	Eref sheller( shelle_, 0 );
	requestSetParserIdleFlag.send( sheller, ScriptThreadNum, isParserIdle);
}

void Shell::handleSetParserIdleFlag( bool isParserIdle )
{
	Shell::isParserIdle_ = isParserIdle;
}


////////////////////////////////////////////////////////////////
// DestFuncs
////////////////////////////////////////////////////////////////

string Shell::doVersion()
{
    return MOOSE_VERSION;
}

string Shell::doRevision()
{
    return SVN_REVISION;
}

void Shell::setCwe( Id val )
{
	cwe_ = val;
}

Id Shell::getCwe() const
{
	return cwe_;
}

bool Shell::isRunning() const
{
	Id clockId( 1 );
	assert( clockId() != 0 );

	return ( reinterpret_cast< const Clock* >( clockId.eref().data() ) )->isRunning();
}

const vector< double* >& Shell::getBuf() const
{
	return getBuf_;
}


/**
 * This function handles the message request to create an Element.
 * This request specifies the Id of the new Element and is handled on
 * all nodes.
 *
 * In due course we also have to set up the node decomposition of the
 * Element, but for now the num indicates the total # of array entries.
 * This gets a bit complicated if the Element is a multidim array.
 */
void Shell::handleCreate( const Eref& e, const Qinfo* q, 
	string type, Id parent, Id newElm, string name,
	vector< int > dims )
{
	// cout << myNode_ << ": Shell::handleCreate inner Create done for element " << name << " id " << newElm << endl;
	if ( q->addToStructuralQ() )
		return;

	/*
	vector< int > dims;
	for ( unsigned int i = 0; i < dimensions.size(); ++i ) {
		dims[i] = dimensions[i];
	}
	*/
	innerCreate( type, parent, newElm, name, dims );
//	if ( myNode_ != 0 )
	ack()->send( e, q->threadNum(), Shell::myNode(), OkStatus );
	// cout << myNode_ << ": Shell::handleCreate ack sent" << endl;
}



/**
 * Static utility function. Attaches child element to parent element.
 * Must only be called from functions executing in parallel on all nodes,
 * as it does a local message addition
 */
bool Shell::adopt( Id parent, Id child ) {
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	// static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	// static const FuncId pafid = pf2->getFid();
	static const Finfo* f1 = Neutral::initCinfo()->findFinfo( "childMsg" );

	assert( !( child() == 0 ) );
	assert( !( child == Id() ) );
	assert( !( parent() == 0 ) );

	Msg* m = new OneToAllMsg( Msg::nextMsgId(), parent.eref(), child() );
	assert( m );

	// cout << myNode_ << ", Shell::adopt: mid = " << m->mid() << ", pa =" << parent << "." << parent()->getName() << ", kid=" << child << "." << child()->getName() << "\n";

	if ( !f1->addMsg( pf, m->mid(), parent() ) ) {
		cout << "move: Error: unable to add parent->child msg from " <<
			parent()->getName() << " to " << child()->getName() << "\n";
		return 0;
	}
	child()->setGroup( parent()->getGroup() );
	return 1;
}

unsigned int cleanDimensions( vector< int >& dims )
{
	vector< int > temp = dims;
	dims.resize( 0 );
	unsigned int raggedStart = 0; // Can't have the root being ragged!
	for ( unsigned int i = 0; i < temp.size(); ++i ) {
		if ( temp[i] > 1 )
			dims.push_back( temp[i] );
		else if ( temp[i] < -1 ) {
			dims.push_back( -temp[i] );
			raggedStart = i;
		}
	}
	return raggedStart;
}

/**
 * This function actually creates the object. Runs on all nodes.
 */
void Shell::innerCreate( string type, Id parent, Id newElm, string name,
	const vector< int >& dimensions )
{
	assert( dimensions.size() >= 1 );
	// cout << "in Shell::innerCreate for " << parent.path() << "/" << name << endl << flush;
	const Cinfo* c = Cinfo::find( type );
	if ( c ) {
		Element* pa = parent();
		if ( !pa ) {
			stringstream ss;
			ss << "innerCreate: Parent Element'" << parent << "' not found. No Element created";
			warning( ss.str() );
			return;
		}
		vector< int > dims( dimensions );
		bool isGlobal = dims.back();
		dims.pop_back();
		cleanDimensions( dims );
		vector< DimInfo > myDims = pa->dataHandler()->dims();
		unsigned int myPathDepth = pa->dataHandler()->pathDepth() + 1;
		for ( unsigned int i = 0; i < dims.size(); ++i ) {
			if ( dims[i] > 1 || dims[i] < -1 ) {
				DimInfo di = { dims[i], myPathDepth, 0 };
				myDims.push_back( di );
			}
		}
		Element* ret = new Element( newElm, c, name, myDims, 
			myPathDepth, isGlobal);
		assert( ret );
		adopt( parent, newElm );

	} else {
		stringstream ss;
		ss << "innerCreate: Class '" << type << "' not known. No Element created";
		warning( ss.str() );
	}
}

void Shell::destroy( const Eref& e, const Qinfo* q, Id eid)
{
	if ( q->addToStructuralQ() )
		return;

	Neutral *n = reinterpret_cast< Neutral* >( e.data() );
	assert( n );
	n->destroy( eid.eref(), 0, 0 );
	// eid.destroy();
	// cout << myNode_ << ": Shell::destroy done for element id " << eid << endl;

	ack()->send( e, q->threadNum(), Shell::myNode(), OkStatus );
}


/**
 * Wrapper function, that does the ack. Other functions also use the
 * inner function to build message trees, so we don't want it to emit
 * multiple acks.
 */
void Shell::handleAddMsg( const Eref& e, const Qinfo* q,
	string msgType, MsgId mid, ObjId src, string srcField, 
	ObjId dest, string destField )
{
	if ( q->addToStructuralQ() )
		return;
	if ( innerAddMsg( msgType, mid, src, srcField, dest, destField ) )
		ack()->send( Eref( shelle_, 0 ), q->threadNum(), Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0), q->threadNum(), Shell::myNode(), ErrorStatus );
}

/**
 * The actual function that adds messages. Does NOT send an ack.
 */
bool Shell::innerAddMsg( string msgType, MsgId mid,
	ObjId src, string srcField, 
	ObjId dest, string destField )
{
	/*
	cout << myNode_ << ", Shell::handleAddMsg: " << 
		msgType << ", " << mid <<
		", src =" << src << "." << srcField << 
		", dest =" << dest << "." << destField << "\n";
		*/
	const Finfo* f1 = src.id()->cinfo()->findFinfo( srcField );
	if ( f1 == 0 ) return 0;
	// assert( f1 != 0 );
	const Finfo* f2 = dest.id()->cinfo()->findFinfo( destField );
	if ( f2 == 0 ) return 0;
	// assert( f2 != 0 );
	
	// Should have been done before msgs request went out.
	assert( f1->checkTarget( f2 ) );

	latestMsgId_ = Msg::bad;

	Msg *m = 0;
	if ( msgType == "diagonal" || msgType == "Diagonal" ) {
		m = new DiagonalMsg( mid, src.id(), dest.id() );
	} else if ( msgType == "sparse" || msgType == "Sparse" ) {
		m = new SparseMsg( mid, src.id(), dest.id() );
	} else if ( msgType == "Single" || msgType == "single" ) {
		m = new SingleMsg( mid, src.eref(), dest.eref() );
	} else if ( msgType == "OneToAll" || msgType == "oneToAll" ) {
		m = new OneToAllMsg( mid, src.eref(), dest.id() );
	} else if ( msgType == "AllToOne" || msgType == "allToOne" ) {
		m = new OneToAllMsg( mid, dest.eref(), src.id() ); // Little hack.
	} else if ( msgType == "OneToOne" || msgType == "oneToOne" ) {
		m = new OneToOneMsg( mid, src.id(), dest.id() );
	} else if ( msgType == "Reduce" || msgType == "reduce" ) {
		const ReduceFinfoBase* rfb = 
			dynamic_cast< const ReduceFinfoBase* >( f1 );
		assert( rfb );
		m = new ReduceMsg( mid, src.eref(), dest.eref().element(), rfb );
	} else {
		cout << myNode_ << 
			": Error: Shell::handleAddMsg: msgType not known: "
			<< msgType << endl;
		// ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus );
		return 0;
	}
	if ( m ) {
		if ( f1->addMsg( f2, m->mid(), src.id() ) ) {
			latestMsgId_ = m->mid();
			// ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus );
			return 1;
		}
		delete m;
	}
	cout << myNode_ << 
			": Error: Shell::handleAddMsg: Unable to make/connect Msg: "
			<< msgType << " from " << src.id()->getName() <<
			" to " << dest.id()->getName() << endl;
	return 1;
//	ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus );
}

static bool changeTreeDepth( Id id, short delta )
{
	DataHandler* dh = id.element()->dataHandler();
	if ( delta < 0 ) {
		unsigned short i = -delta;
		if ( i > dh->pathDepth() )
			return 0;
	}

	unsigned short newDepth = dh->pathDepth() + delta;
	if ( !dh->changeDepth( newDepth ) )
		return 0;
	
	vector< Id > children;
	Neutral::children( id.eref(), children );
	bool ret = 1;
	for ( unsigned int i = 0; i < children.size(); ++i ) {
		ret &= changeTreeDepth( children[i], delta );
	}
	return ret;
}

void Shell::handleMove( const Eref& e, const Qinfo* q,
	Id orig, Id newParent )
{
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* f1 = Neutral::initCinfo()->findFinfo( "childMsg" );

	assert( !( orig == Id() ) );
	assert( !( newParent() == 0 ) );

	if ( q->addToStructuralQ() )
		return;
	
	short origDepth = orig.element()->dataHandler()->pathDepth();
	short newDepth = 1 +
		newParent.element()->dataHandler()->pathDepth();
	
	if ( !changeTreeDepth( orig, newDepth - origDepth ) )
		return;

	MsgId mid = orig()->findCaller( pafid );
	Msg::deleteMsg( mid );

	Msg* m = new OneToAllMsg( Msg::nextMsgId(), newParent.eref(), orig() );
	assert( m );
	if ( !f1->addMsg( pf, m->mid(), newParent() ) ) {
		cout << "move: Error: unable to add parent->child msg from " <<
			newParent()->getName() << " to " << orig()->getName() << "\n";
		return;
	}
	
	ack()->send( Eref( shelle_, 0 ), q->threadNum(), Shell::myNode(), OkStatus );
}

void Shell::handleUseClock( const Eref& e, const Qinfo* q,
	string path, string field, unsigned int tick)
{
	if ( q->addToStructuralQ() )
		return;
	// cout << q->getProcInfo()->threadIndexInGroup << ": in Shell::handleUseClock with path " << path << endl << flush;
	vector< Id > list;
	wildcard( path, list ); // By default scans only Elements.
	if ( list.size() == 0 ) {
		cout << "Warning: no Elements found on path " << path << endl;
		ack()->send( Eref( shelle_, 0 ), q->threadNum(), Shell::myNode(), OkStatus );
		return;
	}
	string tickField = "proc";
	// Hack to get around a common error.
	if ( field.substr( 0, 4 ) == "proc" || field.substr( 0, 4 ) == "Proc" )
		field = "proc"; 
	/*
	string tickField = "process";
	if ( field.substr( 0, 4 ) == "proc" || field.substr( 0, 4 ) == "Proc" )
		field = tickField = "proc"; // Use the shared Msg with process and reinit.
		*/
	for ( vector< Id >::iterator i = list.begin(); i != list.end(); ++i ) {
		stringstream ss;
		ObjId tickId( Id( 2 ), DataId( tick ) );
		ss << tickField << tick;
		// bool ret = 
		innerAddMsg( "OneToAll", Msg::nextMsgId(), 
			tickId, ss.str(), 
			ObjId( *i, 0 ), field);
		// We just skip messages that don't work.
		/*
		if ( !ret ) {
			cout << Shell::myNode() << "Error: Shell::handleUseClock: Messaging failed\n";
			ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), 
				ErrorStatus );
			return;
		}
		*/
	}
	ack()->send( Eref( shelle_, 0 ), q->threadNum(), Shell::myNode(), OkStatus );
}

void Shell::handleQuit()
{
	Shell::keepLooping_ = 0;
}

void Shell::warning( const string& text )
{
	cout << "Warning: Shell:: " << text << endl;
}

void Shell::error( const string& text )
{
	cout << "Error: Shell:: " << text << endl;
}

void Shell::wildcard( const string& path, vector< Id >& list )
{
	wildcardFind( path, list );
}

const ProcInfo* Shell::getProcInfo( unsigned int index ) const
{
	assert( index < threadProcs_.size() );
	return &( threadProcs_[index] );
}

///////////////////////////////////////////////////////////////////////////
// Functions for handling acks for blocking Shell function calls are
// moved to ShellThreads.cpp
///////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// Some static utility functions
////////////////////////////////////////////////////////////////////////

// Statid func for returning the pet ProcInfo of the shell.
const ProcInfo* Shell::procInfo()
{
	return &p_;
}

void Shell::digestReduceFieldDimension( 
	const Eref& er, const ReduceFieldDimension* arg )
{
	maxIndex_ = arg->maxIndex();
	// ack()->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus );
	ack()->send( er, ScriptThreadNum, Shell::myNode(), OkStatus );
}


// Static func
void Shell::cleanSimulation()
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( sheller.data() );
	vector< Id > kids;
	Neutral::children( sheller, kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
	{
		if ( i->value() > 4 ) {
			cout << "Shell::cleanSimulation: deleted cruft at " << 
				i->value() << ": " << i->path() << endl;
			s->doDelete( *i );
		}
	}
}
