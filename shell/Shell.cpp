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
#include "Shell.h"
#include "Dinfo.h"
#include "Wildcard.h"

// Want to separate out this search path into the Makefile options
#include "../scheduling/Clock.h"

#ifdef USE_SBML
#include "../sbml/SbmlWriter.h"
#include "../sbml/SbmlReader.h"
#endif

const unsigned int Shell::OkStatus = ~0;
const unsigned int Shell::ErrorStatus = ~1;

bool Shell::isBlockedOnParser_ = 0;
bool Shell::keepLooping_ = 0;
unsigned int Shell::numCores_;
unsigned int Shell::numNodes_;
unsigned int Shell::myNode_;
ProcInfo Shell::p_;
unsigned int Shell::numAcks_ = 0;
vector< unsigned int > Shell::acked_( 1, 0 );
bool Shell::doReinit_( 0 );
bool Shell::isParserIdle_( 0 );
double Shell::runtime_( 0.0 );

static SrcFinfo5< string, Id, Id, string, vector< int > > *requestCreate() {
	static SrcFinfo5< string, Id, Id, string, vector< int > > requestCreate( "requestCreate",
			"requestCreate( class, parent, newElm, name, dimensions ): "
			"creates a new Element on all nodes with the specified Id. "
			"Initiates a callback to indicate completion of operation. "
			"Goes to all nodes including self."
			);
	return &requestCreate;
}

static SrcFinfo2< unsigned int, unsigned int >* ack()
{
	static SrcFinfo2< unsigned int, unsigned int > temp( "ack",
			"ack( unsigned int node#, unsigned int status ):"
			"Acknowledges receipt and completion of a command on a worker node."
			"Goes back only to master node."
			);
	return &temp;
}

static SrcFinfo1< Id  > *requestDelete() {
	static SrcFinfo1< Id  > requestDelete( "requestDelete",
			"requestDelete( doomedElement ):"
			"Deletes specified Element on all nodes."
			"Initiates a callback to indicate completion of operation."
			"Goes to all nodes including self." ); 
	return &requestDelete;
}

static SrcFinfo0 *requestQuit() {
	static SrcFinfo0 requestQuit( "requestQuit",
			"requestQuit():"
			"Emerges from the inner loop, and wraps up. No return value." );
	return &requestQuit;
}

static SrcFinfo1< double > *requestStart() {
	static SrcFinfo1< double > requestStart( "requestStart",
			"requestStart( runtime ):"
			"Starts a simulation. Goes to all nodes including self."
			"Initiates a callback to indicate completion of run."
			);
	return &requestStart;
}

static SrcFinfo1< unsigned int > *requestStep() {
	static SrcFinfo1< unsigned int > requestStep( "requestStep",
			"requestStep():"
			"Advances a simulation for the specified # of steps."
			"Goes to all nodes including self."
			);
	return &requestStep;
}

static SrcFinfo0 *requestStop() {
	static SrcFinfo0 requestStop( "requestStop",
			"requestStop():"
			"Gently stops a simulation after completing current ops."
			"After this op it is save to do 'start' again, and it will"
			"resume where it left off"
			"Goes to all nodes including self."
			);
	return &requestStop;
}

static SrcFinfo2< unsigned int, double > *requestSetupTick() {
	static SrcFinfo2< unsigned int, double > requestSetupTick( 
			"requestSetupTick",
			"requestSetupTick():"
			"Asks the Clock to coordinate the assignment of a specific"
			"clock tick. Args: Tick#, dt."
			"Goes to all nodes including self."
			);
	return &requestSetupTick;
}

static SrcFinfo0 *requestReinit() {
	static SrcFinfo0 requestReinit( "requestReinit",
			"requestReinit():"
			"Reinits a simulation: sets to time 0."
			"If simulation is running it stops it first."
			"Goes to all nodes including self."
			);
	return &requestReinit;
}

static SrcFinfo6< string, MsgId, ObjId, string, ObjId, string > *requestAddMsg() {
	static SrcFinfo6< string, MsgId, ObjId, string, ObjId, string > 
		requestAddMsg( 
				"requestAddMsg",
				"requestAddMsg( type, src, srcField, dest, destField );"
				"Creates specified Msg between specified Element on all nodes."
				"Initiates a callback to indicate completion of operation."
				"Goes to all nodes including self."
			     ); 
	return &requestAddMsg;
}

static SrcFinfo2< Id, Id > *requestMove() {
	static SrcFinfo2< Id, Id > requestMove(
			"move",
			"move( origId, newParent);"
			"Moves origId to become a child of newParent"
			);
	return &requestMove;
}

static SrcFinfo5< vector< Id >, string, unsigned int, bool, bool > *requestCopy() {
	static SrcFinfo5< vector< Id >, string, unsigned int, bool, bool > requestCopy(
			"copy",
			"copy( origId, newParent, numRepeats, toGlobal, copyExtMsg );"
			"Copies origId to become a child of newParent"
			);
	return &requestCopy;
}

static SrcFinfo3< string, string, unsigned int > *requestUseClock() {
	static SrcFinfo3< string, string, unsigned int > requestUseClock(
			"useClock",
			"useClock( path, field, tick# );"
			"Specifies which clock tick to use for all elements in Path."
			"The 'field' is typically process, but some cases need to send"
			"updates to the 'init' field."
			"Tick # specifies which tick to be attached to the objects."
			);
	return &requestUseClock;
}

static SrcFinfo1< bool > *requestSetParserIdleFlag() {
	static SrcFinfo1< bool > requestSetParserIdleFlag(
			"requestSetParserIdleFlag",
			"SetParserIdleFlag( bool isParserIdle );"
			"When True, the main ProcessLoop waits a little each cycle"
			"so as to avoid pounding on the CPU."
			);
	return &requestSetParserIdleFlag;
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
	static DestFinfo handleAck( "handleAck", 
			"Keeps track of # of acks to a blocking shell command. Arg: Source node num.",
			new OpFunc2< Shell, unsigned int, unsigned int >( 
				& Shell::handleAck ) );
	static Finfo* shellMaster[] = {
		requestCreate(), requestDelete(),
		requestAddMsg(), 
		requestQuit(),
		requestMove(), requestCopy(), requestUseClock(),
		requestSetParserIdleFlag(),
		&handleAck };
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
	static DestFinfo handleAddMsg( "handleAddMsg", 
			"Makes a msg",
			new EpFunc6< Shell, string, MsgId, ObjId, string, ObjId, string >
			( & Shell::handleAddMsg ) );
	static DestFinfo handleQuit( "handleQuit", 
			"Stops simulation running and quits the simulator",
			new OpFunc0< Shell >( & Shell::handleQuit ) );
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
	static DestFinfo handleSync( "handleSync", 
			"handleSync( Id Element): "
			"Synchronizes DataHandler indexing across nodes"
			"The ElementId is the element being synchronized."
			"The FuncId is the 'get' function for the synchronized field.",
			new EpFunc2< Shell, Id, FuncId >( & Shell::handleSync ) );
	static DestFinfo handleReMesh( "handleReMesh", 
			"handleReMesh( Id BaseMesh): "
			"Deals with outcome of resizing the meshing in a cellular"
			"compartment (the ChemCompt class). The mesh change has to"
			"propagate down to the molecules and reactions managed by this."
			"Mesh. The ElementId is the mesh being synchronized.",
			new OpFunc1< Shell, Id >( & Shell::handleReMesh ) );
	static DestFinfo handleSetParserIdleFlag( "handleSetParserIdleFlag", 
			"handleSetParserIdleFlag( bool isParserIdle ): "
			"When True, tells the ProcessLoop to wait as the Parser is idle.",
			new OpFunc1< Shell, bool >( & Shell::handleSetParserIdleFlag ) );

	static Finfo* shellWorker[] = {
		&handleCreate, &handleDelete,
		&handleAddMsg,
		&handleQuit,
		&handleMove, &handleCopy, &handleUseClock,
		&handleSync, &handleReMesh, &handleSetParserIdleFlag,
		ack() };
	static Finfo* clockControlFinfos[] = 
	{
		requestStart(), requestStep(), requestStop(), requestSetupTick(),
		requestReinit(), &handleAck
	};

		static DestFinfo setclock( "setclock", 
			"Assigns clock ticks. Args: tick#, dt",
			new OpFunc3< Shell, unsigned int, double, bool >( & Shell::doSetClock ) );
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
		cwe_( Id() )
{
	getBuf_.resize( 1, 0 );
}

Shell::~Shell()
{;}

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
Id Shell::doCreate( string type, Id parent, string name, vector< int > dimensions, bool isGlobal, bool qFlag )
{
	Id ret = Id::nextId();
	vector< int > dims( dimensions );
	dims.push_back( isGlobal );
		innerCreate( type, parent, ret, name, dims );
	/*
	initAck(); // Nasty thread stuff happens here for multithread mode.
		requestCreate()->send( Id().eref(), ScriptThreadNum, type, parent, ret, name, dims );
	waitForAck();
	*/
	return ret;
}

bool Shell::doDelete( Id i, bool qFlag )
{
	Neutral n;
	n.destroy( i.eref(), 0 );
	return 1;
}

MsgId Shell::doAddMsg( const string& msgType, 
	ObjId src, const string& srcField, 
	ObjId dest, const string& destField, bool qFlag )
{
	if ( !src.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: src not found" << endl;
		return Msg::bad;
	}
	if ( !dest.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: dest not found" << endl;
		return Msg::bad;
	}
	const Finfo* f1 = src.id()->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src: " << src.id()->getName() << endl;
		return Msg::bad;
	}
	const Finfo* f2 = dest.id()->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << destField << 
			" on dest: " << dest.id()->getName() << endl;
		return Msg::bad;
	}
	if ( ! f1->checkTarget( f2 ) ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Src/Dest Msg type mismatch: " << srcField << "/" << destField << endl;
		return Msg::bad;
	}
		MsgId mid = Msg::nextMsgId();
		innerAddMsg( msgType, mid, src, srcField, dest, destField );
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

void Shell::doQuit( bool qFlag )
{
		Shell::keepLooping_ = 0;
}

void Shell::doStart( double runtime, bool qFlag )
{
	Id clockId( 1 );
	SetGet1< double >::set( clockId, "start", runtime );
}

bool isDoingReinit()
{
	static Id clockId( 1 );
	assert( clockId() != 0 );

	return ( reinterpret_cast< const Clock* >( 
		clockId.eref().data() ) )->isDoingReinit();
}

void Shell::doReinit( bool qFlag )
{
	if ( !keepLooping() ) {
		cout << "Error: Shell::doReinit: Should not be called unless ProcessLoop is running\n";
		return;
	}
	Id clockId( 1 );
	SetGet0::set( clockId, "reinit" );
}

void Shell::doStop( bool qFlag )
{
	if ( !keepLooping() ) {
		cout << "Error: Shell::doStop: Should not be called unless ProcessLoop is running\n";
		return;
	}
	Id clockId( 1 );
	SetGet0::set( clockId, "stop" );
}
////////////////////////////////////////////////////////////////////////

void Shell::doSetClock( unsigned int tickNum, double dt, bool qFlag )
{
		SetGet2< unsigned int, double >::set( ObjId( 1 ), "setupTick", tickNum, dt );
}

void Shell::doUseClock( string path, string field, unsigned int tick,
	bool qFlag )
{
	innerUseClock( path, field, tick);
}

/**
 * Write given model to SBML file. Returns success value.
 */
int Shell::doWriteSBML( const string& fname, const string& modelpath,
	bool qFlag )
{
#ifdef USE_SBML
	SbmlWriter sw;
	return sw.write( fname, modelpath );
#else
    cerr << "Shell::WriteSBML: This copy of MOOSE has not been compiled with SBML writing support.\n";
	return 0;
#endif
}
/**
 * read given SBML model to moose. Returns success value.
 */
Id Shell::doReadSBML( const string& fname, const string& modelpath, const string& solverclass,
	bool qFlag )
{
#ifdef USE_SBML
	SbmlReader sr;
	return sr.read( fname, modelpath,solverclass);
#else
    cerr << "Shell::ReadSBML: This copy of MOOSE has not been compiled with SBML reading support.\n";
    return Id();
#endif
}


////////////////////////////////////////////////////////////////////////

void Shell::doMove( Id orig, Id newParent, bool qFlag )
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
		innerMove( orig, newParent );
	// Put in check here that newParent is not a child of orig.
	/*
	Eref sheller( shelle_, 0 );
	initAck();
		requestMove()->send( sheller, ScriptThreadNum, orig, newParent );
	waitForAck();
	*/
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
		static vector< vector< unsigned int > > tempIndex( 1 );
		// index = cwe.element()->pathIndices( 0 );
		index = tempIndex;
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

/*
/// non-static func. Fallback which treats index brackets as part of 
/// name string, and does not try to extract integer indices.
ObjId Shell::doFindWithoutIndexing( const string& path ) const
{
	Id curr = Id();
	vector< string > names;
	vector< vector< unsigned int > > indices;
	bool isAbsolute = chopString( path, names, '/' );

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
	return ObjId( curr, 0 );
}
*/

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
	DataId di = 0; // temporary 03 Nov 2013.
	return ObjId( curr, di );
}

/// Static func.
void Shell::clearRestructuringQ()
{
	// cout << "o";
}

/**
 * This function synchronizes lastDimension (i.e., field Dimension)
 * on the DataHandler across 
 * nodes. Used after function calls that might alter the number of
 * Field entries in the table.
 * The tgt is the FieldElement whose lastDimension needs updating.
 * Deprecated
 */
void Shell::doSyncDataHandler( Id tgt )
{
	/*
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
		requestSync()->send( sheller, ScriptThreadNum, tgt, sizeFid );
	waitForAck();
	// Now the data is back, assign the field.
	// Check: will this function work on all nodes? No, only on master.
	// So we have to do the ugly but reliable 'set' function.
	// fv->setFieldDimension( maxIndex_ );
	Field< unsigned int >::set( ObjId( tgt, 0 ), 
		"lastDimension", maxIndex_ );
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
		/*
	static const Finfo* finfo = MeshEntry::initCinfo()->findFinfo( "get_volume" );
	static const DestFinfo* df = dynamic_cast< const DestFinfo* >( finfo );
	assert( df );
	vector< Id > tgts;
	unsigned int numTgts = baseMesh()->getNeighbours( tgts, df );
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
	*/
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
		/*
	Eref sheller( shelle_, 0 );
	assert( baseCompartment()->dataHandler()->isGlobal() );
	assert( baseCompartment()->cinfo()->isA( "ChemCompt" ) );
	Id baseMesh( baseCompartment.value() + 1 );
	doSyncDataHandler( baseMesh );

	initAck();
		requestReMesh()->send( sheller, ScriptThreadNum, baseMesh );
	waitForAck();

	// Traverse all child compts and do their meshes.
	vector< Id > kids;
	Neutral::children( baseCompartment.eref(), kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i)
	{
		if ( i->operator()()->cinfo()->isA( "ChemCompt" ) ) {
			doReacDiffMesh( *i );
		}
	}

	// Here we need to check on any non-matching Reacs and enzymes.
	// */
	
}

// We don't need to do this through acks, as it has no effect on processing
// other than to slow it down or speed it up.
void Shell::doSetParserIdleFlag( bool isParserIdle )
{
	Eref sheller( shelle_, 0 );
	requestSetParserIdleFlag()->send( sheller, isParserIdle);
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
	static Id clockId( 1 );
	assert( clockId() != 0 );

	return ( reinterpret_cast< const Clock* >( clockId.eref().data() ) )->isRunning();
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
void Shell::handleCreate( const Eref& e,
	string type, Id parent, Id newElm, string name,
	vector< int > dims )
{
	innerCreate( type, parent, newElm, name, dims );
	ack()->send( e, Shell::myNode(), OkStatus );
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
		if ( Neutral::child( parent.eref(), name ) != Id() ) {
			stringstream ss;
			ss << "innerCreate: Object with same name already present: '"
				   	<< parent.path() << "/" << name << "'. No Element created";
			warning( ss.str() );
			return;
		}
		vector< int > dims( dimensions );
		bool isGlobal = dims.back();
		dims.pop_back();
		cleanDimensions( dims );
		unsigned int myPathDepth = 0;
		Element* ret = new Element( newElm, c, name, 
						myPathDepth, isGlobal);
		assert( ret );
		adopt( parent, newElm );

	} else {
		stringstream ss;
		ss << "innerCreate: Class '" << type << "' not known. No Element created";
		warning( ss.str() );
	}
}

void Shell::destroy( const Eref& e, Id eid)
{
	Neutral *n = reinterpret_cast< Neutral* >( e.data() );
	assert( n );
	n->destroy( eid.eref(), 0 );
	// cout << myNode_ << ": Shell::destroy done for element id " << eid << endl;
	if ( cwe_ == eid )
		cwe_ = Id();

	ack()->send( e, Shell::myNode(), OkStatus );
}


/**
 * Wrapper function, that does the ack. Other functions also use the
 * inner function to build message trees, so we don't want it to emit
 * multiple acks.
 */
void Shell::handleAddMsg( const Eref& e,
	string msgType, MsgId mid, ObjId src, string srcField, 
	ObjId dest, string destField )
{
	if ( innerAddMsg( msgType, mid, src, srcField, dest, destField ) )
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0), Shell::myNode(), ErrorStatus );
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
	} else {
		cout << myNode_ << 
			": Error: Shell::handleAddMsg: msgType not known: "
			<< msgType << endl;
		// ack()->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus );
		return 0;
	}
	if ( m ) {
		if ( f1->addMsg( f2, m->mid(), src.id() ) ) {
			latestMsgId_ = m->mid();
			// ack()->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus );
			return 1;
		}
		delete m;
	}
	cout << myNode_ << 
			": Error: Shell::handleAddMsg: Unable to make/connect Msg: "
			<< msgType << " from " << src.id()->getName() <<
			" to " << dest.id()->getName() << endl;
	return 1;
//	ack()->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus );
}

bool Shell::innerMove( Id orig, Id newParent )
{
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* f1 = Neutral::initCinfo()->findFinfo( "childMsg" );

	assert( !( orig == Id() ) );
	assert( !( newParent() == 0 ) );

	MsgId mid = orig()->findCaller( pafid );
	Msg::deleteMsg( mid );

	Msg* m = new OneToAllMsg( Msg::nextMsgId(), newParent.eref(), orig() );
	assert( m );
	if ( !f1->addMsg( pf, m->mid(), newParent() ) ) {
		cout << "move: Error: unable to add parent->child msg from " <<
			newParent()->getName() << " to " << orig()->getName() << "\n";
		return 0;
	}
	return 1;
}

void Shell::handleMove( const Eref& e, Id orig, Id newParent )
{
	
	if ( innerMove( orig, newParent ) )
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), ErrorStatus );
}

void Shell::addClockMsgs( 
	const vector< Id >& list, const string& field, unsigned int tick )
{
	if ( !Id( 2 ).element() )
		return;
	ObjId tickId( Id( 2 ), DataId( tick ) );
	for ( vector< Id >::const_iterator i = list.begin(); 
		i != list.end(); ++i ) {
		if ( i->element() ) {
			stringstream ss;
			ss << "proc" << tick;
			innerAddMsg( "OneToAll", Msg::nextMsgId(), 
				tickId, ss.str(), 
				ObjId( *i, 0 ), field );
		}
	}
}

bool Shell::innerUseClock( string path, string field, unsigned int tick)
{
	vector< Id > list;
	wildcard( path, list ); // By default scans only Elements.
	if ( list.size() == 0 ) {
		cout << "Warning: no Elements found on path " << path << endl;
		return 0;
	}
	// string tickField = "proc";
	// Hack to get around a common error.
	if ( field.substr( 0, 4 ) == "proc" || field.substr( 0, 4 ) == "Proc" )
		field = "proc"; 
	if ( field.substr( 0, 4 ) == "init" || field.substr( 0, 4 ) == "Init" )
		field = "init"; 
	
	addClockMsgs( list, field, tick );
	return 1;
}

void Shell::handleUseClock( const Eref& e, 
	string path, string field, unsigned int tick)
{
	// cout << q->getProcInfo()->threadIndexInGroup << ": in Shell::handleUseClock with path " << path << endl << flush;
	if ( innerUseClock( path, field, tick ) )
		ack()->send( Eref( shelle_, 0 ), 
			Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0 ), 
			Shell::myNode(), ErrorStatus );
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

////////////////////////////////////////////////////////////////////////
// Some static utility functions
////////////////////////////////////////////////////////////////////////

// Statid func for returning the pet ProcInfo of the shell.
const ProcInfo* Shell::procInfo()
{
	return &p_;
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
