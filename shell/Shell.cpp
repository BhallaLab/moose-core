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
bool Shell::keepLooping_ = 1;
unsigned int Shell::numCores_;
unsigned int Shell::numNodes_;
unsigned int Shell::myNode_;
ProcInfo Shell::p_;
unsigned int Shell::numAcks_ = 0;
vector< unsigned int > Shell::acked_( 1, 0 );
bool Shell::doReinit_( 0 );
bool Shell::isParserIdle_( 0 );
double Shell::runtime_( 0.0 );

/*
static SrcFinfo5< string, Id, Id, string, vector< int > > *requestCreate() {
	static SrcFinfo5< string, Id, Id, string, vector< int > > requestCreate( "requestCreate",
			"requestCreate( class, parent, newElm, name, dimensions ): "
			"creates a new Element on all nodes with the specified Id. "
			"Initiates a callback to indicate completion of operation. "
			"Goes to all nodes including self."
			);
	return &requestCreate;
}
*/

/*
static SrcFinfo2< unsigned int, unsigned int >* ack()
{
	static SrcFinfo2< unsigned int, unsigned int > temp( "ack",
			"ack( unsigned int node#, unsigned int status ):"
			"Acknowledges receipt and completion of a command on a worker node."
			"Goes back only to master node."
			);
	return &temp;
}
*/

/*
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
*/

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

/*
static SrcFinfo5< string, ObjId, string, ObjId, string > *requestAddMsg() {
	static SrcFinfo5< string, ObjId, string, ObjId, string > 
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
*/

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
	static DestFinfo handleUseClock( "useClock", 
			"Deals with assignment of path to a given clock."
			" Arguments: path, field, tick number. ",
			new EpFunc4< Shell, string, string, unsigned int, unsigned int >( 
				&Shell::handleUseClock )
			);
	static DestFinfo handleCreate( "create", 
			"create( class, parent, newElm, name, numData, isGlobal )",
			new EpFunc6< Shell, string, ObjId, Id, string, NodeBalance, unsigned int >( &Shell::handleCreate ) );
	static DestFinfo handleDelete( "delete", 
			"Destroys Element, all its messages, and all its children. Args: Id",
			new EpFunc1< Shell, Id >( & Shell::destroy ) );
	static DestFinfo handleAddMsg( "addMsg", 
			"Makes a msg. Arguments are:"
			" msgtype, src object, src field, dest object, dest field",
			new EpFunc6< Shell, string, ObjId, string, ObjId, string, unsigned int >
			( & Shell::handleAddMsg ) );
	static DestFinfo handleQuit( "quit", 
			"Stops simulation running and quits the simulator",
			new OpFunc0< Shell >( & Shell::handleQuit ) );
	static DestFinfo handleMove( "move", 
			"handleMove( Id orig, Id newParent ): "
			"moves an Element to a new parent",
			new EpFunc2< Shell, Id, ObjId >( & Shell::handleMove ) );
	static DestFinfo handleCopy( "copy", 
			"handleCopy( vector< Id > args, string newName, unsigned int nCopies, bool toGlobal, bool copyExtMsgs ): "
			" The vector< Id > has Id orig, Id newParent, Id newElm. " 
			"This function copies an Element and all its children to a new parent."
			" May also expand out the original into nCopies copies."
			" Normally all messages within the copy tree are also copied. "
			" If the flag copyExtMsgs is true, then all msgs going out are also copied.",
			new EpFunc5< Shell, vector< ObjId >, string, unsigned int, bool, bool >( 
				& Shell::handleCopy ) );
	/*
	static Finfo* shellWorker[] = {
		&handleCreate, &handleDelete,
		&handleAddMsg,
		&handleQuit,
		&handleMove, &handleCopy, &handleUseClock,
		&handleSync, &handleReMesh, &handleSetParserIdleFlag,
		ack() };
		*/
	static Finfo* clockControlFinfos[] = 
	{
		requestStart(), requestStep(), requestStop(), requestSetupTick(),
		requestReinit()
	};

		static DestFinfo setclock( "setclock", 
			"Assigns clock ticks. Args: tick#, dt",
			new OpFunc2< Shell, unsigned int, double >( & Shell::doSetClock ) );
		/*
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
		*/
		static SharedFinfo clockControl( "clockControl",
			"Controls the system Clock",
			clockControlFinfos, 
			sizeof( clockControlFinfos ) / sizeof( const Finfo* )
		);
	
	static Finfo* shellFinfos[] = {
		&setclock,
////////////////////////////////////////////////////////////////
//  Shared msg
////////////////////////////////////////////////////////////////
		// &master,
		// &worker,
		&handleCreate,
		&handleDelete,
		&handleCopy,
		&handleMove,
		&handleAddMsg,
		&handleQuit,
		&handleUseClock,
		&clockControl,
	};

	static Dinfo< Shell > d;
	static Cinfo shellCinfo (
		"Shell",
		Neutral::initCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		&d
		//new Dinfo< Shell >()
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
Id Shell::doCreate( string type, ObjId parent, string name, 
				unsigned int numData, 
				NodePolicy nodePolicy,
				unsigned int preferredNode )
{
	// Get the new Id ahead of time and pass to all nodes.
	Id ret = Id::nextId();
	NodeBalance nb( numData, nodePolicy, preferredNode );
	// Get the parent MsgIndex ahead of time and pass to all nodes.
	unsigned int parentMsgIndex = OneToAllMsg::numMsg();
	SetGet6< string, ObjId, Id, string, NodeBalance, unsigned int >::set(
		ObjId(), // Apply command to Shell
		"create",	// Function to call.
		type, 		// class of new object
		parent,		// Parent
		ret,		// id of new object
		name,		// name of new object
		nb,			// Node balance configuration
		parentMsgIndex	// Message index of child-parent msg.
	);
	// innerCreate( type, parent, ret, name, numData, isGlobal );
	return ret;
}

bool Shell::doDelete( Id id )
{
	SetGet1< Id >::set( ObjId(), "delete", id );
	/*
	Neutral n;
	n.destroy( i.eref(), 0 );
	*/
	return true;
}

ObjId Shell::doAddMsg( const string& msgType, 
	ObjId src, const string& srcField, 
	ObjId dest, const string& destField )
{
	if ( !src.id.element() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: src not found" << endl;
		return ObjId();
	}
	if ( !dest.id.element() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: dest not found" << endl;
		return ObjId();
	}
	const Finfo* f1 = src.id.element()->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src: " << src.id.element()->getName() << endl;
		return ObjId();
	}
	const Finfo* f2 = dest.id.element()->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << destField << 
			" on dest: " << dest.id.element()->getName() << endl;
		return ObjId();
	}
	if ( ! f1->checkTarget( f2 ) ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Src/Dest Msg type mismatch: " << srcField << "/" << destField << endl;
		return ObjId();
	}

	const Msg* m = innerAddMsg( msgType, src, srcField, dest, destField, 0 );

	SetGet6< string, ObjId, string, ObjId, string, unsigned int >::set(
		ObjId(), // Apply command to Shell
		"addMsg",	// Function to call.
		msgType,
		src,
		srcField,
		dest,
		destField,
		m->mid().dataId
	);

	return m->mid();

	// const Msg* m = innerAddMsg( msgType, src, srcField, dest, destField );
	// return m->mid();
	// return Msg::lastMsg()->mid();
}

void Shell::doQuit()
{
	SetGet0::set( ObjId(), "quit" );
	// Shell::keepLooping_ = 0;
}

void Shell::doStart( double runtime )
{
	Id clockId( 1 );
	SetGet1< double >::set( clockId, "start", runtime );
}

bool isDoingReinit()
{
	static Id clockId( 1 );
	assert( clockId.element() != 0 );

	return ( reinterpret_cast< const Clock* >( 
		clockId.eref().data() ) )->isDoingReinit();
}

void Shell::doReinit( )
{
	Id clockId( 1 );
	SetGet0::set( clockId, "reinit" );
}

void Shell::doStop( )
{
	Id clockId( 1 );
	SetGet0::set( clockId, "stop" );
}
////////////////////////////////////////////////////////////////////////

void Shell::doSetClock( unsigned int tickNum, double dt )
{
		LookupField< unsigned int, double >::set( ObjId( 1 ), "tickDt", tickNum, dt );
}

void Shell::doUseClock( string path, string field, unsigned int tick )
{
	unsigned int msgIndex = OneToAllMsg::numMsg();
	SetGet4< string, string, unsigned int, unsigned int >::set( ObjId(), 
		"useClock", path, field, tick, msgIndex );
	// innerUseClock( path, field, tick);
}

/**
 * Write given model to SBML file. Returns success value.
 */
int Shell::doWriteSBML( const string& fname, const string& modelpath )
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
Id Shell::doReadSBML( const string& fname, const string& modelpath, const string& solverclass )
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

void Shell::doMove( Id orig, ObjId newParent )
{
	if ( orig == Id() ) {
		cout << "Error: Shell::doMove: Cannot move root Element\n";
		return;
	}

	if ( newParent.element() == 0 ) {
		cout << "Error: Shell::doMove: Cannot move object to null parent \n";
		return;
	}
	if ( Neutral::isDescendant( newParent, orig ) ) {
		cout << "Error: Shell::doMove: Cannot move object to descendant in tree\n";
		return;
		
	}

	SetGet2< Id, ObjId >::set( ObjId(), "move", orig, newParent );
	// innerMove( orig, newParent );
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
	vector< vector< unsigned int > >& index, ObjId cwe )
{
	bool isAbsolute = chopString( path, ret, '/' );
	vector< unsigned int > empty;
	if ( isAbsolute ) {
		index.clear();
	} else {
		index.clear();
	}
	for ( unsigned int i = 0; i < ret.size(); ++i )
	{
		index.push_back( empty );
		if ( ret[i] == "." )
			continue;
		if ( ret[i] == ".." ) {
			continue;
		}
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
	ObjId curr;
	vector< string > names;
	vector< vector< unsigned int > > indices;
	bool isAbsolute = chopPath( path, names, indices, cwe_ );
	assert( names.size() == indices.size() );

	if ( !isAbsolute )
		curr = cwe_;
	
	for ( unsigned int i = 0; i < names.size(); ++i ) {
		if ( names[i] == "." ) {
		} else if ( names[i] == ".." ) {
			curr = Neutral::parent( curr.eref() ).id;
		} else {
			curr = Neutral::child( curr.eref(), names[i] );
			if ( indices[i].size() > 0 )
				curr.dataId = indices[i][0];
			if ( indices[i].size() > 1 )
				curr.fieldIndex = indices[i][1];
		}
	}
	
	assert( curr.element() );
	return curr;
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
	assert( clockId.element() != 0 );

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
	string type, ObjId parent, Id newElm, string name, 
	NodeBalance nb, unsigned int parentMsgIndex )
{
	innerCreate( type, parent, newElm, name, nb, parentMsgIndex );
}



/**
 * Static utility function. Attaches child element to parent element.
 * Must only be called from functions executing in parallel on all nodes,
 * as it does a local message addition
 * MsgIndex is needed to be sure that the same msg identifies parent-child
 * connection on all nodes.
 */
bool Shell::adopt( ObjId parent, Id child, unsigned int msgIndex ) {
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	// static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	// static const FuncId pafid = pf2->getFid();
	static const Finfo* f1 = Neutral::initCinfo()->findFinfo( "childMsg" );

	assert( !( child.element() == 0 ) );
	assert( !( child == Id() ) );
	assert( !( parent.element() == 0 ) );

	Msg* m = new OneToAllMsg( parent.eref(), child.element(), msgIndex );
	assert( m );

	// cout << myNode_ << ", Shell::adopt: mid = " << m->mid() << ", pa =" << parent << "." << parent()->getName() << ", kid=" << child << "." << child()->getName() << "\n";

	if ( !f1->addMsg( pf, m->mid(), parent.element() ) ) {
		cout << "move: Error: unable to add parent->child msg from " <<
			parent.element()->getName() << " to " << 
			child.element()->getName() << "\n";
		return 0;
	}
	return 1;
}

/// Adaptor for above function. Also static function.
bool Shell::adopt( Id parent, Id child, unsigned int msgIndex ) {
	return adopt( ObjId( parent ), child, msgIndex );
}

/**
 * This function actually creates the object. Runs on all nodes.
 */
void Shell::innerCreate( string type, ObjId parent, Id newElm, string name,
	const NodeBalance& nb, unsigned int msgIndex )
	// unsigned int numData, bool isGlobal
{
	const Cinfo* c = Cinfo::find( type );
	if ( c ) {
		Element* pa = parent.element();
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
		Element* ret;
		switch ( nb.policy ) {
			case MooseGlobal:
				ret = new GlobalDataElement( newElm, c, name, nb.numData );
				break;
			case MooseBlockBalance:
				ret = new LocalDataElement( newElm, c, name, nb.numData );
				break;
			case MooseSingleNode:
				cout << "Error: Shell::innerCreate: Yet to implement SingleNodeDataElement\n";
				// ret = new SingleNodeDataElement( newElm, c, name, numData, nb.preferredNode );
				break;
		};
		assert( ret );
		adopt( parent, newElm, msgIndex );
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
	// cout << myNode_ << ": Shell::destroy done for element id: " << eid << ", name = " << eid.element()->getName() << endl;
	n->destroy( eid.eref(), 0 );
	if ( cwe_ == eid )
		cwe_ = Id();

	// ack()->send( e, Shell::myNode(), OkStatus );
}


/**
 * Wrapper function, that does the ack. Other functions also use the
 * inner function to build message trees, so we don't want it to emit
 * multiple acks.
 */
void Shell::handleAddMsg( const Eref& e,
	string msgType, ObjId src, string srcField, 
	ObjId dest, string destField, unsigned int msgIndex )
{
		// Node 0 will have already called innerAddMsg to get the msgIndex
	if ( myNode() != 0 )
		innerAddMsg( msgType, src, srcField, dest, destField, msgIndex );
	/*
	if ( innerAddMsg( msgType, src, srcField, dest, destField ) )
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0), Shell::myNode(), ErrorStatus );
	*/
}

/**
 * The actual function that adds messages. Does NOT send an ack.
 * The msgIndex specifies the index on which to place this message. If the
 * value is zero it does an automatic placement.
 */
const Msg* Shell::innerAddMsg( string msgType,
	ObjId src, string srcField, 
	ObjId dest, string destField, unsigned int msgIndex )
{
	/*
	cout << myNode_ << ", Shell::handleAddMsg: " << 
		msgType << ", " << mid <<
		", src =" << src << "." << srcField << 
		", dest =" << dest << "." << destField << "\n";
		*/
	const Finfo* f1 = src.id.element()->cinfo()->findFinfo( srcField );
	if ( f1 == 0 ) return 0;
	const Finfo* f2 = dest.id.element()->cinfo()->findFinfo( destField );
	if ( f2 == 0 ) return 0;
	
	// Should have been done before msgs request went out.
	assert( f1->checkTarget( f2 ) );

	Msg *m = 0;
	if ( msgType == "diagonal" || msgType == "Diagonal" ) {
		m = new DiagonalMsg( src.id.element(), dest.id.element(), 
						msgIndex );
	} else if ( msgType == "sparse" || msgType == "Sparse" ) {
		m = new SparseMsg( src.id.element(), dest.id.element(), msgIndex );
	} else if ( msgType == "Single" || msgType == "single" ) {
		m = new SingleMsg( src.eref(), dest.eref(), msgIndex );
	} else if ( msgType == "OneToAll" || msgType == "oneToAll" ) {
		m = new OneToAllMsg( src.eref(), dest.id.element(), msgIndex );
	} else if ( msgType == "AllToOne" || msgType == "allToOne" ) {
		m = new OneToAllMsg( dest.eref(), src.id.element(), msgIndex ); // Little hack.
	} else if ( msgType == "OneToOne" || msgType == "oneToOne" ) {
		m = new OneToOneMsg( src.id.element(), dest.id.element(), msgIndex );
	} else {
		cout << myNode_ << 
			": Error: Shell::handleAddMsg: msgType not known: "
			<< msgType << endl;
		return m;
	}
	if ( m ) {
		if ( f1->addMsg( f2, m->mid(), src.id.element() ) ) {
			return m;
		}
		delete m;
		m = 0;
	}
	cout << myNode_ << 
			": Error: Shell::handleAddMsg: Unable to make/connect Msg: "
			<< msgType << " from " << src.id.element()->getName() <<
			" to " << dest.id.element()->getName() << endl;
	return m;
}

bool Shell::innerMove( Id orig, ObjId newParent )
{
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* f1 = Neutral::initCinfo()->findFinfo( "childMsg" );

	assert( !( orig == Id() ) );
	assert( !( newParent.element() == 0 ) );

	ObjId mid = orig.element()->findCaller( pafid );
	Msg::deleteMsg( mid );

	Msg* m = new OneToAllMsg( newParent.eref(), orig.element(), 0 );
	assert( m );
	if ( !f1->addMsg( pf, m->mid(), newParent.element() ) ) {
		cout << "move: Error: unable to add parent->child msg from " <<
			newParent.element()->getName() << " to " << 
			orig.element()->getName() << "\n";
		return 0;
	}
	return 1;
}

void Shell::handleMove( const Eref& e, Id orig, ObjId newParent )
{
	
	innerMove( orig, newParent );
	/*
	if ( innerMove( orig, newParent ) )
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0 ), Shell::myNode(), ErrorStatus );
		*/
}

void Shell::addClockMsgs( 
	const vector< Id >& list, const string& field, unsigned int tick,
   	unsigned int msgIndex	)
{
	if ( !Id( 1 ).element() )
		return;
	ObjId clockId( 1 );
	for ( vector< Id >::const_iterator i = list.begin(); 
		i != list.end(); ++i ) {
		if ( i->element() ) {
			stringstream ss;
			ss << "proc" << tick;
			innerAddMsg( "OneToAll",
				clockId, ss.str(), 
				ObjId( *i, 0 ), field, msgIndex++ );
		}
	}
}

bool Shell::innerUseClock( string path, string field, unsigned int tick,
				unsigned int msgIndex )
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
	
	addClockMsgs( list, field, tick, msgIndex );
	return 1;
}

void Shell::handleUseClock( const Eref& e, 
	string path, string field, unsigned int tick, unsigned int msgIndex )
{
	// cout << q->getProcInfo()->threadIndexInGroup << ": in Shell::handleUseClock with path " << path << endl << flush;
	innerUseClock( path, field, tick, msgIndex );
	/*
	if ( innerUseClock( path, field, tick ) )
		ack()->send( Eref( shelle_, 0 ), 
			Shell::myNode(), OkStatus );
	else
		ack()->send( Eref( shelle_, 0 ), 
			Shell::myNode(), ErrorStatus );
			*/
}

void Shell::handleQuit()
{
	Shell::keepLooping_ = 0;
}

// Static function
bool Shell::keepLooping()
{
	return keepLooping_;
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

////////////////////////////////////////////////////////////////////////
// Some static utility functions
////////////////////////////////////////////////////////////////////////

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
