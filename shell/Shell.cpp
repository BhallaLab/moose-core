/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgManager.h"
#include "SingleMsg.h"
#include "DiagonalMsg.h"
#include "OneToOneMsg.h"
#include "OneToAllMsg.h"
#include "AssignmentMsg.h"
#include "AssignVecMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "Shell.h"
#include "Dinfo.h"

// Want to separate out this search path into the Makefile options
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

const unsigned int Shell::OkStatus = ~0;
const unsigned int Shell::ErrorStatus = ~1;
unsigned int Shell::numCores_;
unsigned int Shell::numNodes_;
unsigned int Shell::myNode_;
ProcInfo Shell::p_;

static SrcFinfo5< string, Id, Id, string, vector< unsigned int > > requestCreate( "requestCreate",
			"requestCreate( class, parent, newElm, name, dimensions ): "
			"creates a new Element on all nodes with the specified Id. "
			"Initiates a callback to indicate completion of operation. "
			"Goes to all nodes including self."
			);

static SrcFinfo2< unsigned int, unsigned int > ack( "ack",
			"ack( unsigned int node# ):"
			"Acknowledges receipt and completion of a command on a worker node."
			"Goes back only to master node."
			);

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
static SrcFinfo5< string, FullId, string, FullId, string > requestAddMsg( 
			"requestAddMsg",
			"requestAddMsg( type, src, srcField, dest, destField );"
			"Creates specified Msg between specified Element on all nodes."
			"Initiates a callback to indicate completion of operation."
			"Goes to all nodes including self."
			); 
static SrcFinfo4< Id, DataId, FuncId, PrepackedBuffer > requestSet(
			"requestSet",
			"requestSet( tgtId, tgtDataId, tgtFieldId, value ):"
			"Assigns a value on target field."
			);

static DestFinfo create( "create", 
			"create( class, parent, newElm, name, dimensions )",
			new EpFunc5< Shell, string, Id, Id, string, vector< unsigned int > >( &Shell::create ) );

static DestFinfo del( "delete", 
			"Destroys Element, all its messages, and all its children. Args: Id",
			new EpFunc1< Shell, Id >( & Shell::destroy ) );

static DestFinfo handleAck( "handleAck", 
			"Keeps track of # of acks to a blocking shell command. Arg: Source node num.",
			new OpFunc2< Shell, unsigned int, unsigned int >( 
				& Shell::handleAck ) );

static DestFinfo handleQuit( "handleQuit", 
			"quit(): Quits simulation.",
			new OpFunc0< Shell >( & Shell::handleQuit ) );

static DestFinfo handleStart( "start", 
			"Starts off a simulation for the specified run time, automatically partitioning among threads if the settings are right",
			new OpFunc1< Shell, double >( & Shell::handleStart ) );
			

static DestFinfo handleAddMsg( "handleAddMsg", 
			"Makes a msg",
			new OpFunc5< Shell, string, FullId, string, FullId, string >
				( & Shell::handleAddMsg ) );

static DestFinfo handleSet( "handleSet", 
			"Deals with request, to set specified field on any node to a value.",
			new OpFunc4< Shell, Id, DataId, FuncId, PrepackedBuffer >( 
				&Shell::handleSet )
			);

static SrcFinfo0 lowLevelSet(
			"lowLevelSet",
			"lowlevelSet():"
			"Low-level SrcFinfo. Not for external use, internally used as"
			"a handle to assign a value on target field."
);


/**
 * Sequence is:
 * innerDispatchGet->requestGet->handleGet->lowLevelGet->get_field->
 * 	receiveGet->completeGet
 */

static SrcFinfo3< Id, DataId, FuncId > requestGet( "requestGet",
			"Function to request another Element for a value" );

static DestFinfo handleGet( "handleGet", 
			"handleGet( Id elementId, DataId index, FuncId fid )"
			"Deals with requestGet, to get specified field from any node.",
			new OpFunc3< Shell, Id, DataId, FuncId >( 
				&Shell::handleGet )
			);
static SrcFinfo1< FuncId > lowLevelGet(
			"lowLevelGet",
			"lowlevelGet():"
			"Low-level SrcFinfo. Not for external use, internally used as"
			"a handle to request a value from target field."
);

// This function is called by directly inserting entries into the queue,
// when getting a value.
static DestFinfo lowLevelReceiveGet( "lowLevelReceiveGet", 
	"lowLevelReceiveGet( PrepackedBuffer data )"
	"Function on worker node Shell to handle the value returned by object.",
	new OpFunc1< Shell, PrepackedBuffer >( &Shell::lowLevelRecvGet )
);

static DestFinfo receiveGet( "receiveGet", 
	"receiveGet( Uint node#, Uint status, PrepackedBuffer data )"
	"Function on master shell that handles the value relayed from worker.",
	new OpFunc3< Shell, unsigned int, unsigned int, PrepackedBuffer >( &Shell::recvGet )
);

static SrcFinfo3< unsigned int, unsigned int, PrepackedBuffer > relayGet(
	"relayGet",
	"relayGet( node, status, data ): Passes 'get' data back to master node"
);
/*
static SrcFinfo3< unsigned int, double, unsigned int > requestSetClock(
			"requestSetClock",
			"requestSetClock( tickNum, dt, unsigned int stage )"
		);

static DestFinfo handleSetClock( "handleSetClock", 
			"handleSetClock( unsigned int tickNum, double dt, unsigned int stage )",
			new OpFunc3< Shell, unsigned int, double, unsigned int >(
				&shell::handleSetClock )
			);
			*/

static Finfo* shellMaster[] = {
	&requestCreate, &requestDelete, &requestQuit, &requestStart,
	&requestAddMsg, &requestSet, &requestGet, &receiveGet,
	&handleAck };
static Finfo* shellWorker[] = {
	&create, &del, &handleQuit, &handleStart, 
		&handleAddMsg, &handleSet, &handleGet, &relayGet,
	&ack };


const Cinfo* Shell::initCinfo()
{
////////////////////////////////////////////////////////////////
// Value Finfos
////////////////////////////////////////////////////////////////
	static ValueFinfo< Shell, string > name( 
			"name",
			"Name of object", 
			&Shell::setName, 
			&Shell::getName );

	static ValueFinfo< Shell, bool > quit( 
			"quit",
			"Flag to tell the system to quit", 
			&Shell::setQuit, 
			&Shell::getQuit );

////////////////////////////////////////////////////////////////
// Dest Finfos: Functions handled by Shell
////////////////////////////////////////////////////////////////
		static DestFinfo setclock( "setclock", 
			"Assigns clock ticks. Args: tick#, dt, stage",
			new OpFunc3< Shell, unsigned int, double, unsigned int >( & Shell::setclock ) );
		static DestFinfo loadBalance( "loadBalance", 
			"Set up load balancing",
			new OpFunc0< Shell >( & Shell::loadBalance ) );

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

	
	static Finfo* shellFinfos[] = {
		&name,
		&quit,
		&lowLevelReceiveGet,
		&setclock,
		&loadBalance,

////////////////////////////////////////////////////////////////
//  Predefined Msg Src and MsgDests.
////////////////////////////////////////////////////////////////

		&requestGet,
		// &requestSet,
		&lowLevelSet,
		&lowLevelGet,
////////////////////////////////////////////////////////////////
//  Shared msg
////////////////////////////////////////////////////////////////
		&master,
		&worker,
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
	: name_( "" ),
		quit_( 0 ), 
		isSingleThreaded_( 0 ),
		numAcks_( 0 ),
		barrier1_( 0 ),
		barrier2_( 0 ),
		isRunning_( 0 ),
		runtime_( 0.0 )
{
	;
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
Id Shell::doCreate( string type, Id parent, string name, vector< unsigned int > dimensions )
{
	Id ret = Id::nextId();
	initAck(); // Always put the init before the request.
	// Here we would do the 'send' on an internode msg to do the actual
	// Create.
	requestCreate.send( Id().eref(), &p_, type, parent, ret, name, dimensions );
	// innerCreate( type, parent, ret, name );
	// cout << myNode_ << ": Shell::doCreate: request sent\n";

	// Now we wait till all nodes are done.
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
	
	// Here we might choose to check if success on all nodes.
	// cout << myNode_ << ": Shell::doCreate: ack received\n";
	
	return ret;
}

bool Shell::doDelete( Id i )
{
	initAck();
	requestDelete.send( Id().eref(), &p_, i );
	// cout << myNode_ << ": Shell::doDelete: request sent\n";
	// Now we wait till all nodes are done.
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
	// cout << myNode_ << ": Shell::doDelete: ack received\n";

	return 1;
}

MsgId Shell::doAddMsg( const string& msgType, 
	FullId src, const string& srcField, 
	FullId dest, const string& destField )
{
	if ( !src.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: src not found\n";
		return Msg::badMsg;
	}
	if ( !dest.id() ) {
		cout << myNode_ << ": Error: Shell::doAddMsg: dest not found\n";
		return Msg::badMsg;
	}
	const Finfo* f1 = src.id()->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src: " << src.id()->getName() << "\n";
		return Msg::badMsg;
	}
	const Finfo* f2 = dest.id()->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << destField << 
			" on dest: " << dest.id()->getName() << "\n";
		return Msg::badMsg;
	}
	if ( ! f1->checkTarget( f2 ) ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Src/Dest Msg type mismatch: " << srcField << "/" << destField << endl;
		return Msg::badMsg;
	}

	initAck();
	requestAddMsg.send( Eref( shelle_, 0 ), &p_, 
		msgType, src, srcField, dest, destField );
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );

	return latestMsgId_;
}

/**
 * Static function, sets up the master message that connects
 * all shells on all nodes to each other. Uses low-level calls to
 * do so.
 */
void Shell::connectMasterMsg()
{
	Id shellId;
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
		m = new OneToOneMsg( shelle, shelle );
	if ( m ) {
		if ( f1->addMsg( f2, m->mid(), shelle ) ) {
			return;
		} else {
			cout << "Error: failed in Shell::connectMasterMsg()\n";
			delete m; // Nasty, but rare.
		}
	}
	exit( 0 ); // Bad!
}

void Shell::doQuit( )
{
	requestQuit.send( Id().eref(), &p_, 1 );
	// cout << myNode_ << ": Shell::doQuit: request sent\n";
	while ( !quit_ )
		Qinfo::mpiClearQ( &p_ );
//	Qinfo::mpiClearQ( &p_ );
	// cout << myNode_ << ": Shell::doQuit: quitting\n";
}

void Shell::doStart( double runtime )
{
	initAck();
	Eref sheller( shelle_, 0 );
	requestStart.send( sheller, &p_, runtime, 1 );
	// cout << myNode_ << ": Shell::doStart: request sent\n";
	while ( isAckPending() ) {
		Qinfo::mpiClearQ( &p_ );
		process( &p_, sheller );
	}
	// cout << Shell::myNode() << ": Shell::doStart(" << runtime << ")" << endl;
	// Qinfo::reportQ();
	// cout << myNode_ << ": Shell::doStart: quitting\n";
}

////////////////////////////////////////////////////////////////
// DestFuncs
////////////////////////////////////////////////////////////////

void Shell::process( const ProcInfo* p, const Eref& e )
{
	if ( isRunning_ ) {
		start( runtime_ ); // This is a blocking call
		ack.send( Eref( shelle_, 0 ), &p_, myNode_, OkStatus, 0 );
	}
}


void Shell::setName( string name )
{
	name_ = name;
}

string Shell::getName() const
{
	return name_;
}

void Shell::setQuit( bool val )
{
	quit_ = val;
}

bool Shell::getQuit() const
{
	return quit_;
}

void Shell::handleQuit()
{
	quit_ = 1;
}

const char* Shell::getBuf() const
{
	if ( getBuf_.size() > 0 )
		return &( getBuf_[0] );
	return 0;
}

void Shell::handleStart( double runtime )
{
	isRunning_ = 1;
	runtime_ = runtime;
	// start( runtime );
	// ack.send( Eref( shelle_, 0 ), &p_, myNode_, OkStatus, 0 );
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
void Shell::create( Eref e, const Qinfo* q, 
	string type, Id parent, Id newElm, string name,
	vector< unsigned int > dimensions )
{
	// cout << myNode_ << ": In Shell::create for element " << name << " id " << newElm << ", dim = " << dimensions[0] << endl;
	innerCreate( type, parent, newElm, name, dimensions );
	// cout << myNode_ << ": Shell::create inner Create done for element " << name << " id " << newElm << endl;
//	if ( myNode_ != 0 )
	ack.send( e, &p_, Shell::myNode(), OkStatus, 0 );
	// cout << myNode_ << ": Shell::create ack sent" << endl;
}

/**
 * This function actually creates the object.
 */
void Shell::innerCreate( string type, Id parent, Id newElm, string name,
	const vector< unsigned int >& dimensions )
{
	const Cinfo* c = Cinfo::find( type );
	if ( c ) {
		Element* pa = parent();
		if ( !pa ) {
			stringstream ss;
			ss << "create: Parent Element'" << parent << "' not found. No Element created";
			return;
		}
		// cout << myNode_ << ": Shell::innerCreate newElmId= " << newElm << endl;
	//	Element* ret = 
		new Element( newElm, c, name, dimensions );
		//ret = c->create( newElm, name, n, Element::Decomposition::Block );
	} else {
		stringstream ss;
		ss << "create: Class '" << type << "' not known. No Element created";
		warning( ss.str() );
	}
}

void Shell::destroy( Eref e, const Qinfo* q, Id eid)
{
	eid.destroy();
	// cout << myNode_ << ": Shell::destroy done for element id " << eid << endl;

	//if ( myNode_ != 0 )
		ack.send( e, &p_, Shell::myNode(), OkStatus, 0 );
}


// I really also want to put in a message type. But each message has its
// own features and these may well be done separately
void Shell::handleAddMsg( string msgType, FullId src, string srcField, 
	FullId dest, string destField )
{
	// cout << myNode_ << ", Shell::handleAddMsg" << "\n";
	const Finfo* f1 = src.id()->cinfo()->findFinfo( srcField );
	assert( f1 != 0 );
	const Finfo* f2 = dest.id()->cinfo()->findFinfo( destField );
	assert( f2 != 0 );
	
	// Should have been done before msgs request went out.
	assert( f1->checkTarget( f2 ) );

	latestMsgId_ = Msg::badMsg;

	Msg *m = 0;
	if ( msgType == "diagonal" || msgType == "Diagonal" ) {
		m = new DiagonalMsg( src.id(), dest.id() );
	} else if ( msgType == "sparse" || msgType == "Sparse" ) {
		m = new SparseMsg( src.id(), dest.id() );
	} else if ( msgType == "Single" || msgType == "single" ) {
		m = new SingleMsg( src.eref(), dest.eref() );
	} else if ( msgType == "OneToAll" || msgType == "oneToAll" ) {
		m = new OneToAllMsg( src.eref(), dest.id() );
	} else if ( msgType == "OneToOne" || msgType == "oneToOne" ) {
		m = new OneToOneMsg( src.id(), dest.id() );
	} else {
		cout << myNode_ << 
			": Error: Shell::handleAddMsg: msgType not known: "
			<< msgType << endl;
		ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus, 0 );
		return;
	}
	if ( m ) {
		if ( f1->addMsg( f2, m->mid(), src.id() ) ) {
			latestMsgId_ = m->mid();
			ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus, 0 );
			return;
		}
		delete m;
	}
	cout << myNode_ << 
			": Error: Shell::handleAddMsg: Unable to make/connect Msg: "
			<< msgType << " from " << src.id()->getName() <<
			" to " << dest.id()->getName() << endl;
	ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus, 0 );
}

void Shell::warning( const string& text )
{
	cout << "Warning: Shell:: " << text << endl;
}

void Shell::error( const string& text )
{
	cout << "Error: Shell:: " << text << endl;
}

///////////////////////////////////////////////////////////////////////////
// Functions for handling acks for blocking Shell function calls.
///////////////////////////////////////////////////////////////////////////

/**
 * Initialize acks. This call should be done before the 'send' goes out,
 * because with the wonders of threading we might get a response to the
 * 'send' before this call is executed.
 */
void Shell::initAck()
{
	acked_.assign( numNodes_, 0 );
	numAcks_ = 0;
	// Could put in timeout check here.
}

/**
 * Generic handler for ack msgs from various nodes. Keeps track of
 * which nodes have responded.
 */
void Shell::handleAck( unsigned int ackNode, unsigned int status )
{
	assert( ackNode <= numNodes_ );
	acked_[ ackNode ] = status;
		// Here we could also check which node(s) are last, in order to do
		// some dynamic load balancing.
	++numAcks_;
	if ( status != OkStatus ) {
		cout << myNode_ << ": Shell::handleAck: Error: status = " <<
			status << " from node " << ackNode << endl;
	}
}

/**
 * Test for receipt of acks from all nodes
 */ 
bool Shell::isAckPending() const
{
	// Put in timeout check here. At this point we would inspect the
	// acked vector to see which is last.
	return ( numAcks_ < numNodes_ );
}

////////////////////////////////////////////////////////////////////////
// Some static utility functions
////////////////////////////////////////////////////////////////////////

// Statid func for returning the pet ProcInfo of the shell.
const ProcInfo* Shell::procInfo()
{
	return &p_;
}

/**
 * Static global, returns contents of shell buffer.
 */
const char* Shell::buf() 
{
	static Id shellid;
	static Element* shell = shellid();
	assert( shell );
	return (reinterpret_cast< Shell* >(shell->dataHandler()->data( 0 )) )->getBuf();
}

////////////////////////////////////////////////////////////////////////

void Shell::setclock( unsigned int tickNum, double dt, unsigned int stage )
{
	Eref ce = Id( 1 ).eref();
	SetGet3< unsigned int, double, unsigned int >::set( ce, "setupTick",
		tickNum, dt, stage );
}

////////////////////////////////////////////////////////////////////////
// Functions for handling field set/get and func calls
////////////////////////////////////////////////////////////////////////

void Shell::innerSetVec( const Eref& er, FuncId fid, const PrepackedBuffer& arg )
{
	shelle_->clearBinding ( lowLevelGet.getBindIndex() );
	Msg* m = new AssignVecMsg( Eref( shelle_, 0 ), er.element(), Msg::setMsg );
	shelle_->addMsgAndFunc( m->mid(), fid, lowLevelGet.getBindIndex() );
	char* temp = new char[ arg.size() ];
	arg.conv2buf( temp );

	Qinfo q( fid, 0, arg.size() );
	shelle_->asend( q, lowLevelGet.getBindIndex(), &p_, temp );

	delete[] temp;
}

void Shell::innerSet( const Eref& er, FuncId fid, const char* args, 
	unsigned int size )
{
	if ( er.isDataHere() ) {
		shelle_->clearBinding ( lowLevelGet.getBindIndex() );
		Msg* m = new AssignmentMsg( Eref( shelle_, 0 ), er, Msg::setMsg );
		shelle_->addMsgAndFunc( m->mid(), fid, lowLevelGet.getBindIndex() );
	
		Qinfo q( fid, 0, size );
		shelle_->asend( q, lowLevelGet.getBindIndex(), &p_, args );
	}
}

void Shell::handleSet( Id id, DataId d, FuncId fid, PrepackedBuffer arg )
{
	Eref er( id(), d );
	if ( arg.isVector() ) {
		innerSetVec( er, fid, arg );
	} else {
		innerSet( er, fid, arg.data(), arg.dataSize() );
	}
	ack.send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus, 0 );
	// We assume that the ack will get back to the master node no sooner
	// than the field assignment. This is probably pretty safe. More to the
	// point, the Parser thread won't be able to do anything else before
	// the field assignment is done.
}

/*
void Shell::handleSetAck()
{
	ack.send( e, &p_, OkStatus, 0 );
}
*/

// Static function, used for developer-code triggered SetGet functions.
// Should only be issued from master node.
// This is a blocking function, and returns only when the job is done.
// mode = 0 is single value set, mode = 1 is vector set, mode = 2 is
// to set the entire target array to a single value.
void Shell::dispatchSet( const Eref& tgt, FuncId fid, const char* args,
	unsigned int size )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	PrepackedBuffer buf( args, size );
	s->innerDispatchSet( sheller, tgt, fid, buf );
}

// regular function, does the actual dispatching.
void Shell::innerDispatchSet( Eref& sheller, const Eref& tgt, 
	FuncId fid, const PrepackedBuffer& buf )
{
	Id tgtId( tgt.element()->id() );
	initAck();
	requestSet.send( sheller, &p_,  tgtId, tgt.index(), fid, buf );
	// requestSetAck.send( sheller, &p_, 1 );

	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
}

// Static function.
void Shell::dispatchSetVec( const Eref& tgt, FuncId fid, 
	const PrepackedBuffer& pb )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->innerDispatchSet( sheller, tgt, fid, pb );
}

/**
 * Returns buffer containing desired data.
 * Static function, used for developer-code triggered SetGet fucntions.
 * Should only be issued from master node.
 * This is a blocking function and returns only when the job is done.
 */
const char* Shell::dispatchGet( const Eref& tgt, const string& field, 
	const SetGet* sg )
{
	string getField = "get_" + field;
	const Finfo* gf = tgt.element()->cinfo()->findFinfo( getField );
	const DestFinfo * df = dynamic_cast< const DestFinfo* >( gf );
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	if ( !df ) {
		cout << s->myNode() << ": Error: Shell::dispatchGet: field '" << field << "' not found on " << tgt << endl;
		return 0;
	}

	if ( df->getOpFunc()->checkSet( sg ) ) { // Type validation
		Eref sheller = Id().eref();
		return s->innerDispatchGet( sheller, tgt, df->getFid() );
	} else {
		cout << s->myNode() << ": Error: Shell::dispatchGet: type mismatch for field " << field << " on " << tgt << endl;
	}
	return 0;
}

/**
 * Tells all nodes to dig up specified field, if object is present on node.
 * Not thread safe: this should only run on master node.
 */
const char* Shell::innerDispatchGet( const Eref& sheller, const Eref& tgt, 
	FuncId fid )
{
	initAck();
	requestGet.send( sheller, &p_, tgt.element()->id(), tgt.index(), fid );

	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
	
	return &getBuf_[0];
}


/**
 * This operates on the worker node. It handles the Get request from
 * the master node, and dispatches if need to the local object.
 */
void Shell::handleGet( Id id, DataId index, FuncId fid )
{
	Eref sheller( shelle_, 0 );
	Eref tgt( id(), index );
	if ( id()->dataHandler()->isDataHere( index ) ) {
		shelle_->clearBinding( lowLevelGet.getBindIndex() );
		Msg* m = new AssignmentMsg( sheller, tgt, Msg::setMsg );
		shelle_->addMsgAndFunc( m->mid(), fid, lowLevelGet.getBindIndex() );
		FuncId retFunc = lowLevelReceiveGet.getFid();
		lowLevelGet.send( sheller, &p_, retFunc );
	} else {
		ack.send( sheller, &p_, myNode_, OkStatus, 0 );
	}
}

void Shell::recvGet( 
	unsigned int node, unsigned int status, PrepackedBuffer pb )
{
	if ( myNode_ == 0 ) {
		getBuf_.resize( pb.dataSize() );
		memcpy( &getBuf_[0], pb.data(), pb.dataSize() );
		handleAck( node, status );
	} else {
		// cout << myNode_ << ": Error: Shell::recvGet: should never be called except on node 0\n";
	}
}

void Shell::lowLevelRecvGet( PrepackedBuffer pb )
{
	relayGet.send( Eref( shelle_, 0 ), &p_, myNode(), OkStatus, pb );
}

////////////////////////////////////////////////////////////////////////
