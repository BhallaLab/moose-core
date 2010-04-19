/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DiagonalMsg.h"
#include "AssignmentMsg.h"
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
static SrcFinfo5< vector< unsigned int >, string, string, string,
	vector< double > > requestAddMsg( "requestAddMsg",
			"requestAddMsg( ids, field1, field2, msgtype, args );"
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
			new OpFunc1< Shell, double >( & Shell::start ) );
			

static DestFinfo handleAddMsg( "handleAddMsg", 
			"Makes a msg",
			new EpFunc5< Shell, 
				vector< unsigned int >, string, string, string,
					vector< double >
				>( & Shell::handleAddMsg ) );

static DestFinfo handleSet( "handleSet", 
			"Deals with request, to set specified field on any node to a value.",
			new OpFunc4< Shell, Id, DataId, FuncId, PrepackedBuffer >( 
				&Shell::handleSet )
			);


static Finfo* shellMaster[] = {
	&requestCreate, &requestDelete, &requestQuit, &requestStart,
	&requestAddMsg, &handleAck };
static Finfo* shellWorker[] = {
	&create, &del, &handleQuit, &handleStart, &handleAddMsg, &ack };

static SrcFinfo1< FuncId > requestGet( "requestGet",
			"Function to request another Element for a value" );

/*
static SrcFinfo1< PrepackedBuffer > setField( "setField",
			"set( PrepackagedBuffer ):"
			"Assigns a value on target field."
			"This is a special SrcFinfo, as it uses preconverted arguments"
			"to communicate with the target."
			);
			*/

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
		static DestFinfo handleGet( "handleGet", 
			"Function to handle returning values for 'get' calls.",
			new RetFunc< Shell >( &Shell::handleGet ) );
		static DestFinfo setclock( "setclock", 
			"Assigns clock ticks. Args: tick#, dt, stage",
			new OpFunc3< Shell, unsigned int, double, unsigned int >( & Shell::setclock ) );
		static DestFinfo loadBalance( "loadBalance", 
			"Set up load balancing",
			new OpFunc0< Shell >( & Shell::loadBalance ) );

		/*
		static DestFinfo( "create", 
			"create( class, parent, newElm, name",
			new EpFunc4< Shell, string, Id, Id, string>( &Shell::create )),
		static DestFinfo( "delete", 
			"Destroys Element, all its messages, and all its children. Args: Id",
			new EpFunc1< Shell, Id >( & Shell::destroy ) ),

		new DestFinfo( "addmsg", 
			"Adds a Msg between specified Elements. Args: Src, Dest, srcField, destField",
			new OpFunc4< Shell, Id, Id, string, string >( & Shell::addmsg ) ),
			*/

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
		&handleGet,
		&handleStart,
		&setclock,
		&loadBalance,

////////////////////////////////////////////////////////////////
//  Predefined Msg Src and MsgDests.
////////////////////////////////////////////////////////////////

		&requestGet,
		&requestSet,
////////////////////////////////////////////////////////////////
//  Shared msg
////////////////////////////////////////////////////////////////
		&master,
		&worker,
	};

	static Cinfo shellCinfo (
		"Shell",
		0, // No base class. Make it neutral soon.
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
		isRunning_( 0 )
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

MsgId Shell::doAddMsg( Id src, const string& srcField, Id dest,
	const string& destField, const string& msgType, vector< double > args )
{
	const Finfo* f1 = src()->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << myNode_ << ": Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src:\n"; // Put name here.
		return 0;
	}
	const Finfo* f2 = dest()->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << "Shell::doAddMsg: Error: Failed to find field " << srcField << 
			" on src:\n"; // Put name here.
		return 0;
	}
	unsigned int mid = 0;
	vector< unsigned int > ids;
	ids.push_back( src.value() );
	ids.push_back( dest.value() );
	ids.push_back( mid );

	initAck();
	requestAddMsg.send( Id().eref(), &p_, ids, 
		srcField, destField, msgType, args );

	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );

	return Msg::badMsg; // Shouldn't we return the msg data?
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
		if ( f1->addMsg( f2, m->mid(), shelle ) )
			return;
		else
			delete m; // Nasty, but rare.
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
	requestStart.send( Id().eref(), &p_, runtime, 1 );
	// cout << myNode_ << ": Shell::doStart: request sent\n";
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
	// cout << myNode_ << ": Shell::doStart: quitting\n";
}

/*
void Shell::doSetDouble( Id id, DataId d, string field, double value )
{
	initAck();
	requestSet.send( Id().eref(), &p_, id, d, field, value );
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
}

double Shell::doGetDouble( Id id, DataId d, string field )
{
	initAck();
	requestSet.send( Id().eref(), &p_, field, value );
	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );


	return 0.0;
}
*/

////////////////////////////////////////////////////////////////
// DestFuncs
////////////////////////////////////////////////////////////////

void Shell::process( const ProcInfo* p, const Eref& e )
{
	;
	// quit_ = 0;
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

/*
void Shell::handleGet2( Eref e, const Qinfo* q, const char* arg )
{
	getBuf_.resize( q->size() );
	memcpy( &getBuf_[0], arg, q->size() );
	// Instead of deleting and recreating the msg, it could be a 
	// permanent msg on this object, reaching out whenever needed
	// to targets.
}
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
*/

void Shell::handleGet( Eref e, const Qinfo* q, const char* arg )
{
	getBuf_.resize( q->size() );
	memcpy( &getBuf_[0], arg, q->size() );
	// Instead of deleting and recreating the msg, it could be a 
	// permanent msg on this object, reaching out whenever needed
	// to targets.
}

const char* Shell::getBuf() const
{
	if ( getBuf_.size() > 0 )
		return &( getBuf_[0] );
	return 0;
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
void Shell::handleAddMsg( Eref e, const Qinfo* q,
		vector< unsigned int > ids, string srcField,
		string destField, string msgType, vector< double > args )
{
	cout << myNode_ << ", Shell::handleAddMsg << \n";
	Id e1( ids[0] );
	Id e2( ids[1] );
	// Could actually make a required static function in all the Msgs,
	// that has the arguments srcId, srcField, destId, destField, 
	// vector< double > args. 
	if ( !e1() ) {
		cout << myNode_ << ": Error: Shell::handleAddMsg: e1 not found\n";
		ack.send( e, &p_, ErrorStatus, 0 );
		return;
	}
	if ( !e2() ) {
		cout << myNode_ << ": Error: Shell::handleAddMsg: e2 not found\n";
		ack.send( e, &p_, ErrorStatus, 0 );
		return;
	}

	if ( msgType == "diagonal" || msgType == "Diagonal" ) {
		if ( args.size() != 1 ) {
			cout << myNode_ << ": Error: Shell::handleAddMsg: Should have 1 arg, was " << args.size() << endl;
			ack.send( e, &p_, ErrorStatus, 0 );
			return;
		}
		int stride = args[0];
		bool ret = 
			DiagonalMsg::add( e1(), srcField, e2(), destField, stride );
		if ( ret ) {
			ack.send( e, &p_, OkStatus, 0 );
			return;
		}
	}
	cout << myNode_ << ": Error: Shell::handleAddMsg: msgType not known: "
		<< msgType << endl;
	ack.send( e, &p_, ErrorStatus, 0 );
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

void Shell::setclock( unsigned int tickNum, double dt, unsigned int stage )
{
	Eref ce = Id( 1 ).eref();
	SetGet3< unsigned int, double, unsigned int >::set( ce, "setupTick",
		tickNum, dt, stage );
}

////////////////////////////////////////////////////////////////////////
// Functions for handling field set/get and func calls
////////////////////////////////////////////////////////////////////////

void Shell::innerSet( const Eref& er, FuncId fid, const char* args, 
	unsigned int size )
{
	shelle_->clearBinding ( requestSet.getBindIndex() );
	Msg* m = new AssignmentMsg( Eref( shelle_, 0 ), er, Msg::setMsg );
	shelle_->addMsgAndFunc( m->mid(), fid, requestSet.getBindIndex() );

	Qinfo q( fid, 0, size );
	shelle_->asend( q, requestSet.getBindIndex(), &p_, args );
}

void Shell::handleSet( Id id, DataId d, FuncId fid, PrepackedBuffer arg)
{
	Eref er( id(), d );
	innerSet( er, fid, arg.data(), arg.size() );
	ack.send( Eref( shelle_, 0 ), &p_, OkStatus, 0 );
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
void Shell::dispatchSet( const Eref& tgt, FuncId fid, const char* args,
	unsigned int size )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->innerDispatchSet( sheller, tgt, fid, args, size );
}

// regular function, does the actual dispatching.
void Shell::innerDispatchSet( Eref& sheller, const Eref& tgt, 
	FuncId fid, const char* args, unsigned int size )
{
	PrepackedBuffer buf( args, size );
	Id tgtId( tgt.element()->id() );
	initAck();
	requestSet.send( sheller, &p_,  tgtId, tgt.index(), fid, buf );
	// requestSetAck.send( sheller, &p_, 1 );

	while ( isAckPending() )
		Qinfo::mpiClearQ( &p_ );
}

////////////////////////////////////////////////////////////////////////

// Deprecated. Will go into Shell.
bool set( Eref& dest, const string& destField, const string& val )
{
	static Id shellid;
	static BindIndex setBinding = 0; // Need to fix up.
	Element* shell = shellid();
	SrcFinfo1< string > sf( "set", "dummy" );

	const Finfo* f = dest.element()->cinfo()->findFinfo( destField );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	if ( !df )
		return 0;
	
	FuncId fid = df->getFid();
	const OpFunc* func = df->getOpFunc();

	// FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	// const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );
	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			shell->clearBinding( setBinding );
			shell->clearBinding( setBinding );
			Eref shelle = shellid.eref();
			Msg* m = new SingleMsg( shelle, dest );
			shell->addMsgAndFunc( m->mid(), fid, setBinding );
			sf.send( shelle, Shell::procInfo(), val );
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

bool get( const Eref& dest, const string& destField )
{
	static Id shellid;
	static BindIndex getBindIndex = 0;

	static const Finfo* reqFinfo = shellCinfo->findFinfo( "requestGet" );
	static const SrcFinfo1< FuncId >* rf = 
		dynamic_cast< const SrcFinfo1< FuncId >* >( reqFinfo );

	// static FuncId retFunc = shellCinfo->getOpFuncId( "handleGet" );
	static SrcFinfo1< string > sf( "get", "dummy" );

	static Element* shell = shellid();
	static Eref shelle( shell, 0 );

	const DestFinfo* hf = dynamic_cast< const DestFinfo* >( 
		shellCinfo->findFinfo( "handleGet" ) );
	assert( hf );
	FuncId retFunc = hf->getFid();

	const Finfo* f = dest.element()->cinfo()->findFinfo( destField );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	if ( !df )
		return 0;
	
	FuncId fid = df->getFid();
	const OpFunc* func = df->getOpFunc();
	// FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	// const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );

	assert( rf != 0 );

	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			shell->clearBinding( getBindIndex );
			Msg* m = new SingleMsg( shelle, dest );
			shell->addMsgAndFunc( m->mid(), fid, getBindIndex );
			rf->send( shelle, Shell::procInfo(), retFunc );
			// Now, dest has to clearQ, do its stuff, then src has to clearQ
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

