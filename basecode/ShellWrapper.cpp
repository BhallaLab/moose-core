#include "header.h"
typedef double ProcArg;
typedef int  SynInfo;
#include "Shell.h"
#include "ShellWrapper.h"


Finfo* ShellWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"isInteractive", &ShellWrapper::getIsInteractive,
			&ShellWrapper::setIsInteractive, "int" ),
	new ValueFinfo< int >(
		"totalnodes", &ShellWrapper::getTotalNodes,
			&ShellWrapper::setTotalNodes, "int" ),
	new ValueFinfo< int >(
		"mynode", &ShellWrapper::getMyNode,
			&ShellWrapper::setMyNode, "int" ),
	new ValueFinfo< string >(
		"parser", &ShellWrapper::getParser,
		&ShellWrapper::setParser, "string" ),
	new ValueFinfo< string >(
		"response", &ShellWrapper::getResponse,
		&ShellWrapper::setResponse, "string" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc1Finfo< string > (
		"commandReturn", &ShellWrapper::getCommandReturnSrc,
		"commandIn" ),
	new NSrc1Finfo< string > (
		"remoteCommandOut", &ShellWrapper::getRemoteCommandSrc, "" ),
	new SingleSrc1Finfo< Element* > (
		"schedNewObjectOut", &ShellWrapper::getSchedNewObjectSrc, "" ),
	new NSrc3Finfo< Field, int, int > (
		"addOutgoingOut", &ShellWrapper::getAddOutgoingSrc, "" ),
	new NSrc3Finfo< Field, int, int > (
		"addIncomingOut", &ShellWrapper::getAddIncomingSrc, "" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< string, string >(
		"addIn", &ShellWrapper::addFunc,
		&ShellWrapper::getAddInConn, "" ),
	new Dest2Finfo< string, string >(
		"dropIn", &ShellWrapper::dropFunc,
		&ShellWrapper::getDropInConn, "" ),
	new Dest2Finfo< string, string >(
		"setIn", &ShellWrapper::setFunc,
		&ShellWrapper::getSetInConn, "" ),
	new Dest2Finfo< string, string >(
		"createIn", &ShellWrapper::createFunc,
		&ShellWrapper::getCreateInConn, "" ),
	new Dest1Finfo< string >(
		"deleteIn", &ShellWrapper::deleteFunc,
		&ShellWrapper::getDeleteInConn, "" ),
	new Dest2Finfo< string, string >(
		"moveIn", &ShellWrapper::moveFunc,
		&ShellWrapper::getMoveInConn, "" ),
	new Dest2Finfo< string, string >(
		"copyIn", &ShellWrapper::copyFunc,
		&ShellWrapper::getCopyInConn, "" ),
	new Dest2Finfo< string, string >(
		"copyShallowIn", &ShellWrapper::copyShallowFunc,
		&ShellWrapper::getCopyShallowInConn, "" ),
	new Dest2Finfo< string, string >(
		"copyHaloIn", &ShellWrapper::copyHaloFunc,
		&ShellWrapper::getCopyHaloInConn, "" ),
	new Dest1Finfo< string >(
		"ceIn", &ShellWrapper::ceFunc,
		&ShellWrapper::getCeInConn, "" ),
	new Dest1Finfo< string >(
		"pusheIn", &ShellWrapper::pusheFunc,
		&ShellWrapper::getPusheInConn, "" ),
	new Dest0Finfo(
		"popeIn", &ShellWrapper::popeFunc,
		&ShellWrapper::getPopeInConn, "" ),
	new Dest2Finfo< string, string >(
		"aliasIn", &ShellWrapper::aliasFunc,
		&ShellWrapper::getAliasInConn, "" ),
	new Dest0Finfo(
		"quitIn", &ShellWrapper::quitFunc,
		&ShellWrapper::getQuitInConn, "" ),
	new Dest0Finfo(
		"stopIn", &ShellWrapper::stopFunc,
		&ShellWrapper::getStopInConn, "" ),
	new Dest0Finfo(
		"resetIn", &ShellWrapper::resetFunc,
		&ShellWrapper::getResetInConn, "" ),
	new Dest2Finfo< string, string >(
		"stepIn", &ShellWrapper::stepFunc,
		&ShellWrapper::getStepInConn, "" ),
	new Dest1Finfo< string >(
		"callIn", &ShellWrapper::callFunc,
		&ShellWrapper::getCallInConn, "" ),
	new Dest1Finfo< string >(
		"getIn", &ShellWrapper::getFunc,
		&ShellWrapper::getGetInConn, "" ),
	new Dest2Finfo< string, string >(
		"getmsgIn", &ShellWrapper::getmsgFunc,
		&ShellWrapper::getGetmsgInConn, "" ),
	new Dest2Finfo< string, string >(
		"isaIn", &ShellWrapper::isaFunc,
		&ShellWrapper::getIsaInConn, "" ),
	new Dest1Finfo< string >(
		"showIn", &ShellWrapper::showFunc,
		&ShellWrapper::getShowInConn, "" ),
	new Dest1Finfo< string >(
		"showmsgIn", &ShellWrapper::showmsgFunc,
		&ShellWrapper::getShowmsgInConn, "" ),
	new Dest1Finfo< string >(
		"showobjectIn", &ShellWrapper::showobjectFunc,
		&ShellWrapper::getShowobjectInConn, "" ),
	new Dest0Finfo(
		"pweIn", &ShellWrapper::pweFunc,
		&ShellWrapper::getPweInConn, "" ),
	new Dest1Finfo< string >(
		"leIn", &ShellWrapper::leFunc,
		&ShellWrapper::getLeInConn, "" ),
	new Dest0Finfo(
		"listCommandsIn", &ShellWrapper::listCommandsFunc,
		&ShellWrapper::getListCommandsInConn, "" ),
	new Dest0Finfo(
		"listClassesIn", &ShellWrapper::listClassesFunc,
		&ShellWrapper::getListClassesInConn, "" ),
	new Dest2Finfo< vector< string >*, int >(
		"echoIn", &ShellWrapper::echoFunc,
		&ShellWrapper::getEchoInConn, "" ),
	new Dest2Finfo< int, const char** >(
		"commandIn", &ShellWrapper::commandFunc,
		&ShellWrapper::getCommandConn, "commandReturn" ),
	new Dest1Finfo< string >(
		"remoteCommandIn", &ShellWrapper::remoteCommandFunc,
		&ShellWrapper::getRemoteCommandConn, "" ),
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"command", &ShellWrapper::getCommandConn,
		"commandIn, commandReturn" ),
	new SharedFinfo(
		"remoteCommand", &ShellWrapper::getRemoteCommandConn,
		"remoteCommandOut, remoteCommandIn, addOutgoingOut, addIncomingOut" ),
};

const Cinfo ShellWrapper::cinfo_(
	"Shell",
	"Upinder S. Bhalla, Oct 2005, NCBS",
	"Shell: Shell class, provides many of the basic environment functions\nused by the parser.",
	"Neutral",
	ShellWrapper::fieldArray_,
	sizeof(ShellWrapper::fieldArray_)/sizeof(Finfo *),
	&ShellWrapper::create
);

///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* addInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, addInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* dropInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, dropInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* setInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, setInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* createInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, createInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* deleteInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, deleteInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* moveInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, moveInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* copyInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, copyInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* copyShallowInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, copyShallowInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* copyHaloInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, copyHaloInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* ceInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, ceInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* pusheInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, pusheInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* popeInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, popeInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* aliasInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, aliasInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* quitInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, quitInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stopInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, stopInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* resetInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, resetInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stepInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, stepInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* callInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, callInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* getInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, getInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* getmsgInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, getmsgInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* isaInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, isaInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* showInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, showInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* showmsgInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, showmsgInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* showobjectInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, showobjectInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* pweInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, pweInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* leInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, leInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* listCommandsInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, listCommandsInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* listClassesInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, listClassesInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* echoInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, echoInConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* commandConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, commandConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

Element* schedNewObjectConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ShellWrapper, schedNewObjectConn_ );
	return reinterpret_cast< ShellWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
void ShellWrapper::setIsInteractive( Conn* c, int value ) {
	static_cast< ShellWrapper* >( c->parent() )->isInteractive_ = value;
}
int ShellWrapper::getIsInteractive( const Element* e ) {
	return static_cast< const ShellWrapper* >( e )->isInteractive_;
}
void ShellWrapper::setTotalNodes( Conn* c, int value ) {
	static_cast< ShellWrapper* >( c->parent() )->totalNodes_ = value;
}
int ShellWrapper::getTotalNodes( const Element* e ) {
	return static_cast< const ShellWrapper* >( e )->totalNodes_;
}
void ShellWrapper::setMyNode( Conn* c, int value ) {
	static_cast< ShellWrapper* >( c->parent() )->myNode_ = value;
}
int ShellWrapper::getMyNode( const Element* e ) {
	return static_cast< const ShellWrapper* >( e )->myNode_;
}
void ShellWrapper::setParser( Conn* c, string value ) {
	static_cast< ShellWrapper* >( c->parent() )->parser_ = value;
}
string ShellWrapper::getParser( const Element* e ) {
	return static_cast< const ShellWrapper* >( e )->parser_;
}
void ShellWrapper::setResponse( Conn* c, string value ) {
	static_cast< ShellWrapper* >( c->parent() )->response_ = value;
}
string ShellWrapper::getResponse( const Element* e ) {
	return static_cast< const ShellWrapper* >( e )->response_;
}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
SingleMsgSrc* ShellWrapper::getCommandReturnSrc( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->commandReturnSrc_ );
}

NMsgSrc* ShellWrapper::getRemoteCommandSrc( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->remoteCommandSrc_ );
}

SingleMsgSrc* ShellWrapper::getSchedNewObjectSrc( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->schedNewObjectSrc_ );
}

NMsgSrc* ShellWrapper::getAddOutgoingSrc( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->addOutgoingSrc_ );
}

NMsgSrc* ShellWrapper::getAddIncomingSrc( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->addIncomingSrc_ );
}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
void ShellWrapper::addFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		addFuncLocal( src, dest );
}
void ShellWrapper::dropFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		dropFuncLocal( src, dest );
}
void ShellWrapper::setFunc( Conn* c, string field, string value ) {
	static_cast< ShellWrapper* >( c->parent() )->
		setFuncLocal( field, value );
}
void ShellWrapper::createFunc( Conn* c, string type, string path ) {
	static_cast< ShellWrapper* >( c->parent() )->
		createFuncLocal( type, path );
}
void ShellWrapper::deleteFunc( Conn* c, string path ) {
	static_cast< ShellWrapper* >( c->parent() )->
		deleteFuncLocal( path );
}
void ShellWrapper::moveFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		moveFuncLocal( src, dest );
}
void ShellWrapper::copyFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		copyFuncLocal( src, dest );
}
void ShellWrapper::copyShallowFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		copyShallowFuncLocal( src, dest );
}
void ShellWrapper::copyHaloFunc( Conn* c, string src, string dest ) {
	static_cast< ShellWrapper* >( c->parent() )->
		copyHaloFuncLocal( src, dest );
}
void ShellWrapper::ceFunc( Conn* c, string newpath ) {
	static_cast< ShellWrapper* >( c->parent() )->
		ceFuncLocal( newpath );
}
void ShellWrapper::pusheFunc( Conn* c, string newpath ) {
	static_cast< ShellWrapper* >( c->parent() )->
		pusheFuncLocal( newpath );
}
void ShellWrapper::popeFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		popeFuncLocal(  );
}
void ShellWrapper::aliasFunc( Conn* c, string origfunc, string newfunc ) {
	static_cast< ShellWrapper* >( c->parent() )->
		aliasFuncLocal( origfunc, newfunc );
}
void ShellWrapper::quitFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		quitFuncLocal(  );
}
void ShellWrapper::stopFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		stopFuncLocal(  );
}
void ShellWrapper::resetFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		resetFuncLocal(  );
}
void ShellWrapper::stepFunc( Conn* c, string steptime, string options ) {
	static_cast< ShellWrapper* >( c->parent() )->
		stepFuncLocal( steptime, options );
}
void ShellWrapper::callFunc( Conn* c, string args ) {
	static_cast< ShellWrapper* >( c->parent() )->
		callFuncLocal( args );
}
void ShellWrapper::getFunc( Conn* c, string field ) {
	static_cast< ShellWrapper* >( c->parent() )->
		getFuncLocal( field );
}
void ShellWrapper::getmsgFunc( Conn* c, string field, string options ){
	static_cast< ShellWrapper* >( c->parent() )->
		getmsgFuncLocal( field, options );
}
void ShellWrapper::isaFunc( Conn* c, string type, string field ) {
	static_cast< ShellWrapper* >( c->parent() )->
		isaFuncLocal( type, field );
}
void ShellWrapper::showFunc( Conn* c, string field ) {
	static_cast< ShellWrapper* >( c->parent() )->
		showFuncLocal( field );
}
void ShellWrapper::showmsgFunc( Conn* c, string field ) {
	static_cast< ShellWrapper* >( c->parent() )->
		showmsgFuncLocal( field );
}
void ShellWrapper::showobjectFunc( Conn* c, string classname ) {
	static_cast< ShellWrapper* >( c->parent() )->
		showobjectFuncLocal( classname );
}
void ShellWrapper::pweFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		pweFuncLocal(  );
}
void ShellWrapper::leFunc( Conn* c, string start ) {
	static_cast< ShellWrapper* >( c->parent() )->
		leFuncLocal( start );
}
void ShellWrapper::listCommandsFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		listCommandsFuncLocal(  );
}
void ShellWrapper::listClassesFunc( Conn* c ) {
	static_cast< ShellWrapper* >( c->parent() )->
		listClassesFuncLocal(  );
}
void ShellWrapper::echoFunc( Conn* c, vector< string >* s, int options ) {
	static_cast< ShellWrapper* >( c->parent() )->
		echoFuncLocal( *s, options );
}

void ShellWrapper::commandFunc( Conn* c, int argc, const char** argv ){
	static_cast< ShellWrapper* >( c->parent() )->
		commandFuncLocal( argc, argv );
}

void ShellWrapper::remoteCommandFunc( Conn* c, string arglist ) {
	static_cast< ShellWrapper* >( c->parent() )->
		remoteCommandFuncLocal( arglist );
}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
Conn* ShellWrapper::getAddInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->addInConn_ );
}
Conn* ShellWrapper::getDropInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->dropInConn_ );
}
Conn* ShellWrapper::getSetInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->setInConn_ );
}
Conn* ShellWrapper::getCreateInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->createInConn_ );
}
Conn* ShellWrapper::getDeleteInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->deleteInConn_ );
}
Conn* ShellWrapper::getMoveInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->moveInConn_ );
}
Conn* ShellWrapper::getCopyInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->copyInConn_ );
}
Conn* ShellWrapper::getCopyShallowInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->copyShallowInConn_ );
}
Conn* ShellWrapper::getCopyHaloInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->copyHaloInConn_ );
}
Conn* ShellWrapper::getCeInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->ceInConn_ );
}
Conn* ShellWrapper::getPusheInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->pusheInConn_ );
}
Conn* ShellWrapper::getPopeInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->popeInConn_ );
}
Conn* ShellWrapper::getAliasInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->aliasInConn_ );
}
Conn* ShellWrapper::getQuitInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->quitInConn_ );
}
Conn* ShellWrapper::getStopInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->stopInConn_ );
}
Conn* ShellWrapper::getResetInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->resetInConn_ );
}
Conn* ShellWrapper::getStepInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->stepInConn_ );
}
Conn* ShellWrapper::getCallInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->callInConn_ );
}
Conn* ShellWrapper::getGetInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->getInConn_ );
}
Conn* ShellWrapper::getGetmsgInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->getmsgInConn_ );
}
Conn* ShellWrapper::getIsaInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->isaInConn_ );
}
Conn* ShellWrapper::getShowInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->showInConn_ );
}
Conn* ShellWrapper::getShowmsgInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->showmsgInConn_ );
}
Conn* ShellWrapper::getShowobjectInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->showobjectInConn_ );
}
Conn* ShellWrapper::getPweInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->pweInConn_ );
}
Conn* ShellWrapper::getLeInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->leInConn_ );
}
Conn* ShellWrapper::getListCommandsInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->listCommandsInConn_ );
}
Conn* ShellWrapper::getListClassesInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->listClassesInConn_ );
}
Conn* ShellWrapper::getEchoInConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->echoInConn_ );
}

// Note that this is a shared conn, so no direction pertains.
Conn* ShellWrapper::getCommandConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->commandConn_ );
}
Conn* ShellWrapper::getRemoteCommandConn( Element* e ) {
	return &( static_cast< ShellWrapper* >( e )->remoteCommandConn_ );
}

//////////////////////////////////////////////////////////////////
// Parallel Access utility fuctions.
//////////////////////////////////////////////////////////////////

// This connects to the appropriate postmaster, and forwards a request
// to the target node's shell to complete the message.
bool ShellWrapper::addToRemoteNode( Field& s, const string& dest, int destNode )
{
	char destLine[200];
	sprintf( destLine, "/postmasters/node%d/destIn", destNode );
	Field d( destLine );

	Element* t = traverseSrcToTick( s );
	int tick = -1; // Default indicates it is an async call.
	if ( t ) {
		Ftype1< int >::get( t, "ordinal", tick );
	}
	int size = s->ftype()->size();
	sprintf( destLine, "addfromremote %d %s %d %d",
		myNode_, dest.c_str(), tick, size );
	int tgtNode = destNode;
	if ( tgtNode > myNode_ )
		tgtNode--; // to skip the entry of the current node.
	addOutgoingSrc_.sendTo( tgtNode, s, tick, size );
	remoteCommandSrc_.sendTo( tgtNode, destLine );
	// cout << "remoteCommandSrc_.sendTo( destNode = " << destNode <<
			// ", destLine = " << destLine << " );\n";
	return 1;
}

// Needs to be an atomic operation, to make the connection from the
// postmaster to the target object, and also to set up the size and
// schedule of the just-connected message. The message still cannot
// be used, till the reset is done.
void ShellWrapper::addFromRemoteNode( int srcNode, Field& dest,
				int tick, int size)
{
	addIncomingSrc_.sendTo( srcNode, dest, tick, size );
}

void ShellWrapper::sendRemoteCommand( 
				const string& command, int destNode )
{
	if ( destNode < 0 ) {
		remoteCommandSrc_.send( command );
	} else if ( destNode >= 0 && destNode < totalNodes_ ) {
		remoteCommandSrc_.sendTo( destNode, command );
	}
}

// Utility function so that the Shell can access the message.
void ShellWrapper::schedNewObject( Element* e )
{
	schedNewObjectSrc_.send( e );
}
