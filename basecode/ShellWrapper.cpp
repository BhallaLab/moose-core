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
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"command", &ShellWrapper::getCommandConn,
		"commandIn, commandReturn" ),
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
