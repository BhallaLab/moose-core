#include "header.h"
#include "Reaction.h"
#include "ReactionWrapper.h"


Finfo* ReactionWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"kf", &ReactionWrapper::getKf, 
		&ReactionWrapper::setKf, "double" ),
	new ValueFinfo< double >(
		"kb", &ReactionWrapper::getKb, 
		&ReactionWrapper::setKb, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc2Finfo< double, double >(
		"subOut", &ReactionWrapper::getSubSrc, 
		"processIn", 1 ),
	new NSrc2Finfo< double, double >(
		"prdOut", &ReactionWrapper::getPrdSrc, 
		"processIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &ReactionWrapper::reinitFunc,
		&ReactionWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ReactionWrapper::processFunc,
		&ReactionWrapper::getProcessConn, "subOut, prdOut", 1 ),
	new Dest1Finfo< double >(
		"subIn", &ReactionWrapper::subFunc,
		&ReactionWrapper::getSubConn, "", 1 ),
	new Dest1Finfo< double >(
		"prdIn", &ReactionWrapper::prdFunc,
		&ReactionWrapper::getPrdConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &ReactionWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"sub", &ReactionWrapper::getSubConn,
		"subIn, subOut" ),
	new SharedFinfo(
		"prd", &ReactionWrapper::getPrdConn,
		"prdIn, prdOut" ),
};

const Cinfo ReactionWrapper::cinfo_(
	"Reaction",
	"Upinder S. Bhalla, 2005, NCBS",
	"Reaction: Reaction class, handles binding and conversion reactions\nnot involving enzymatic steps. Computes reversible reactions\nbut the rates can be set to zero to give irreversibility.\nOrder of substrates and products set by the number of \nmessages between them.",
	"Neutral",
	ReactionWrapper::fieldArray_,
	sizeof(ReactionWrapper::fieldArray_)/sizeof(Finfo *),
	&ReactionWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ReactionWrapper::processFuncLocal( ProcInfo info )
{
	subSrc_.send( B_, A_ );
	prdSrc_.send( A_, B_ );
	A_ = kf_;
	B_ = kb_;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnReactionLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ReactionWrapper, processConn_ );
	return reinterpret_cast< ReactionWrapper* >( ( unsigned long )c - OFFSET );
}

