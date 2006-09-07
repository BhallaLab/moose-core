/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "UIWidget.h"
#include "UIWidgetWrapper.h"


Finfo* UIWidgetWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"defaultBase", &UIWidgetWrapper::getDefaultBase, 
		&UIWidgetWrapper::setDefaultBase, "string" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc0Finfo(
		"uiActionOut", &UIWidgetWrapper::getUiActionSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"uiActionIn", &UIWidgetWrapper::uiActionFunc,
		&UIWidgetWrapper::getUiActionConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"uiAction", &UIWidgetWrapper::getUiActionConn,
		"uiActionOut, uiActionIn" ),
};

const Cinfo UIWidgetWrapper::cinfo_(
	"UIWidget",
	"Josef Svitak, 2006-02-13",
	"UIWidget: First attempt at drilling a gui into Moose.\nHandles all gui creation requests.",
	"Neutral",
	UIWidgetWrapper::fieldArray_,
	sizeof(UIWidgetWrapper::fieldArray_)/sizeof(Finfo *),
	&UIWidgetWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* uiActionConnUIWidgetLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( UIWidgetWrapper, uiActionConn_ );
	return reinterpret_cast< UIWidgetWrapper* >( ( unsigned long )c - OFFSET );
}

