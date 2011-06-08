/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"

/**
 * Used to do the serious argument juggling in GetOpFunc::op
 */
void fieldOp( const Eref& e, const Qinfo* q, const char* buf, 
	const char* data, unsigned int size )
{

	FuncId retFunc = *reinterpret_cast< const FuncId* >( buf );

	PrepackedBuffer pb( data, size );
	Conv< PrepackedBuffer > conv( pb );
	unsigned int totSize = conv.size();
	char* temp = new char[ totSize ];
	conv.val2buf( temp );

	MsgFuncBinding mfb( q->mid(), retFunc );
	Qinfo retq( retFunc, e.index(), totSize, 0 );
	retq.addToQbackward( q->getProcInfo(), mfb, temp );
	delete[] temp;
}

/**
 * Used to check if the Eref e should be sending any data back to
 * the master node.
 * We avoid sending data back to master node from globals,
 * as the global object will also send data back.
 */
 bool skipWorkerNodeGlobal( const Eref& e )
 {
	return ( Shell::myNode() != 0 && 
		e.element()->dataHandler()->isGlobal() );
}

//////////////////////////////////////////////////////////////////

OpFuncDummy::OpFuncDummy()
{;}

bool OpFuncDummy::checkFinfo( const Finfo* s ) const
{
	return dynamic_cast< const SrcFinfo0* >( s );
}

bool OpFuncDummy::checkSet( const SetGet* s ) const {
	return dynamic_cast< const SetGet0* >( s );
}

bool OpFuncDummy::strSet( const Eref& tgt,
	const string& field, const string& arg ) const
{
	cout << "In OpFuncDummy::strSet. should not happen\n";
	return 0;
}

void OpFuncDummy::op( const Eref& e, const char* buf ) const {
	;
}

void OpFuncDummy::op( const Eref& e, const Qinfo* q, 
	const char* buf ) const {
	;
}

string OpFuncDummy::rttiType() const
{
	return "void";
}
