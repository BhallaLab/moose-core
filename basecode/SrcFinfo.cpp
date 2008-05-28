/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SimpleConn.h"
#include "SrcFinfo.h"

bool SrcFinfo::add(
	Eref e, Eref destElm, const Finfo* destFinfo, 
	unsigned int connTainerOption
) const
{
	assert( destFinfo != 0 );
	assert( e.e != 0 );
	assert( destElm.e != 0 );

	unsigned int srcFuncId = 0;
	unsigned int destFuncId = 0;
	int destMsg = 0;
	unsigned int destIndex = 0;
	if ( destFinfo->respondToAdd( destElm, e, ftype(),
							srcFuncId, destFuncId,
							destMsg, destIndex ) )
	{
		// All these assertions say that this is a single message,
		// not a shared one.
		assert( FuncVec::getFuncVec( srcFuncId )->size() == 0 );
		assert( FuncVec::getFuncVec( destFuncId )->size() == 1 );
		unsigned int srcIndex = e.e->numTargets( msg_, e.i );

		return Msg::add( e, destElm,
			msg_, destMsg, srcIndex, destIndex, 
			srcFuncId, destFuncId,
			connTainerOption );
	}
	return 0;
}

bool SrcFinfo::respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& destFuncId,
					int& destMsgId, unsigned int& destIndex
) const
{
	return 0; // for now we cannot handle this.
}

bool SrcFinfo::strSet( Eref e, const std::string &s ) const
{
		return 0;
}


const Finfo* SrcFinfo::match( const Element* e, const ConnTainer* c ) const
{
	const Msg* m = e->msg( msg() ); 
	assert ( !m->isDest() ); // This is a SrcFinfo so it must be a src
	// If we wanted to be really really picky we should go through the
	// ConnTainer vector on the msg, and find 'c' on it. But this is
	// fast and should work provided we're not in the middle of rebuilding
	// the messaging.
	if ( c->e1() == e && c->msg1() == msg() )
		return this;
	return 0;
}

bool SrcFinfo::inherit( const Finfo* baseFinfo )
{
	const SrcFinfo* other =
			dynamic_cast< const SrcFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			msg_ = other->msg_;
			return 1;
	} 
	return 0;
}


bool SrcFinfo::getSlot( const string& name, Slot& ret ) const
{
	if ( name != this->name() ) return 0;
	ret = Slot( msg_, 0 );
	return 1;
}
