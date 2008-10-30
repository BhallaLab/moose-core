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
#include "DestFinfo.h"

DestFinfo::DestFinfo( const string& name,
						const Ftype *f, 
						RecvFunc rfunc,
						const string& doc,
						unsigned int msg )
	: Finfo( name, f, doc ), 
	rfunc_( rfunc ), 
	msg_( msg )
{
	;
}

/**
 * Check the request to form a message, and return true with 
 * the necessary information if all is well.
 */
bool DestFinfo::respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int & returnFuncId,
					int& destMsgId, unsigned int& destIndex
) const
{
	assert ( srcType != 0 );

	if ( ftype()->isSameType( srcType ) && 
		FuncVec::getFuncVec( srcFuncId )->size() == 0 ) {
		assert ( src.e != 0 && e.e != 0 );
		returnFuncId = fv_->id();
		destMsgId = msg_;
		destIndex = e.e->numTargets( msg_, e.i );
		// numDest = 1;
		return 1;
	}
	cout << "DestFinfo::add: Error: Message Type mismatch for " << e.name() << " to " << src.name() << "\n";
	return 0;
}

void DestFinfo::countMessages( unsigned int& num )
{
	msg_ = -num;
	num++;
}

int DestFinfo::msg() const
{
	return msg_;
}

/**
 * Directly call the recvFunc on the element with the string argument
 * typecast appropriately.
 */
bool DestFinfo::strSet( Eref e, const std::string &s ) const
{
	/**
	 * \todo Here we will ask the Ftype to do the string conversion
	 * and call the properly typecast rfunc.
	 */
	return ftype()->strSet( e, this, s );
}

const Finfo* DestFinfo::match( const Element* e, const ConnTainer* c ) const
{
	const Msg* m = e->msg( msg() ); 
	assert ( m->isDest() ); // This is a DestFinfo so it must be a dest.
	// If we wanted to be really really picky we should go through the
	// ConnTainer vector on the msg, and find 'c' on it. But this is
	// fast and should work provided we're not in the middle of rebuilding
	// the messaging.
	if ( c->e2() == e && c->msg2() == msg() )
		return this;
	return 0;
}

bool DestFinfo::inherit( const Finfo* baseFinfo )
{
	const DestFinfo* other =
			dynamic_cast< const DestFinfo* >( baseFinfo );
	if ( other && ftype()->isSameType( baseFinfo->ftype() ) ) {
			msg_ = other->msg_;
			return 1;
	} 
	return 0;
}

bool DestFinfo::getSlot( const string& name, Slot& ret ) const
{
	if ( name != this->name() ) return 0;
	ret = Slot( msg_, 0 );
	return 1;
}

void DestFinfo::addFuncVec( const string& cname )
{
	fv_ = new FuncVec( cname, name() );
	fv_->addFunc( rfunc_, ftype() );
	fv_->setDest();
}
