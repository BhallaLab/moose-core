/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FieldOpFunc.h"

/////////////////////////////////////////////////////////////////
// Here we define the functions used by the FieldOpFunc for setting
// up the 'foreach' iteration of the FieldDataHandlerBase.
/////////////////////////////////////////////////////////////////

FieldOpFunc::FieldOpFunc( const OpFunc* f, Element* e, 
	unsigned int argIncrement, unsigned int* argOffset )
	: 
		f_( f ),
		e_( e ),
		argIncrement_( argIncrement ),
		argOffset_( argOffset )
{
	*argOffset_ = 0;
	fdh_ = dynamic_cast< FieldDataHandlerBase* >( e_->dataHandler() );
}

void FieldOpFunc::op(const Eref& e, const Qinfo* q, const double* buf ) 
	const
{
	unsigned long long index = e.index().value() << fdh_->numFieldBits();

	for ( unsigned int i = 0; i < fdh_->localEntries(); ++i ) {
		Eref fielder( e_, index + i );
		f_->op( fielder, q, buf + *argOffset_ ); 
		*argOffset_ += argIncrement_;
	}
}
//////////////////////////////////////////////////////////////////
