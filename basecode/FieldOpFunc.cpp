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
// up the 'forall' iteration of the FieldDataHandlerBase.
/////////////////////////////////////////////////////////////////

FieldOpFunc::FieldOpFunc( const OpFunc* f, Element* e, 
	unsigned int argSize, unsigned int numArgs, unsigned int* argOffset )
	: 
		f_( f ),
		e_( e ),
		argSize_( argSize ),
		maxArgOffset_( numArgs * argSize  ),
		argOffset_( argOffset )
{
	*argOffset_ = 0;
	fdh_ = dynamic_cast< FieldDataHandlerBase* >( e_->dataHandler() );
}

void FieldOpFunc::op(const Eref& e, const Qinfo* q, const double* buf ) 
	const
{
	unsigned int index = e.index().value() << fdh_->numFieldBits();

	unsigned int fieldArraySize = 
		fdh_->getFieldArraySize( e.index().value() );

	for ( unsigned int i = 0; i < fieldArraySize; ++i ) {
		Eref fielder( e_, index + i );
		//  There is a little problem here, if we run out of numArgs.
		f_->op( fielder, q, buf + i * argSize_ ); 

		// f_->op( fielder, q, buf + *argOffset_ ); 
		/*
		*argOffset_ += argSize_;
		if ( *argOffset_ >= maxArgOffset_ )
			*argOffset_ = 0;
		*/
	}
}
//////////////////////////////////////////////////////////////////
// Define funcs for FieldOpFuncSingle
//////////////////////////////////////////////////////////////////

FieldOpFuncSingle::FieldOpFuncSingle( const OpFunc* f, Element* e )
	: 
		f_( f ),
		e_( e )
{
	fdh_ = dynamic_cast< FieldDataHandlerBase* >( e_->dataHandler() );
}

void FieldOpFuncSingle::op(const Eref& e, const Qinfo* q, const double* buf)
	const
{
	unsigned long long index = e.index().value() << fdh_->numFieldBits();
	unsigned int fieldArraySize = 
		fdh_->getFieldArraySize( e.index().value() );

	for ( unsigned int i = 0; i < fieldArraySize; ++i ) {
		Eref fielder( e_, index + i );
		f_->op( fielder, q, buf );
	}
}
//////////////////////////////////////////////////////////////////
