/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ParallelMsgSrc.h"

bool ParallelMsgSrc::add( RecvFunc rf, const Ftype* ft, Conn* target )
{
	unsigned long j = indexOfMatchingFunc( rf );

	if ( c_->connect( target, j ) ) {
		if ( j == rfuncs_.size() ) { // new function
			rfuncs_.push_back( rf );
			targetType_.push_back( ft );
		}
		return 1;
	}
	return 0;
}

void ParallelMsgSrc::send( char* dataPtr ) 
{
	if ( rfuncs_.size() != targetType_.size() ) {
		cerr << "Warning: ParallelMsgSrc::send: size of rfuncs != targetType\n";
		return;
	}
	vector< Conn* >::const_iterator j;
	// char* dataPtr = &( inbuf_.front() );
	//const Ftype* ft = &( targetType_.front() );
	for (size_t i = 0; i < rfuncs_.size(); i++) {
		const Ftype* ft = targetType_[ i ];
		RecvFunc rf = rfuncs_[ i ];
		for ( j = c_->begin( i ); j != c_->end( i ); j++ )
			dataPtr = ft->rfuncAdapter( *j, rf, dataPtr );
	}
}
