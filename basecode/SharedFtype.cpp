/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SimpleElement.h"
#include "Send.h"
#include "SetConn.h"
#include "SharedFtype.h"
#include "DestFinfo.h"
#include "SrcFinfo.h"

/**
 * Here we construct the SharedFtype using Finfos. The array has either
 * Src or Dest Finfos and we extract their Ftypes from them.
 */
SharedFtype::SharedFtype( Finfo** finfos, unsigned int n )
	: Ftype( "shared" )
{
	nValues_ = 0;
	size_ = 0;
	for (unsigned int i = 0; i < n; i++ ) {
		const Ftype *f = finfos[i]->ftype();
		DestFinfo* df = dynamic_cast< DestFinfo* >( finfos[i] );
		nValues_ += f->nValues();
		size_ += f->size();
		if ( df == 0 )
			srcTypes_.push_back( f );
		else
			destTypes_.push_back( f );
	}
	/*
	match_ = new SharedFtype();
	match_->nValues_ = nValues_;
	match_->size_ = size_;
	match_->match_ = this;
	*/
	vector< const Ftype* >::iterator i;
	for ( i = destTypes_.begin(); i != destTypes_.end(); i++ ) {
		// match_->srcTypes_.push_back( ( *i )->makeMatchingType() );
		addProxyFunc( *i );
	}
	for ( i = srcTypes_.begin(); i != srcTypes_.end(); i++ ) {
		// match_->destTypes_.push_back( ( *i )->makeMatchingType() );
		addSyncFunc( *i );
		addAsyncFunc( *i );
	}
}

/*
 * Strictly speaking this operation is not for identical types,
 * but for types that can send messages to each other. In fact
 * there are cases where a given Ftype would fail to match itself
 * on this test. 
 * \todo Should therefore rename it to isMatchingType.
 */
bool SharedFtype::isSameType( const Ftype* other ) const
{
	// Quick and dirty test to eliminate many non-matches.
	if ( nValues() != other->nValues() )
		return 0;
	const SharedFtype* sf = dynamic_cast< const SharedFtype* >( other );
	if ( sf ) {
		unsigned int i = 0;
		unsigned int end = srcTypes_.size();
		if ( end != sf->destTypes_.size() )
			return 0;
		for ( i = 0; i < end; i++ )
			if ( !srcTypes_[i]->isSameType( sf->destTypes_[i] ) )
					return 0;

		end = destTypes_.size();
		if ( end != sf->srcTypes_.size() )
			return 0;
		for ( i = 0; i < end; i++ )
			if ( !destTypes_[i]->isSameType( sf->srcTypes_[i] ) )
					return 0;

		return 1;
	}

	return 0;
}

/*
void inFunc( vector< IncomingFunc >& ret ) const
{
	vector< const Ftype* >::iterator i;
	for ( i = destTypes.begin(); i != destTypes.end(); i++ )
		( *i )->inFunc( ret );
}
*/

/*
void SharedFtype::syncFunc( vector< RecvFunc >& ret ) const
{
	vector< const Ftype* >::const_iterator i;
	for ( i = srcTypes_.begin(); i != srcTypes_.end(); i++ )
		( *i )->syncFunc( ret );
}

void SharedFtype::asyncFunc( vector< RecvFunc >& ret ) const
{
	vector< const Ftype* >::const_iterator i;
	for ( i = srcTypes_.begin(); i != srcTypes_.end(); i++ )
		( *i )->asyncFunc( ret );
}
*/

/**
 * This returns a precomputed Ftype with its baseType
 */
const Ftype* SharedFtype::baseFtype() const {
	return this;
}

/**
 * This generates the type string for the SharedFtype by going through
 * each of the sub-types and catenating their typeStr.
 */
string SharedFtype::typeStr() const
{
	vector< const Ftype* >::const_iterator i;
	string ret = "";
	for ( i = destTypes_.begin(); i != destTypes_.end(); i++ )
		ret = ret + (*i)->typeStr();
	for ( i = srcTypes_.begin(); i != srcTypes_.end(); i++ )
		ret = ret + (*i)->typeStr();
	return ret;
}

#ifdef DO_UNIT_TESTS
#include "ProcInfo.h"
#include "DerivedFtype.h"
void tempFunc( const Conn* c )
{
		string s = c->target().e->name();
		s = s + ".foo";
}

void sharedFtypeTest()
{
	cout << "\nTesting sharedFtype matching";
	// This one can match itself
	static Finfo* testArray1[] = {
			new DestFinfo( "d1", Ftype0::global(), &tempFunc ),
			new DestFinfo( "d2", Ftype1< int >::global(), &tempFunc ),
			new SrcFinfo( "s1", Ftype0::global() ),
			new SrcFinfo( "s2", Ftype1< int >::global() ),
	};
	// This one cannot match itself, but matches 3.
	static Finfo* testArray2[] = {
			new DestFinfo( "d1", Ftype0::global(), &tempFunc ),
			new DestFinfo( "d2", Ftype1< int >::global(), &tempFunc ),
			new SrcFinfo( "s1", Ftype1< int >::global() ),
			new SrcFinfo( "s2", Ftype0::global() ),
	};
	// This one matches with 2 in either direction.
	static Finfo* testArray3[] = {
			new DestFinfo( "d1", Ftype1< int >::global(), &tempFunc ),
			new DestFinfo( "d2", Ftype0::global(), &tempFunc ),
			new SrcFinfo( "s1", Ftype0::global() ),
			new SrcFinfo( "s2", Ftype1< int >::global() ),
	};

	// This one matches with 2 in either direction even 
	// though definition order is different
	static Finfo* testArray4[] = {
			new SrcFinfo( "s1", Ftype0::global() ),
			new SrcFinfo( "s2", Ftype1< int >::global() ),
			new DestFinfo( "d1", Ftype1< int >::global(), &tempFunc ),
			new DestFinfo( "d2", Ftype0::global(), &tempFunc ),
	};
	// This one has a different number of entries and doesn't match.
	static Finfo* testArray5[] = {
			new DestFinfo( "d1", Ftype1< int >::global(), &tempFunc ),
			new DestFinfo( "d2", Ftype0::global(), &tempFunc ),
			new SrcFinfo( "s1", Ftype0::global() ),
			new SrcFinfo( "s2", Ftype1< int >::global() ),
			new SrcFinfo( "s3", Ftype0::global() ),
	};

/*



	// This one can match itself
	static TypeFuncPair testArray1[] = {
			TypeFuncPair( Ftype0::global(), &tempFunc ),
			TypeFuncPair( Ftype1< int >::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), 0 ),
			TypeFuncPair( Ftype1< int >::global(), 0 ),
	};

	// This one cannot match itself, but matches 3.
	static TypeFuncPair testArray2[] = {
			TypeFuncPair( Ftype0::global(), &tempFunc ),
			TypeFuncPair( Ftype1< int >::global(), &tempFunc ),
			TypeFuncPair( Ftype1< int >::global(), 0 ),
			TypeFuncPair( Ftype0::global(), 0 ),
	};

	// This one matches with 2 in either direction.
	static TypeFuncPair testArray3[] = {
			TypeFuncPair( Ftype1< int >::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), 0 ),
			TypeFuncPair( Ftype1< int >::global(), 0 ),
	};

	// This one matches with 2 in either direction even 
	// though definition order is different
	static TypeFuncPair testArray4[] = {
			TypeFuncPair( Ftype0::global(), 0 ),
			TypeFuncPair( Ftype1< int >::global(), 0 ),
			TypeFuncPair( Ftype1< int >::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), &tempFunc ),
	};

	// This one has a different number of entries and doesn't match.
	static TypeFuncPair testArray5[] = {
			TypeFuncPair( Ftype1< int >::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), &tempFunc ),
			TypeFuncPair( Ftype0::global(), 0 ),
			TypeFuncPair( Ftype1< int >::global(), 0 ),
			TypeFuncPair( Ftype0::global(), 0 ),
	};
*/

	SharedFtype s1( testArray1, 4 );
	SharedFtype s2( testArray2, 4 );
	SharedFtype s3( testArray3, 4 );
	SharedFtype s4( testArray4, 4 );
	SharedFtype s5( testArray5, 5 );

	ASSERT( s1.isSameType( &s1 ), "s1 == s1" );
	ASSERT( !s1.isSameType( &s2 ), "s1 != s2" );
	ASSERT( !s1.isSameType( &s3 ), "s1 != s3" );
	ASSERT( !s1.isSameType( &s4 ), "s1 != s4" );
	ASSERT( !s1.isSameType( &s5 ), "s1 != s5" );
	ASSERT( !s1.isSameType( Ftype0::global() ), "s1 != Ftype0" );
	ASSERT( !s1.isSameType( Ftype1<int>::global() ),
					"s1 != Ftype1<int>" );

	ASSERT( !s2.isSameType( &s1 ), "s2 != s1" );
	ASSERT( !s2.isSameType( &s2 ), "s2 != s2" );
	ASSERT( s2.isSameType( &s3 ), "s2 == s3" );
	ASSERT( s2.isSameType( &s4 ), "s2 == s4" );
	ASSERT( !s2.isSameType( &s5 ), "s2 != s5" );

	ASSERT( !s3.isSameType( &s1 ), "s3 != s1" );
	ASSERT( s3.isSameType( &s2 ),  "s3 == s2" );
	ASSERT( !s3.isSameType( &s3 ), "s3 != s3" );
	ASSERT( !s3.isSameType( &s4 ), "s3 != s4" );
	ASSERT( !s3.isSameType( &s5 ), "s3 != s5" );

	ASSERT( !s4.isSameType( &s1 ), "s4 != s1" );
	ASSERT( s4.isSameType( &s2 ),  "s4 == s2" );
	ASSERT( !s4.isSameType( &s3 ), "s4 != s3" );
	ASSERT( !s4.isSameType( &s4 ), "s4 != s4" );
	ASSERT( !s4.isSameType( &s5 ), "s4 != s5" );

	ASSERT( !s5.isSameType( &s1 ), "s5 != s1" );
	ASSERT( !s5.isSameType( &s2 ), "s5 != s2" );
	ASSERT( !s5.isSameType( &s3 ), "s5 != s3" );
	ASSERT( !s5.isSameType( &s4 ), "s5 != s4" );
	ASSERT( !s5.isSameType( &s5 ), "s5 != s5" );
}
#endif
