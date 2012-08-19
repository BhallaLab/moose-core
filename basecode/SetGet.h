/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SETGET_H
#define _SETGET_H

// Forward declaration needed for localGet()
template< class T, class A > class GetOpFunc;

/**
 * Similar to Field< A >::fastGet(), except that an existing Msg is not needed.
 * 
 * Instant-return call for a single value. Bypasses all the queueing stuff.
 * It is hardcoded so type safety will have to be coded in too:
 * the dynamic_cast will catch it only at runtime.
 * 
 * Perhaps analogous localSet(), localLookupGet(), localGetVec(), etc. should
 * also be added.
 * 
 * Also, will be nice to change this to Field< A >::localGet() to make things
 * more uniform.
 */
template< class T, class A >
A localGet( const Eref& er, string field )
{
	const Finfo* finfo = er.element()->cinfo()->findFinfo( "get_" + field );
	assert( finfo );
	
	const DestFinfo* dest = dynamic_cast< const DestFinfo* >( finfo );
	assert( dest );
	
	const OpFunc* op = dest->getOpFunc();
	assert( op );
	
	const GetOpFunc< T, A >* gop =
		dynamic_cast< const GetOpFunc< T, A >* >( op );
	assert( gop );
	
	return gop->reduceOp( er );
}

class SetGet
{
	public:
		SetGet( const ObjId& oid )
			: oid_( oid )
		{
			if ( oid_.id.element() == 0 ) {
				cout << "Warning: SetGet: accessing null element: Was object deleted?\n";
			}
		}

		virtual ~SetGet()
		{;}

		/**
		 * Utility function to check that the target field matches this
		 * source type, to look up and pass back the fid, and to return
		 * the number of targetEntries.
		 * Tgt is passed in as the destination ObjId. May be changed inside,
		 * if the function determines that it should be directed to a 
		 * child Element acting as a Value.
		 * Checks arg # and types for a 'set' call. Can be zero to 3 args.
		 * Returns # of tgts if good. This is 0 if bad. 
		 * Passes back found fid.
		 */
		const OpFunc* checkSet( 
			const string& field, ObjId& tgt, FuncId& fid ) const;

//////////////////////////////////////////////////////////////////////
		/**
		 * Blocking 'get' call, returning into a string.
		 * There is a matching 'get<T> call, returning appropriate type.
		 */
		static bool strGet( const ObjId& tgt, const string& field, string& ret );

		/**
		 * Blocking 'set' call, using automatic string conversion
		 * There is a matching blocking set call with typed arguments.
		 */
		static bool strSet( const ObjId& dest, const string& field, const string& val );

		/// Sends out request for data, and awaits its return.
		static const vector< double* >* dispatchGet( 
			const ObjId& tgt, FuncId tgtFid, 
			const double* arg, unsigned int size );

		static Qinfo qi_;

	private:
		ObjId oid_;
};

class SetGet0: public SetGet
{
	public:
		SetGet0( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field )
		{
			SetGet0 sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				func->op( tgt.eref(), &qi_, 0 );
				/*
				mpiSend( fid, tgt, 0, 0 );
				*/
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			return set( dest, field );
		}
};

template< class A > class SetGet1: public SetGet
{
	public:
		SetGet1( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, A arg )
		{
			SetGet1< A > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A > conv( arg );
				func->op( tgt.eref(), &qi_, conv.ptr() );
				// mpiSend( fid, tgt, conv.ptr, conv.size() );
				return 1;
			}
			return 0;
		}

		/**
		 * I would like to provide a vector set operation. It should
		 * work in three modes: Set all targets to the same value,
		 * set targets one by one to a vector of values, and set targets
		 * one by one to randomly generated values within a range. All
		 * of these can best be collapsed into the vector assignment 
		 * operation.
		 * This variant requires that all vector entries have the same
		 * size. Strings won't work.
		 *
		 * The sequence of data in the buffer is:
		 * first few locations: ObjFid
		 * Remaining locations: Args.
		 */
		static bool setVec( Id destId, const string& field, 
			const vector< A >& arg )
		{
			if ( arg.size() == 0 ) return 0;

			ObjId tgt( destId, 0 );
			SetGet1< A > sg( tgt );
			FuncId fid;

			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A > conv( arg[0] );
				unsigned int entrySize = conv.size();
				tgt.dataId = DataId::any;
				double* data = new double[ entrySize * arg.size() ];
				double* ptr = data;

				for ( unsigned int i = 0; i < arg.size(); ++i ) {
					Conv< A > conv( arg[i] );
					assert( conv.size() == entrySize );
					memcpy( ptr, conv.ptr(), entrySize * sizeof( double ) );
					ptr += entrySize;
				}
				Element* elm = tgt.id.element();
				elm->dataHandler()->forall( func, elm, &qi_, 
					data, entrySize, arg.size() );

				delete[] data;

				return 1;
			}
			return 0;
		}

		/**
		 * Sets all target array values to the single value
		 */
		static bool setRepeat( Id destId, const string& field, 
			const A& arg )
		{
			vector< A >temp ( 1, arg );
			return setVec( destId, field, temp );
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A arg;
			Conv< A >::str2val( arg, val );
			return set( dest, field, arg );
		}
};

template< class A > class Field: public SetGet1< A >
{
	public:
		Field( const ObjId& dest )
			: SetGet1< A >( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::set( dest, temp, arg );
		}

		static bool setVec( Id destId, const string& field, 
			const vector< A >& arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setVec( destId, temp, arg );
		}

		static bool setRepeat( Id destId, const string& field, 
			A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setRepeat( destId, temp, arg );
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A arg;
			// Do NOT add 'set_' to the field name, as the 'set' func
			// does it anyway.
			Conv< A >::str2val( arg, val );
			return set( dest, field, arg );
		}

	//////////////////////////////////////////////////////////////////

		static A get( const ObjId& dest, const string& field)
		{ 
			Field< A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;
			string fullFieldName = "get_" + field;
			if ( const OpFunc* func = 
				sg.checkSet( fullFieldName, tgt, fid ) ) {
				/// Do something else if off-node.
				const GetOpFuncBase< A >* gof = 
					dynamic_cast< const GetOpFuncBase< A >* >( func );
				if ( gof )
					return gof->reduceOp( tgt.eref() );
			}
			cout << "Warning: Field::Get conversion error for " << dest.id.path() <<
				endl;

			return A();
		}

		/**
		 * Blocking call that returns a vector of values
		 * Note that the vector returned by innerGet is sparse due to
		 * the way the indexing is done. We assume
		 * all the entries have come in, and discard the holes.
		 */
		static void getVec( Id dest, const string& field, vector< A >& vec)
		{
			vec.resize( 0 );
			Field< A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;
			string fullFieldName = "get_" + field;
			if ( const OpFunc* func = 
				sg.checkSet( fullFieldName, tgt, fid ) )
			{
				const GetOpFuncBase< A >* gof = 
					dynamic_cast< const GetOpFuncBase< A >* >( func );
				if ( gof ) {
					Element* elm = dest.element();
					DataHandler* dh = elm->dataHandler();
					// This will need some serious MPI work.
					unsigned int fieldMask = dh->fieldMask();
					unsigned int size = dh->totalEntries();
					vec.resize( size );
					if ( fieldMask != 0 ) {
						assert( dh->numDimensions() > 0 );
						unsigned int maxNumEntries = dh->sizeOfDim(
							dh->numDimensions() - 1 );
						unsigned int numObj = size / maxNumEntries;
						for ( unsigned int i = 0; i < numObj; ++i ) {
							unsigned int numEntries = dh->getFieldArraySize( i );
							for ( unsigned int j = 0; j < numEntries; ++j ) {
								unsigned int k = j + i * ( fieldMask + 1 );
								DataId di( k );
								k = j + i * maxNumEntries;
								vec[ k ] = gof->reduceOp( Eref( elm, di ));
							}
						}
					} else {
						for ( unsigned int i = 0; i < size; ++i ) {
							Eref e( elm, i );
							vec[i] = gof->reduceOp( e );
						}
					}
					return;
				}
			}
			cout << "Warning: Field::getVec conversion error for " <<
				dest.path() << endl;
		}
		

		/**
		 * Instant-return call for a single value along an existing Msg.
		 * Bypasses all the queueing stuff.
		 * It is hardcoded so type safety will have to be coded in too:
		 * the dynamic_cast will catch it only at runtime.
		 */
		static A fastGet( const Eref& src, MsgId mid, FuncId fid )
		{
			const Msg* m = Msg::getMsg( mid );
			if ( m ) {
				Eref tgt = m->firstTgt( src );
				if ( tgt.element() ) {
					const GetOpFuncBase< A >* gof = 
						dynamic_cast< const GetOpFuncBase< A >* >(
						tgt.element()->cinfo()->getOpFunc( fid ) );
					if ( gof ) {
						return gof->reduceOp( tgt );
					}
				}
			}
			cout << "Warning: fastGet failed for " << src.id().path() <<
				endl;
			return A();
		}

		/**
		 * Blocking call for finding a value and returning in a
		 * string.
		 */
		static bool innerStrGet( const ObjId& dest, const string& field, 
			string& str )
		{
			Conv< A >::val2str( str, get( dest, field ) );
			return 1;
		}
};

/**
 * SetGet2 handles 2-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2 > class SetGet2: public SetGet
{
	public:
		SetGet2( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, 
			A1 arg1, A2 arg2 )
		{
			SetGet2< A1, A2 > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				unsigned int totSize = conv1.size() + conv2.size();
				double *temp = new double[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				func->op( tgt.eref(), &qi_, temp );
				delete[] temp;
				/*
				Qinfo::addDirectToQ( 
					ObjId(), tgt, 0, fid, 
					conv1.ptr(), conv1.size(),
					conv2.ptr(), conv2.size() );
				Qinfo::waitProcCycles( 1 );
				*/
				return 1;
			}
			return 0;
		}

		/**
		 * Assign a vector of targets, using matching vectors of arguments
		 * arg1 and arg2. Specifically, index i on the target receives
		 * arguments arg1[i], arg2[i].
		 * Note that there is no requirement for the size of the 
		 * argument vectors to be equal to the size of the target array
		 * of objects. If there are fewer arguments then the index cycles
		 * back, so as to tile the target array with as many arguments as
		 * we have.
		 * Need to clean up to handle string arguments later.
		 */
		static bool setVec( Id destId, const string& field, 
			const vector< A1 >& arg1, const vector< A2 >& arg2 )
		{
			if ( arg1.size() != arg2.size() || arg1.size() == 0 )
				return 0;
			ObjId tgt( destId, 0 );
			SetGet2< A1, A2 > sg( tgt );
			FuncId fid;
			// Need to do something similar for MPI.
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1[0] );
				Conv< A2 > conv2( arg2[0] );
				unsigned int entrySize = conv1.size() + conv2.size();
				tgt.dataId = DataId::any;
				double* data = new double[ entrySize * arg1.size() ];
				double* ptr = data;

				for ( unsigned int i = 0; i < arg1.size(); ++i ) {
					Conv< A1 > conv1( arg1[i] );
					Conv< A2 > conv2( arg2[i] );
					memcpy( ptr, conv1.ptr(), conv1.size() * sizeof( double ) );
					ptr += conv1.size();
					memcpy( ptr, conv2.ptr(), conv2.size() * sizeof( double ) );
					ptr += conv2.size();
				}
				Element* elm = tgt.id.element();
				elm->dataHandler()->forall( func, elm, &qi_, 
					data, entrySize, arg1.size() );
				delete[] data;
				return 1;
			}
			return 0;
		}

		/**
		 * This setVec takes a specific object entry, presumably one with
		 * an array of values within it. The it goes through each specified
		 * index and assigns the corresponding argument.
		 * This is a brute-force assignment.
		 */
		static bool setVec( ObjId dest, const string& field, 
			const vector< A1 >& arg1, const vector< A2 >& arg2 )
		{
			unsigned int max = arg1.size();
			if ( max > arg2.size() ) 
				max = arg2.size();
			bool ret = 1;
			for ( unsigned int i = 0; i < max; ++i )
				ret &= 
					SetGet2< A1, A2 >::set( dest, field, arg1[i], arg2[i] );
			return ret;
		}

		/**
		 * Blocking call using string conversion.
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A1 arg1;
			A2 arg2;
			string::size_type pos = val.find_first_of( "," );
			Conv< A1 >::str2val( arg1, val.substr( 0, pos ) );
			Conv< A2 >::str2val( arg2, val.substr( pos + 1 ) );
			return set( dest, field, arg1, arg2 );
		}

	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * LookupField handles fields that have an index arguments. Examples include
 * arrays and maps.
 * The first argument in the 'Set' is the index, the second the value.
 * The first and only argument in the 'get' is the index.
 * Here A is the type of the value, and L the lookup index.
 * 
 */
template< class L, class A > class LookupField: public SetGet2< L, A >
{
	public:
		LookupField( const ObjId& dest )
			: SetGet2< L, A >( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call. Identical to SetGet2::set.
		 */
		static bool set( const ObjId& dest, const string& field, 
			L index, A arg )
		{
			string temp = "set_" + field;
			return SetGet2< L, A >::set( dest, temp, index, arg );
		}

		/** 
		 * This setVec assigns goes through each object entry in the
		 * destId, and assigns the corresponding index and argument to it.
		 */
		static bool setVec( Id destId, const string& field, 
			const vector< L >& index, const vector< A >& arg )
		{
			string temp = "set_" + field;
			return SetGet2< L, A >::setVec( destId, temp, index, arg );
		}

		/**
		 * This setVec takes a specific object entry, presumably one with
		 * an array of values within it. The it goes through each specified
		 * index and assigns the corresponding argument.
		 * This is a brute-force assignment.
		 */
		static bool setVec( ObjId dest, const string& field, 
			const vector< L >& index, const vector< A >& arg )
		{
			string temp = "set_" + field;
			return SetGet2< L, A >::setVec( dest, temp, index, arg );
		}

		/**
		 * Faking setRepeat too. Just plugs into setVec.
		 */
		static bool setRepeat( Id destId, const string& field, 
			const vector< L >& index, A arg )
		{
			vector< A > avec( index.size(), arg );
			return setVec( destId, field, index, avec );
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& indexStr, const string& val )
		{
			L index;
			Conv< L >::str2val( index, indexStr );

			A arg;
			// Do NOT add 'set_' to the field name, as the 'set' func
			// does it anyway.
			Conv< A >::str2val( arg, val );
			return set( dest, field, index, arg );
		}

	//////////////////////////////////////////////////////////////////
		static const vector< double* >* innerGet( 
			const ObjId& dest, const string& field, L index )
		{ 
			LookupField< L, A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;

			cerr << "Error: LookupField::innerGet not yet working\n";
			assert( 0 );

			string fullFieldName = "get_" + field;
			if ( const OpFunc* func = 
				sg.checkSet( fullFieldName, tgt, fid ) ) 
			{
				const GetOpFuncBase< A >* gof = dynamic_cast< const GetOpFuncBase< A >* >( func );
				const vector< double* >* ret = 0;
				if ( gof ) {
					// Iterate over the entries here.
					;
				}
/*
				FuncId retFuncId = receiveGet()->getFid();
				Conv< FuncId > conv1( retFuncId );
				Conv< L > conv2 ( index );
				double* temp = new double[ conv1.size() + conv2.size() ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				const vector< double* >* ret = 
					SetGet::dispatchGet( tgt, fid, temp, 
						conv1.size() + conv2.size() );
				delete[] temp;

				*/
				return ret;
			}
			return 0;
		}

		/**
		 * Gets a value on a specific object, looking it up using the
		 * provided index.
		 */
		static A get( const ObjId& dest, const string& field, L index)
		{ 
			LookupField< L, A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;
			string fullFieldName = "get_" + field;
			if ( const OpFunc* func = 
				sg.checkSet( fullFieldName, tgt, fid ) ) {
				const LookupGetOpFuncBase< L, A >* gof = 
				dynamic_cast< const LookupGetOpFuncBase< L, A >* >( func );
				if ( gof )
					return gof->reduceOp( tgt.eref(), index );
			}
			cout << "Warning: LookupField::Get conversion error for " << 
				dest.id.path() << endl;
			return A();
		}

		/**
		 * Blocking call that returns a vector of values in vec.
		 * This variant goes through each target object entry on dest,
		 * and passes in the same lookup index to each one. The results
		 * are put together in the vector vec.
		 * As the vector returned from the innerGet command is sparse but
		 * ordered, we compact it to build up the return vector.
		 */
		static void getVec( Id dest, const string& field, 
			vector< L >& index, vector< A >& vec )
		{
			ObjId tgt( dest, DataId::any );
			const vector< double* >* ret = 
				innerGet( tgt, field, index, ret );
			vec.resize( 0 );
			if ( ret ) {
				vec.resize( ret->size() );
				for ( unsigned int i = 0; i < ret->size(); ++i ) {
					Conv< A > conv( (*ret)[i] );
					vec[i] = *conv;
					// vec.push_back( *conv );
				}
			}
		}

		/**
		 * Blocking virtual call for finding a value and returning in a
		 * string.
		 */
		static bool innerStrGet( const ObjId& dest, const string& field, 
			const string& indexStr, string& str )
		{
			L index;
			Conv< L >::str2val( index, indexStr );

			A ret = get( dest, field, index );
			Conv<A>::val2str( str, ret );
			return 1;
		}
};

/**
 * SetGet3 handles 3-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3 > class SetGet3: public SetGet
{
	public:
		SetGet3( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3 )
		{
			SetGet3< A1, A2, A3 > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				unsigned int totSize = 
					conv1.size() + conv2.size() + conv3.size();
				double *temp = new double[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				conv3.val2buf( temp + conv1.size() + conv2.size() );

				func->op( tgt.eref(), &qi_, temp );
				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A1 arg1;
			A2 arg2;
			A3 arg3;
			string::size_type pos = val.find_first_of( "," );
			Conv< A1 >::str2val( arg1, val.substr( 0, pos ) );
			string temp = val.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A2 >::str2val( arg2, temp.substr( 0,pos ) );
			Conv< A3 >::str2val( arg3, temp.substr( pos + 1 ) );
			return set( dest, field, arg1, arg2, arg3 );
		}

		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet4 handles 4-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4 > class SetGet4: public SetGet
{
	public:
		SetGet4( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4 )
		{
			SetGet4< A1, A2, A3, A4 > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				unsigned int totSize = 
					conv1.size() + conv2.size() + 
					conv3.size() + conv4.size();
				double* temp = new double[ totSize ];
				double* ptr = temp;
				conv1.val2buf( ptr ); ptr += conv1.size();
				conv2.val2buf( ptr ); ptr += conv2.size();
				conv3.val2buf( ptr ); ptr += conv3.size();
				conv4.val2buf( ptr );
				func->op( tgt.eref(), &qi_, temp );

				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			string::size_type pos = val.find_first_of( "," );
			Conv< A1 >::str2val( arg1, val.substr( 0, pos ) );
			string temp = val.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A2 >::str2val( arg2, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A3 >::str2val( arg3, temp.substr( 0, pos ) );
			Conv< A4 >::str2val( arg4, temp.substr( pos + 1 ) );
			return set( dest, field, arg1, arg2, arg3, arg4 );
		}

	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet5 handles 5-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4, class A5 > class SetGet5:
	public SetGet
{
	public:
		SetGet5( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4, A5 arg5 )
		{
			SetGet5< A1, A2, A3, A4, A5 > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				Conv< A5 > conv5( arg5 );

				unsigned int totSize = 
					conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() + conv5.size();
				double* temp = new double[ totSize ];
				double* ptr = temp;
				conv1.val2buf( ptr ); ptr += conv1.size();
				conv2.val2buf( ptr ); ptr += conv2.size();
				conv3.val2buf( ptr ); ptr += conv3.size();
				conv4.val2buf( ptr ); ptr += conv4.size();
				conv5.val2buf( ptr );
				func->op( tgt.eref(), &qi_, temp );

				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			A5 arg5;
			string::size_type pos = val.find_first_of( "," );
			Conv< A1 >::str2val( arg1, val.substr( 0, pos ) );
			string temp = val.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A2 >::str2val( arg2, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A3 >::str2val( arg3, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A4 >::str2val( arg4, temp.substr( 0, pos ) );
			Conv< A5 >::str2val( arg5, temp.substr( pos + 1 ) );
			return set( dest, field, arg1, arg2, arg3, arg4, arg5 );
		}
	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet6 handles 6-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4, class A5, class A6 > class SetGet6:
	public SetGet
{
	public:
		SetGet6( const ObjId& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const ObjId& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4, A5 arg5, A6 arg6 )
		{
			SetGet6< A1, A2, A3, A4, A5, A6 > sg( dest );
			FuncId fid;
			ObjId tgt( dest );
			if ( const OpFunc* func = sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				Conv< A5 > conv5( arg5 );
				Conv< A6 > conv6( arg6 );

				unsigned int totSize = 
					conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() + conv5.size();
				double* temp = new double[ totSize ];
				double* ptr = temp;
				conv1.val2buf( ptr ); ptr += conv1.size();
				conv2.val2buf( ptr ); ptr += conv2.size();
				conv3.val2buf( ptr ); ptr += conv3.size();
				conv4.val2buf( ptr ); ptr += conv4.size();
				conv5.val2buf( ptr ); ptr += conv5.size();
				conv6.val2buf( ptr );
				func->op( tgt.eref(), &qi_, temp );

				delete[] temp;

				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		static bool innerStrSet( const ObjId& dest, const string& field, 
			const string& val )
		{
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			A5 arg5;
			A6 arg6;
			string::size_type pos = val.find_first_of( "," );
			Conv< A1 >::str2val( arg1, val.substr( 0, pos ) );
			string temp = val.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A2 >::str2val( arg2, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A3 >::str2val( arg3, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A4 >::str2val( arg4, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			pos = temp.find_first_of( "," );
			Conv< A5 >::str2val( arg5, temp.substr( 0, pos ) );
			Conv< A6 >::str2val( arg6, temp.substr( pos + 1 ) );
			return set( dest, field, arg1, arg2, arg3, arg4, arg5, arg6 );
		}
	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

#endif // _SETGET_H
