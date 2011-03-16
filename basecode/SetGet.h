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

class SetGet
{
	public:
		SetGet( const ObjId& oid )
			: oid_( oid )
		{;}

		virtual ~SetGet()
		{;}

		/**
		 * Assigns 'field' on 'tgt' to 'val', after doing necessary type
		 * conversion from the string val. Returns 0 if it can't
		 * handle it, which is unusual.
		bool innerStrSet( const ObjId& tgt, 
			const string& field, const string& val ) const = 0;
		 */

		/**
		 * Looks up field value on tgt, converts to string, and puts on 
		 * 'ret'. Returns 0 if the class does not support 'get' operations,
		 * which is usually the case. Fields are the exception.
		virtual bool innerStrGet( const ObjId& tgt, 
			const string& field, string& ret ) const {
			return 0;
		}
		 */


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
		unsigned int checkSet( 
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

		
		/**
		 * Waits for completion of a nonblocking 'set' call, either
		 * string or typed versions.
		 * Can be skipped if there is an absolute guarantee that there
		 * won't be dependencies between the 'set' and subsequent calls,
		 * till the next completeSet or harvestGet call.
		 * Avoid using. If you have dependencies then use the blocking set.
		 */
		void completeSet() const;

	
		/// Adapter function, just forwards to Shell::dispatchSet
		static void dispatchSet( const ObjId& oid, FuncId fid, 
			const char* args, unsigned int size );

		/// Adapter function, just forwards to Shell::dispatchSetVec
		static void dispatchSetVec( const ObjId& oid, FuncId fid, 
			const PrepackedBuffer& arg );

		/// Adapter function, just forwards to Shell::dispatchGet
		static const vector< char* >& dispatchGet( 
			const ObjId& oid, FuncId fid,
			const PrepackedBuffer& arg );

		/*
		static const vector< char* >& dispatchGetVec( 
			const ObjId& oid, FuncId fid,
			const char* args, unsigned int size );
			*/

		/// Adapter function, forwards to Shell::dispatchLookupGet
		/*
		static const vector< char* >& dispatchLookupGet( 
			const ObjId& oid, const string& field,
			char* indexBuf, const SetGet* sg, 
			unsigned int& numGetEntries );
			*/

		///  char* buf();

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
			if ( sg.checkSet( field, tgt, fid ) ) {
				dispatchSet( tgt, fid, "", 0 );
				/*
				sg.iSetInner( fid, "", 0 );

				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
				return 1;
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A > conv( arg );
				char *temp = new char[ conv.size() ];
				conv.val2buf( temp );
				dispatchSet( tgt, fid, temp, conv.size() );
				delete[] temp;
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
		 */
		static bool setVec( Id destId, const string& field, 
			const vector< A >& arg )
		{
			ObjId tgt( destId, 0 );
			SetGet1< A > sg( tgt );
			FuncId fid;
			if ( arg.size() == 0 )
				return 0;

			if ( sg.checkSet( field, tgt, fid ) ) {
				const char* data = reinterpret_cast< const char* >( &arg[0] );
				PrepackedBuffer pb( data, arg.size() * sizeof( A ), 
					arg.size() ) ;
				dispatchSetVec( tgt, fid, pb );
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

		/**
		 * Blocking call using typed values
		 */
		static A get( const ObjId& dest, const string& field)
		{ 
			Field< A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;

			string fullFieldName = "get_" + field;
			if ( sg.checkSet( fullFieldName, tgt, fid ) ) {
				FuncId retFuncId = receiveGet()->getFid();
				PrepackedBuffer buf( 
					reinterpret_cast< const char* >( &retFuncId ), 
					sizeof( FuncId ) );
				const vector< char* >& ret = 
					SetGet::dispatchGet( tgt, fid, buf );
				if ( ret.size() == 1 ) {
					Conv< A > conv( ret[0] );
					return *conv;
				}
			}
			return A();
			/*
			Conv< FuncId > args( retFuncId );
			char *temp = new char[ args.size() ];
			args.val2buf( temp );
			const char* ret = dispatchGet( &sg, dest, fullFieldName, buf, 
				args.size() );
			delete[] temp;
			*/
		}

		/**
		 * Blocking call that returns a vector of values
		 */
		static void getVec( Id dest, const string& field, vector< A >& vec )
		{
			Field< A > sg( ObjId( dest, 0 ) );
			ObjId tgt( dest );
			FuncId fid;

			string fullFieldName = "get_" + field;
			unsigned int numRetEntries = 
				sg.checkSet( fullFieldName, tgt, fid );
			if ( numRetEntries > 0 ) {
				FuncId retFuncId = receiveGet()->getFid();
				vector< FuncId > fidVec( numRetEntries, retFuncId );

				PrepackedBuffer pb( 
					reinterpret_cast< const char* >( &fidVec[0] ), 
					numRetEntries * sizeof( FuncId ), numRetEntries );

				const vector< char* >& ret = SetGet::dispatchGet( 
					dest, fid, pb );

				assert( ret.size() == numRetEntries );
				vec.resize( numRetEntries );
				for ( unsigned int i = 0; i < numRetEntries; ++i ) {
					Conv< A > conv( ret[i] );
					vec[i] = *conv;
				}
				return;
			}
			vec.resize( 0 );
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
			/*
			SetGet1< A > sg( dest );
			string temp = "get_" + field;
			unsigned int numEntries = 0;
			const vector< char* > & ret = 
				dispatchGet( dest, temp, &sg, numEntries );
			assert( numEntries == 1 );
			Conv< A > conv( ret[0] );
			Conv<A>::val2str( str, *conv );
			return 1;
			*/
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				char *temp = new char[ conv1.size() + conv2.size() ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				dispatchSet( tgt, fid, temp, 
					conv1.size() + conv2.size() );
				delete[] temp;
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
			ObjId tgt( destId, 0 );
			SetGet2< A1, A2 > sg( tgt );
			FuncId fid;
			if ( arg1.size() == 0 || arg1.size() != arg2.size() )
				return 0;
			unsigned int totalSize = 
				arg1.size() * ( sizeof( A1 ) + sizeof( A2 ) );
			if ( sg.checkSet( field, tgt, fid ) ) {
				char* data = new char[ totalSize ];
				char* temp = data;
				for ( unsigned int i = 0; i < arg1.size(); ++i ) {
					memcpy( temp, &arg1[i], sizeof( A1 ) );
					temp += sizeof( A1 );
					memcpy( temp, &arg2[i], sizeof( A2 ) );
					temp += sizeof( A2 );
				}
				PrepackedBuffer pb( data, totalSize, arg1.size() );
				dispatchSetVec( tgt, fid, pb );
				delete[] data;
				return 1;
			}
			return 0;
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
			: LookupField< L, A >( dest )
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
		 * We're faking setVec. There isn't really a good matching
		 * operation, nor do I feel it is likely to be needed often,
		 * so it is done by marching through all the
		 * individual assignments.
		 */
		static bool setVec( Id destId, const string& field, 
			const vector< L >& index, const vector< A >& arg )
		{
			string temp = "set_" + field;
			unsigned int max = index.size();
			if ( max > arg.size() ) 
				max = arg.size();
			bool ret = 1;
			for ( unsigned int i = 0; i < max; ++i )
				ret &= 
					SetGet2< L, A >::set( destId, temp, index[i], arg[i] );
			return ret;
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

		/**
		 * Blocking call using typed values
		static A get( const ObjId& dest, const string& field, L index )
		{ 
			SetGet1< A > sg( dest );
			Conv< L > convL( index );
			char *indexBuf = new char[ convL.size() ];
			convL.val2buf( indexBuf );

			string temp = "get_" + field;
			unsigned int numRetEntries = 0;
			const vector< char* >& ret = 
				dispatchLookupGet( dest, temp, indexBuf, &sg, numRetEntries );
			assert( numRetEntries == 1 );
			Conv< A > conv( ret[0] );
			return *conv;
		}
		 */

		/**
		 * Another variant
		 */
		static A get( const ObjId& dest, const string& field, L index)
		{ 
			Field< A > sg( dest );
			ObjId tgt( dest );
			FuncId fid;

			string fullFieldName = "get_" + field;
			if ( sg->checkSet( fullFieldName, tgt, fid ) ) {
				FuncId retFuncId = receiveGet()->getFid();
				Conv< FuncId > conv1( retFuncId );
				Conv< L > conv2 ( index );
				char* temp = new char[ conv1.size() + conv2.size() ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );

				const vector< char* >& ret = 
					dispatchGet( dest, fid, 
					temp, conv1.size() + conv2.size() );

				if ( ret.size() == 1 ) {
					Conv< A > conv( ret[0] );
					return *conv;
				}
			}
			return A();
		}

		/**
		 * Blocking call that returns a vector of values
		 */
		static void getVec( Id dest, const string& field, 
			vector< L >& index, vector< A >& vec )
		{
			SetGet1< A > sg( ObjId( dest, 0 ) );
			Conv< vector< L > > convL( index );
			char *indexBuf = new char[ convL.size() ];
			convL.val2buf( indexBuf );

			string temp = "get_" + field;
			unsigned int numRetEntries;
			const vector< char* >& ret = 
				dispatchLookupGet( ObjId( dest, DataId::any() ), 
					temp, indexBuf, 
					&sg, numRetEntries );

			vec.resize( numRetEntries );
			for ( unsigned int i = 0; i < numRetEntries; ++i ) {
				Conv< A > conv( ret[i] );
				vec[i] = *conv;
			}
		}

		/**
		 * Blocking virtual call for finding a value and returning in a
		 * string.
		 */
		static bool innerStrGet( const ObjId& dest, const string& field, 
			const string& indexStr, string& str )
		{
			SetGet1< A > sg( dest );
			L index;
			Conv< L >::str2val( index, indexStr );
			Conv< L > convL( index );
			char *indexBuf = new char[ convL.size() ];
			convL.val2buf( indexBuf );

			string temp = "get_" + field;
			unsigned int numRetEntries = 0;
			const vector< char* > & ret = 
				dispatchLookupGet( dest, temp, indexBuf, &sg, numRetEntries );
			assert( numRetEntries == 1 );
			Conv< A > conv( ret[0] );
			Conv<A>::val2str( str, *conv );
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				unsigned int totSize = conv1.size() + conv2.size() + conv3.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				conv3.val2buf( temp + conv1.size() + conv2.size() );
				dispatchSet( tgt, fid, temp, totSize );
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				unsigned int s1 = conv1.size();
				unsigned int s1s2 = s1 + conv2.size();
				unsigned int s1s2s3 = s1s2 + conv3.size();
				unsigned int totSize = s1s2s3 + conv4.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + s1 );
				conv3.val2buf( temp + s1s2 );
				conv4.val2buf( temp + s1s2s3 );
				dispatchSet( tgt, fid, temp, totSize );
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				Conv< A5 > conv5( arg5 );
				unsigned int totSize = conv1.size() + conv2.size() +
					conv3.size() + conv4.size() + conv5.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				conv3.val2buf( temp + conv1.size() + conv2.size() );
				conv4.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() );
				conv5.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() );
				dispatchSet( tgt, fid, temp, totSize );

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
			Conv< A3 >::str2val( arg3, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
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
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				Conv< A5 > conv5( arg5 );
				Conv< A6 > conv6( arg6 );
				unsigned int totSize = conv1.size() + conv2.size() +
					conv3.size() + conv4.size() + conv5.size() + conv6.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				conv3.val2buf( temp + conv1.size() + conv2.size() );
				conv4.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() );
				conv5.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() );
				conv6.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() + conv5.size() );
				dispatchSet( tgt, fid, temp, totSize );

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
			Conv< A3 >::str2val( arg3, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
			Conv< A4 >::str2val( arg4, temp.substr( 0, pos ) );
			temp = temp.substr( pos + 1 );
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
