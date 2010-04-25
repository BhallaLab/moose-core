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
		SetGet( const Eref& e )
			: e_( e )
		{;}

		virtual ~SetGet()
		{;}

		/**
		 * Checks arg # and types for a 'set' call. Can be zero to 3 args.
		 * Returns true if good. Passes back found fid.
		 * Utility function to check that the target field matches this
		 * source type, and to look up and pass back the fid.
		 */
		bool checkSet( const string& field, FuncId& fid ) const;

		/**
		 * Checks arg # and type for a 'get' call. Single argument.
		 * Returns true if good. Passes back found fid.
		 * Utility function to check two things: 
		 * that the dest field matches the type of this request, and that
		 * the dest funcId will accept a funcId argument for the
		 * callback.
		bool checkGet( const string& field, FuncId& fid ) const;
		 */
//////////////////////////////////////////////////////////////////////
		/**
		 * Initiate a nonblocking 'get' call.
		 * This can be harvested either using harvestStrGet or
		 * harvestGet< Type >.
		bool iGet( const string &field ) const;
		 */

		/**
		 * Complete a nonblocking 'get' call, returning a string.
		 * There is also a nonblocking typed counterpart, harvestGet< T >.
		virtual string harvestStrGet() const = 0;
		 */
//////////////////////////////////////////////////////////////////////
		/**
		 * Blocking 'get' call, returning into a string.
		 * There is a matching 'get<T> call, returning appropriate type.
		 */
		static string strGet( const Eref& dest, const string& field);

		/**
		 * Blocking 'set' call, using automatic string conversion
		 * There is a matching blocking set call with typed arguments.
		 */
		static bool strSet( const Eref& dest, const string& field, const string& val );

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		virtual bool iStrSet( const string& field, const string& val ) = 0;
		 */
		
		/**
		 * Waits for completion of a nonblocking 'set' call, either
		 * string or typed versions.
		 * Can be skipped if there is an absolute guarantee that there
		 * won't be dependencies between the 'set' and subsequent calls,
		 * till the next completeSet or harvestGet call.
		 * Avoid using. If you have dependencies then use the blocking set.
		 */
		void completeSet() const;

		/**
		 * Resizes the internal buffer on the SetGet
		 */
		void resizeBuf( unsigned int size );

		char* buf();

		/**
		 * Utility function for the init() function
		static void setShell();
		 */

	protected:
		/**
		 * Puts data into target queue for calling functions and setting
		 * fields. This is a common core function used by the various
		 * type-specialized variants.
		void iSetInner( const FuncId fid, const char* buf,
			unsigned int size );
		 */
		// void clearQ() const;
	private:
		// static Eref shelle_;
		// static Element* shell_;
		// Should have something - baton - to identify the specific set/get
		// call here.
		Eref e_;
		// vector< char > buf_;
};

class SetGet0: public SetGet
{
	public:
		SetGet0( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		bool set( const Eref& dest, const string& field ) const
		{
			SetGet0 sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
				Shell::dispatchSet( dest, fid, "", 0 );
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
		bool strSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			return set( dest, field );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		bool iSet( const string& field )
		{
			FuncId fid;
			if ( checkSet( field, fid ) ) {
				iSetInner( fid, "", 0 );
				return 1;
			}
			return 0;
		}
		 */

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		bool iStrSet( const string& field, const string& val )
		{
			return iSet( field );
		}
		 */
	//////////////////////////////////////////////////////////////////

		void harvestGet() const
		{ 
			;
		}

		string harvestStrGet() const
		{ 
			return "";
		}

		string strGet( const Eref& dest, const string& field) const
		{ 
			return "";
		}
		
};

template< class A > class SetGet1: public SetGet
{
	public:
		SetGet1( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, A arg )
		{
			SetGet1< A > sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
				Conv< A > conv( arg );
				// unsigned int size = Conv< A >::size( arg );
				char *temp = new char[ conv.size() ];
				conv.val2buf( temp );
				Shell::dispatchSet( dest, fid, temp, conv.size() );
				/*
				sg.iSetInner( fid, temp, conv.size() );

				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
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
		static bool setVec( const Eref& dest, const string& field, 
			const vector< A >& arg )
		{
			SetGet1< A > sg( dest );
			FuncId fid;
			if ( arg.size() == 0 )
				return 0;

			if ( sg.checkSet( field, fid ) ) {
				const char* data = reinterpret_cast< const char* >( &arg[0] );
				PrepackedBuffer pb( data, arg.size() * sizeof( A ), 
					arg.size() ) ;
				Shell::dispatchSetVec( dest, fid, pb );
				return 1;
			}
			return 0;
		}

		/**
		 * Sets all target array values to the single value
		 */
		static bool setRepeat( const Eref& dest, const string& field, 
			const A& arg )
		{
			vector< A >temp ( 1, arg );
			return setVec( dest, field, temp );
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			A arg;
			str2val( arg, val );
			return set( dest, field, arg );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		bool iSet( const string& field, const A& arg )
		{
			FuncId fid;
			if ( checkSet( field, fid ) ) {
				Conv< A > conv( arg );
				resizeBuf( conv.size() );
				char *temp = buf();
				conv.val2buf( temp );
				iSetInner( fid, temp, conv.size() );
				return 1;
			}
			return 0;
		}
		 */

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			A temp;
			Conv< A >::str2val( temp, val );
			return iSet( field, temp );
		}
	//////////////////////////////////////////////////////////////////
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

template< class A > class Field: public SetGet1< A >
{
	public:
		Field( const Eref& dest )
			: SetGet1< A >( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::set( dest, temp, arg );
		}

		static bool setVec( const Eref& dest, const string& field, 
			const vector< A >& arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setVec( dest, temp, arg );
		}

		static bool setRepeat( const Eref& dest, const string& field, 
			A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setRepeat( dest, temp, arg );
		}

		/**
		 * Blocking call using string conversion
		 */
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			A arg;
			str2val( arg, val );
			return set( dest, field, arg );
		}

	//////////////////////////////////////////////////////////////////

		/**
		 * Blocking call using typed values
		static A get( Eref& dest, const string& field)
		{ 
			string temp = "get_" + field;
			return SetGet1< A >::get( dest, temp );
		}
		 */

		/**
		 * Blocking call using string conversion
		static string strGet( Eref& dest, const string& field)
		{ 
			string temp = "get_" + field;
			SetGet1< A > sg( dest );
			 if ( sg.iGet( temp ) )
			 	return sg.harvestStrGet();
		}
		 */

		/**
		 * Terminating call using typed values
		 * Again, need to do a lot more checking in threaded and in 
		 * multinode calls. Probably want to record a 'baton' to know
		 * when the call ends.
		 */
#if 0
		A harvestGet() const
		{ 
			/**
			 * First clearQ is so that the target Element receives the
			 * request.
			 * Issue remains: which thread to put this call in. For now
			 * assume thread 0.
			 */
			Qinfo::clearQ( Shell::procInfo() );
			/**
			 * Second clearQ is so that the data can come back to the Shell
			 */
			Qinfo::clearQ( Shell::procInfo() );
			Conv< A > ret( Shell::buf() );
			return *ret;
		}

		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			string s;
			Conv< A >::val2str( s, harvestGet() );
			return s;
		}
#endif
		/**
		 * Blocking call using typed values
		 */
		static A get( const Eref& dest, const string& field)
		{ 
			SetGet1< A > sg( dest );
			const char* ret = Shell::dispatchGet( dest, field, &sg );
			Conv< A > conv( ret );
			return *conv;
		}

		/**
		 * Blocking call using string conversion
		static string strGet( Eref& dest, const string& field)
		{ 
			Field< A > sg( dest );
			 if ( sg.iGet( field ) )
			 	return sg.harvestStrGet();
		}
		 */
};

/**
 * SetGet2 handles 2-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2 > class SetGet2: public SetGet
{
	public:
		SetGet2( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2 )
		{
			SetGet2< A1, A2 > sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				char *temp = new char[ conv1.size() + conv2.size() ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				Shell::dispatchSet( dest, fid, temp, 
					conv1.size() + conv2.size() );

				/*
				sg.iSetInner( fid, temp, conv1.size() + conv2.size() );

				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
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
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			cout << "strSet< A1, A2 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2 );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		 */
		bool iSet( const string& field, const A1& arg1, const A2& arg2 )
		{
			/*
			FuncId fid;
			if ( checkSet( field, fid ) ) {
				unsigned int size1 = Conv< A1 >::size( arg1 );
				unsigned int size2 = Conv< A2 >::size( arg2 );
				resizeBuf( size1 + size2 );
				char *temp = buf();
				Conv< A1 >::val2buf( temp, arg1 );
				Conv< A2 >::val2buf( temp + size1, arg2 );
				iSetInner( fid, temp, size1 + size2 );
				return 1;
			}
			*/
			return 0;
		}

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			/*
			cout << "iStrSet< A1, A2 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			Conv< A1 >::str2val( arg1, val );
			return iSet( field, arg1, arg2 );
			*/
			return 0;
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
 * SetGet3 handles 3-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3 > class SetGet3: public SetGet
{
	public:
		SetGet3( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3 )
		{
			SetGet3< A1, A2, A3 > sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				unsigned int s1 = conv1.size();
				unsigned int s1s2 = s1 + conv2.size();
				unsigned int totSize = s1s2 + conv3.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + s1 );
				conv3.val2buf( temp + s1s2 );
				Shell::dispatchSet( dest, fid, temp, totSize );

				/*
				sg.iSetInner( fid, temp, totSize );
				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
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
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			cout << "strSet< A1, A2, A3 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3 );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		 */
		bool iSet( const string& field, const A1& arg1, const A2& arg2, const A3& arg3 )
		{
			/*
			FuncId fid;
			if ( checkSet( field, fid ) ) {
				unsigned int size1 = Conv< A1 >::size( arg1 );
				unsigned int size2 = Conv< A2 >::size( arg2 );
				unsigned int size3 = Conv< A3 >::size( arg3 );
				unsigned int totSize = size1 + size2 + size3;
				resizeBuf( totSize );
				char *temp = buf();
				Conv< A1 >::val2buf( temp, arg1 );
				Conv< A2 >::val2buf( temp + size1, arg2 );
				Conv< A2 >::val2buf( temp + size1 + size2, arg3 );
				iSetInner( fid, temp, totSize );
				return 1;
			}
			*/
			return 0;
		}

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			/*
			cout << "iStrSet< A1, A2, A3 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			Conv< A1 >::str2val( arg1, val );
			return iSet( field, arg1, arg2, arg3 );
			*/
			return 0;
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
 * SetGet4 handles 4-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4 > class SetGet4: public SetGet
{
	public:
		SetGet4( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4 )
		{
			SetGet4< A1, A2, A3, A4 > sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
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
				Shell::dispatchSet( dest, fid, temp, totSize );

				/*
				sg.iSetInner( fid, temp, totSize );
				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
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
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			cout << "strSet< A1, A2, A3, A4 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3, arg4 );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		 */
		bool iSet( const string& field, const A1& arg1, const A2& arg2, const A3& arg3, const A4& arg4 )
		{
		/*
			FuncId fid;
			if ( checkSet( field, fid ) ) {
				unsigned int size1 = Conv< A1 >::size( arg1 );
				unsigned int size2 = Conv< A2 >::size( arg2 );
				unsigned int size3 = Conv< A3 >::size( arg3 );
				unsigned int size4 = Conv< A4 >::size( arg4 );
				unsigned int totSize = size1 + size2 + size3 + size4;
				resizeBuf( totSize );
				char *temp = buf();
				Conv< A1 >::val2buf( temp, arg1 );
				Conv< A2 >::val2buf( temp + size1, arg2 );
				Conv< A3 >::val2buf( temp + size1 + size2, arg3 );
				Conv< A4 >::val2buf( temp + size1 + size2 + size3, arg4 );
				iSetInner( fid, temp, totSize );
				return 1;
			}
			*/
			return 0;
		}

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			/*
			cout << "iStrSet< A1, A2, A3, A4 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			Conv< A1 >::str2val( arg1, val );
			return iSet( field, arg1, arg2, arg3, arg4 );
			*/
			return 0;
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
		SetGet5( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4, A5 arg5 )
		{
			SetGet5< A1, A2, A3, A4, A5 > sg( dest );
			FuncId fid;
			if ( sg.checkSet( field, fid ) ) {
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
				Shell::dispatchSet( dest, fid, temp, totSize );

				/*
				sg.iSetInner( fid, temp, totSize );
				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
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
		static bool strSet( const Eref& dest, const string& field, 
			const string& val )
		{
			cout << "strSet< A1, A2, A3, A4, A5 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			A5 arg5;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3, arg4, arg5 );
		}

		/**
		 * Nonblocking 'set' call, no args.
		 * There is a matching nonblocking iStrSet call with a string arg.
		 */
		bool iSet( const string& field, const A1& arg1, const A2& arg2, const A3& arg3, const A4& arg4, const A5& arg5 )
		{
			return 0;
		}

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			return 0;
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
