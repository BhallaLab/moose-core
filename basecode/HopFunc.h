/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HOP_FUNC_H
#define _HOP_FUNC_H

double* addToBuf( 
			const Eref& e, HopIndex hopIndex, unsigned int size );
void dispatchBuffers( const Eref& e, HopIndex hopIndex );
double* remoteGet( const Eref& e , unsigned int bindIndex );

/**
 * Function to hop across nodes. This one has no arguments, just tells the
 * remote object that an event has occurred.
 */
class HopFunc0: public OpFunc0Base
{
	public:
		HopFunc0( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}
		void op( const Eref& e ) const
		{
			addToBuf( e, hopIndex_, 0 );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

// Function to hop across nodes, with one argument.
template < class A > class HopFunc1: public OpFunc1Base< A >
{
	public:
		HopFunc1( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}
		void op( const Eref& e, A arg ) const
		{
			double* buf = addToBuf( e, hopIndex_, Conv< A >::size( arg ) );
			Conv< A >::val2buf( arg, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

/**
 * Deferred specification of function from OpFunc1Base, so it is after 
 * the declaration of the HopFunc class to which it refers.
 */
template< class A > 
const OpFunc* OpFunc1Base< A >::makeHopFunc( HopIndex hopIndex ) const
{
	return new HopFunc1< A >( hopIndex );
}

// Function to hop across nodes, with two arguments.
template < class A1, class A2 > class HopFunc2: public OpFunc2Base< A1, A2 >
{
	public:
		HopFunc2( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}
		void op( const Eref& e, A1 arg1, A2 arg2 ) const
		{
			double* buf = addToBuf( e, hopIndex_, 
				Conv< A1 >::size( arg1 ) + Conv< A2 >::size( arg2 ) );
			/*
			Conv< A1 >::val2buf( arg1, buf );
			Conv< A2 >::val2buf( arg2, buf + Conv< A1 >.size( arg1 ) );
			or
			buf = Conv< A1 >.val2buf( arg1, buf );
			Conv< A2 >::val2buf( arg2, buf );
			or 
			*/
			Conv< A1 >::val2buf( arg1, &buf );
			Conv< A2 >::val2buf( arg2, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

template< class A1, class A2 > 
const OpFunc* OpFunc2Base< A1, A2 >::makeHopFunc( 
				HopIndex hopIndex) const 
{
	return new HopFunc2< A1, A2 >( hopIndex );
}

// Function to hop across nodes, with three arguments.
template < class A1, class A2, class A3 > class HopFunc3: 
		public OpFunc3Base< A1, A2, A3 >
{
	public:
		HopFunc3( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}

		void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3 ) const
		{
			double* buf = addToBuf( e, hopIndex_, 
				Conv< A1 >::size( arg1 ) + Conv< A2 >::size( arg2 ) +
				Conv< A3 >::size( arg3 ) );
			Conv< A1 >::val2buf( arg1, &buf );
			Conv< A2 >::val2buf( arg2, &buf );
			Conv< A3 >::val2buf( arg3, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

template< class A1, class A2, class A3 > 
const OpFunc* OpFunc3Base< A1, A2, A3 >::makeHopFunc( 
				HopIndex hopIndex) const 
{
	return new HopFunc3< A1, A2, A3 >( hopIndex );
}

// Function to hop across nodes, with three arguments.
template < class A1, class A2, class A3, class A4 > class HopFunc4: 
		public OpFunc4Base< A1, A2, A3, A4 >
{
	public:
		HopFunc4( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}

		void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3, A4 arg4 ) const
		{
			double* buf = addToBuf( e, hopIndex_, 
				Conv< A1 >::size( arg1 ) + Conv< A2 >::size( arg2 ) +
				Conv< A3 >::size( arg3 ) + Conv< A4 >::size( arg4 ) );
			Conv< A1 >::val2buf( arg1, &buf );
			Conv< A2 >::val2buf( arg2, &buf );
			Conv< A3 >::val2buf( arg3, &buf );
			Conv< A4 >::val2buf( arg4, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

template< class A1, class A2, class A3, class A4 > 
const OpFunc* OpFunc4Base< A1, A2, A3, A4 >::makeHopFunc( 
				HopIndex hopIndex) const 
{
	return new HopFunc4< A1, A2, A3, A4 >( hopIndex );
}

// Function to hop across nodes, with three arguments.
template < class A1, class A2, class A3, class A4, class A5 >
	class HopFunc5: public OpFunc5Base< A1, A2, A3, A4, A5 >
{
	public:
		HopFunc5( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}

		void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3, 
						A4 arg4, A5 arg5 ) const
		{
			double* buf = addToBuf( e, hopIndex_, 
				Conv< A1 >::size( arg1 ) + Conv< A2 >::size( arg2 ) +
				Conv< A3 >::size( arg3 ) + Conv< A4 >::size( arg4 ) +
				Conv< A5 >::size( arg5 ) );
			Conv< A1 >::val2buf( arg1, &buf );
			Conv< A2 >::val2buf( arg2, &buf );
			Conv< A3 >::val2buf( arg3, &buf );
			Conv< A4 >::val2buf( arg4, &buf );
			Conv< A5 >::val2buf( arg5, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

template< class A1, class A2, class A3, class A4, class A5 > 
const OpFunc* OpFunc5Base< A1, A2, A3, A4, A5 >::makeHopFunc( 
				HopIndex hopIndex) const 
{
	return new HopFunc5< A1, A2, A3, A4, A5 >( hopIndex );
}

// Function to hop across nodes, with three arguments.
template < class A1, class A2, class A3, class A4, class A5, class A6 >
	class HopFunc6: public OpFunc6Base< A1, A2, A3, A4, A5, A6 >
{
	public:
		HopFunc6( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}

		void op( const Eref& e, A1 arg1, A2 arg2, A3 arg3, 
						A4 arg4, A5 arg5, A6 arg6 ) const
		{
			double* buf = addToBuf( e, hopIndex_, 
				Conv< A1 >::size( arg1 ) + Conv< A2 >::size( arg2 ) +
				Conv< A3 >::size( arg3 ) + Conv< A4 >::size( arg4 ) +
				Conv< A5 >::size( arg5 ) + Conv< A6 >::size( arg6 ) );
			Conv< A1 >::val2buf( arg1, &buf );
			Conv< A2 >::val2buf( arg2, &buf );
			Conv< A3 >::val2buf( arg3, &buf );
			Conv< A4 >::val2buf( arg4, &buf );
			Conv< A5 >::val2buf( arg5, &buf );
			Conv< A6 >::val2buf( arg6, &buf );
			dispatchBuffers( e, hopIndex_ );
		}
	private:
		HopIndex hopIndex_;
};

template< class A1, class A2, class A3, class A4, class A5, class A6 > 
const OpFunc* OpFunc6Base< A1, A2, A3, A4, A5, A6 >::makeHopFunc( 
				HopIndex hopIndex) const 
{
	return new HopFunc6< A1, A2, A3, A4, A5, A6 >( hopIndex );
}


// Function to Get value after hop across nodes, with one argument.
template < class A > class GetHopFunc: public OpFunc1Base< A* >
{
	public:
		GetHopFunc( HopIndex hopIndex )
				: hopIndex_( hopIndex )
		{;}
		void op( const Eref& e, A* ret ) const
		{
			double* buf = remoteGet( e, hopIndex_.bindIndex() );
			*ret = Conv< A >::buf2val( &buf );
		}
	private:
		HopIndex hopIndex_;
};

/**
 * Deferred specification of function from OpFunc1Base, so it is after 
 * the declaration of the HopFunc class to which it refers.
 */
template< class A > 
const OpFunc* GetOpFuncBase< A >::makeHopFunc( HopIndex hopIndex ) const
{
	return new GetHopFunc< A >( hopIndex );
}

#endif // _HOP_FUNC_H
