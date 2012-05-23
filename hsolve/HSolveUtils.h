/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_UTILS_H
#define _HSOLVE_UTILS_H

class HSolveUtils
{
public:
	static void initialize( Id object );
	
	template< class T, class A >
	static void set( Id id, string field, A value )
	{
		const Finfo* finfo = id()->cinfo()->findFinfo( "set_" + field );
		assert( finfo );
		
		const DestFinfo* dest = dynamic_cast< const DestFinfo* >( finfo );
		assert( dest );
		
		const OpFunc* op = dest->getOpFunc();
		assert( op );
		
		Qinfo* qDummy = NULL;
		const double* vp = reinterpret_cast< const double* >( &value );
		op->op( id.eref(), qDummy, vp );
	}
	
	template< class T, class A >
	static A get( Id id, string field )
	{
		const Finfo* finfo = id()->cinfo()->findFinfo( "get_" + field );
		assert( finfo );
		
		const DestFinfo* dest = dynamic_cast< const DestFinfo* >( finfo );
		assert( dest );
		
		const OpFunc* op = dest->getOpFunc();
		assert( op );
		
		const GetOpFunc< T, A >* gop =
			dynamic_cast< const GetOpFunc< T, A >* >( op );
		const GetEpFunc< T, A >* gep =
			dynamic_cast< const GetEpFunc< T, A >* >( op );
		assert( gop || gep );
		
		if ( gop )
			return gop->reduceOp( id.eref() );
		else // gep
			return gep->reduceOp( id.eref() );
	}
	
	static int adjacent( Id compartment, vector< Id >& ret );
	static int adjacent( Id compartment, Id exclude, vector< Id >& ret );
	static int children( Id compartment, vector< Id >& ret );
	static int channels( Id compartment, vector< Id >& ret );
	static int hhchannels( Id compartment, vector< Id >& ret );
	static int gates( Id channel, vector< Id >& ret, bool getOriginals = true );
	static int spikegens( Id compartment, vector< Id >& ret );
	static int synchans( Id compartment, vector< Id >& ret );
	static int leakageChannels( Id compartment, vector< Id >& ret );
	static int caTarget( Id channel, vector< Id >& ret );
	static int caDepend( Id channel, vector< Id >& ret );
	
	class Grid
	{
	public:
		Grid( double min, double max, unsigned int divs )
			:
			min_( min ),
			max_( max ),
			divs_( divs )
		{
			dx_ = ( max_ - min_ ) / divs_;
		}
		
		unsigned int size();
		double entry( unsigned int i );
		
		double min_;
		double max_;
		unsigned int divs_;
		double dx_;
	};
	
	// Functions for accessing gates' lookup tables.
	//~ static int domain(
		//~ Id gate,
		//~ double& min,
		//~ double& max );
	static void rates(
		Id gate,
		Grid grid,
		vector< double >& A,
		vector< double >& B );
	//~ static int modes(
		//~ Id gate,
		//~ int& AMode,
		//~ int& BMode );
	
	static int targets(
		Id object,
		string msg,
		vector< Id >& target,
		string filter = "",
		bool include = true );
	
	static int targets(
		Id object,
		string msg,
		vector< Id >& target,
		const vector< string >& filter,
		bool include = true );
};

#endif // _HSOLVE_UTILS_H
