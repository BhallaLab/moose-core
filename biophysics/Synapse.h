/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SYN_INFO_H
#define _SYN_INFO_H

class SynInfo
{
	public:
		SynInfo() 
			: weight( 1.0 ), delay( 0.0 )
		{
			;
		}

		SynInfo( double w, double d ) 
			: weight( w ), delay( d )
		{
			;
		}

		SynInfo( const SynInfo& other, double time )
			: weight( other.weight ), delay( time + other.delay )
		{
			;
		}

		// This is backward because the syntax of the priority
		// queue puts the _largest_ element on top.
		bool operator< ( const SynInfo& other ) const {
			return delay > other.delay;
		}
		
		bool operator== ( const SynInfo& other ) const {
			return delay == other.delay && weight == other.weight;
		}

		SynInfo event( double time ) {
			return SynInfo( weight, time + delay );
		}

		double weight;
		double delay;
};

class SynElement: public Element
{
	public:
		void process( const ProcInfo* p ); // Don't do anything.

		/**
		 * Return the Synapse
		 */
		virtual char* data( DataId index );

		/**
		 * Return the IntFire
		 */
		virtual char* data1( DataId index );

		/**
		 * Return the # of synapses
		 */
		virtual unsigned int numData() const;

		/**
		 * Return 2.
		 */
		virtual unsigned int numDimensions() const;
};

#endif // _SYN_INFO_H
