/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SynInfo_h
#define _SynInfo_h

class SynInfo
{
	public:
		SynInfo() 
			: weight( 0.0 ), delay( 0.0 )
		{
			;
		}

		SynInfo( double w, double d ) 
			: weight( w ), delay( d )
		{
			;
		}

		bool operator< ( const SynInfo& other ) const {
			return delay < other.delay;
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

#endif // _SynInfo_h
