/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SYNAPSE_H
#define _SYNAPSE_H

class Synapse
{
	public:
		Synapse();
		Synapse( double w, double d );
		Synapse( const Synapse& other, double time );

		// This is backward because the syntax of the priority
		// queue puts the _largest_ element on top.
		bool operator< ( const Synapse& other ) const {
			return delay_ > other.delay_;
		}
		
		bool operator== ( const Synapse& other ) const {
			return delay_ == other.delay_ && weight_ == other.weight_;
		}

		void setWeight( double v );
		void setDelay( double v );

		double getWeight() const;
		double getDelay() const;
		static const Cinfo* initCinfo();
	private:

		double weight_;
		double delay_;
};

class SynElement: public Element
{
	public:
		SynElement( const Cinfo* c, const Element* other );

		void process( const ProcInfo* p ); // Don't do anything.

		/**
		 * Return the Synapse
		 */
		char* data( DataId index );

		/**
		 * Return the IntFire
		 */
		char* data1( DataId index );

		/**
		 * Return the # of synapses
		 */
		unsigned int numData() const;

		/**
		 * Return the # of synapses on a given IntFire
		 */
		unsigned int numData2( unsigned int index1 ) const;

		/**
		 * Return 2.
		 */
		unsigned int numDimensions() const;

		void setArraySizes( const vector< unsigned int >& sizes );
		void getArraySizes( vector< unsigned int >& sizes ) const;
};

#endif // _SYNAPSE_H
