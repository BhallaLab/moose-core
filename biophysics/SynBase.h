/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SYN_HANDLER_H
#define _SYN_HANDLER_H


/**
 * This is a base class for accessing and handling synapses.
 * It provides a uniform interface so that all classes that use synapses
 * can do so without duplication.
 * Derived classes must provide a function to handle the synaptic
 * events.
 * It does not assume anything about how to manage synaptic events.
 */
class SynBase
{
	public: 
		SynBase();
		virtual ~SynBase();

		SynBase( double thresh, double tau );

		/**
 		 * Inserts an event into the pendingEvents queue for spikes.
 		 */
		void addSpike( unsigned int synIndex, const double time );
		
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		/**
		 * Resizes the synapse storage
		 */
		void setNumSynapses( unsigned int v );

		/**
		 * Returns number of synapses defined.
		 */
		unsigned int getNumSynapses() const;

		/**
		 * Gets specified synapse
		 */
		Synapse* getSynapse( unsigned int i );

		/// Sets weight of specified synapse
		void setWeight( unsigned int index, double v );
		/// Gets weight of specified synapse
		double getWeight( unsigned int index ) const;

		/// Sets delay of specified synapse
		void setDelay( unsigned int index, double v );
		/// Gets delay of specified synapse
		double getDelay( unsigned int index ) const;

		////////////////////////////////////////////////////////////////
		/**
		 * This is the key function of this class: meant to be overridden.
		 * It doesn't do anything here, but the derived classes use this
		 * to decide what to do with their spike events.
		 */
		virtual void innerAddSpike( unsigned int synIndex, double time );

		static const unsigned int MAX_SYNAPSES;
		static const Cinfo* initCinfo();
	private:
		vector< Synapse > synapses_;
};

#endif // _SYN_HANDLER_H
