/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
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
 * It does not assume anything about how to manage synaptic events.
 */
class SynHandler
{
	public: 
		SynHandler();
		virtual ~SynHandler();
		
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

		////////////////////////////////////////////////////////////////
		// Buffer operations
		////////////////////////////////////////////////////////////////
		
		/**
		 * Sets up the buffer with 'size' entries and dt time for each bin.
		 */
		void reinitBuffer( double dt );

		/**
		 * Returns the current buffer entry, and advances it.
		 */
		double popBuffer();
		////////////////////////////////////////////////////////////////
		/// Adds a new synapse, returns its index.
		unsigned int addSynapse();
		void dropSynapse( unsigned int droppedSynNumber );
		////////////////////////////////////////////////////////////////
		static const unsigned int MAX_SYNAPSES;
		static const Cinfo* initCinfo();
	private:
		vector< Synapse > synapses_;
		SpikeRingBuffer buf_;
};

#endif // _SYN_HANDLER_H
