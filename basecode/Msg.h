/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * Manages data flow between two elements. Is always many-to-many, with
 * assorted variants.
 */
class Msg
{
	public:
		Msg( Element* src, Element* dest );
		virtual ~Msg();
	/**
 	* Find the targets identified by the src element index and push the
	* time onto all of them.
 	*/
		virtual void pushQ( unsigned int srcElementIndex, double time )
			const = 0;
	protected:
		Element* src_;
		Element* dest_;
};

/**
 * Also need a bidirectional variant if there is heavy reverse traffic.
 * This may be rather rare, since the cases with heavy bidirectional
 * traffic will typically be sync messages, which don't use the queues
 * for sending data.
 */
class SparseMsg: public Msg
{
	public:
		SparseMsg( Element* src, Element* dest );
		void pushQ( unsigned int srcElementIndex, double time ) const;
	private:
		// May have to be a pair of ints, to handle reverse msg indexing.
		// But this indexing is used only to identify src........
		// Can we use one of the SparseMatrix other tables for it?
		SparseMatrix< unsigned int > m_;
};

/**
 * This could be handy: maps onto same index.
 */
class One2OneMsg: public Msg
{
	public:
		One2OneMsg( Element* src, Element* dest );
		void pushQ( unsigned int srcElementIndex, double time ) const;
	private:
		unsigned int synIndex_;
};
