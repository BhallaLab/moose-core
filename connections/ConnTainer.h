/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CONN_TAINER_H
#define _CONN_TAINER_H

class ConnTainer
{
	public:
		ConnTainer( Element* e1, Element* e2,
			int msg1, int msg2 )
			:
			e1_( e1 ), e2_( e2 ),
			msg1_( msg1 ), msg2_( msg2 )
		{;}

		virtual ~ConnTainer()
		{;}

		/**
		 * Generates an iterator for this ConnTainer. Eref
		 * specifies two things: the direction of the msg is worked
		 * out by comparing e.e with the internal e1 and e2 element ptrs,
		 * and the e.eIndex specifies the index of the originating element.
		 */
		virtual Conn* conn( Eref e, unsigned int funcIndex ) const = 0;

		Element* e1() const {
			return e1_;
		}

		Element* e2() const {
			return e2_;
		}

		int msg1() const {
			return msg1_;
		}

		int msg2() const {
			return msg2_;
		}

		/**
		 * Returns the number of entries (src/dest pairs) in this ConnTainer
		 */
		virtual unsigned int size() const = 0;

		/**
		 * Returns the number of sources coming to the specified
		 * eIndex,
		 */
		virtual unsigned int numSrc( unsigned int eIndex ) const = 0;

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		virtual unsigned int numDest( unsigned int eIndex ) const = 0;

		/**
 		 * Creates a duplicate ConnTainer for message(s) between 
 		 * new elements e1 and e2, isArray tell whether e1, e2 are arrays
 		 * It checks the original version for which msgs to put the new
 		 * one on. e1 must be the new source element.
 		 * Returns the new ConnTainer on success, otherwise 0.
 		*/
		virtual ConnTainer* copy( Element* e1, Element* e2, bool isArray ) const = 0;

		/**
		 * Returns the identifying integer for the ConnTainer type.
		 */
		virtual unsigned int option() const = 0;

		/**
		 * Add a new connection to the ConnTainer. Returns true on success
		 */
		virtual bool addToConnTainer( unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index ) = 0;

		static const unsigned int Default;
		static const unsigned int Simple;
		static const unsigned int One2All;
		static const unsigned int One2OneMap;
		static const unsigned int All2One;
		static const unsigned int Many2Many;
		
	private:
		Element* e1_; // Pointer to element 1
		Element* e2_; // Pointer to element 2
		
		/**
		 * Identifier1 for Conntainer. 
		 * If +ve, it is a src and msg1 looks up the Msg on e1->msg_. 
		 * This in turn has a vector of ConnTainers.
		 * If -ve, it is a dest, and msg1 does a map lookup on e1->dest_.
		 * This too has a vector of ConnTainers.
		 */
		int msg1_;	

		/**
		 * Identifier2 for Conntainer. 
		 * If +ve, it is a src and msg2 looks up the Msg on e2->msg_.
		 * This in turn has a vector of ConnTainers.
		 * If -ve, it is a dest, and msg2 does a map lookup on e2->dest_.
		 * This too has a vector of ConnTainers.
		 */
		int msg2_;
};

/**
 * Utility function to generate a suitable ConnTainer for the message,
 * depending on src and dest Element subclass and indexing.
 * Currently most of the options revert to SimpleConnTainer.
 */
extern ConnTainer* selectConnTainer( Eref src, Eref dest, 
	int srcMsg, int destMsg,
	unsigned int srcIndex, unsigned int destIndex,
	unsigned int connTainerOption = ConnTainer::Default );

/**
 * This function picks a suitable container depending on the
 * properties of the src and dest, and the status of the incoming option.
 * It is flawed in various ways:
 * - Uses string comparisons to find element type, rather than a 
 *   virtual func
 * - Doesn't handle proxies
 * - Can't handle conversion of a SimpleElement to ArrayElement.
 *
 * Also not clear how to deal with conversion of a Many2Many to an
 * All2All if needed.
 */
extern unsigned int connOption( Eref src, Eref dest, unsigned int option );

#endif // _CONN_TAINER_H
