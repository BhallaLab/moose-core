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
		 * Generates an iterator for this ConnTainer. The eIndex
		 * specifies the index of the originating element.
		 * The isReverse flag is passed in from the Msg and specifies
		 * if we are going forward or back along the direction of the Msg.
		 */
		virtual Conn* conn( unsigned int eIndex, bool isReverse ) const = 0;

		/**
		 * Generates an iterator for this ConnTainer. The eIndex
		 * specifies the index of the originating element.
		 * The isReverse flag is passed in from the Msg and specifies
		 * if we are going forward or back along the direction of the Msg.
		 * The connIndex specifies a conn within this ConnTainer.
		 */
		virtual Conn* conn( unsigned int eIndex, bool isReverse, 
			unsigned int connIndex ) const = 0;

		// virtual bool add( Element* e1, Element* e2 ) = 0;

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
		 * Returns the number of targets on this ConnTainer
		 */
		virtual unsigned int size() const = 0;

		/**
 		 * Creates a duplicate ConnTainer for message(s) between 
 		 * new elements e1 and e2,
 		 * It checks the original version for which msgs to put the new
 		 * one on. e1 must be the new source element.
 		 * Returns the new ConnTainer on success, otherwise 0.
 		*/
		virtual ConnTainer* copy( Element* e1, Element* e2 ) const = 0;

		static const unsigned int Default;
		
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

#endif // _CONN_TAINER_H
