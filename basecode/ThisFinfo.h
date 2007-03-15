/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _THIS_FINFO_H
#define _THIS_FINFO_H

/**
 * This Finfo represents the entire data object in an Element.
 * It permits access to the entirety of the data, and also supports
 * going to the class info for further info.
 */
class ThisFinfo: public Finfo
{
		public:
			ThisFinfo( const Cinfo* c )
					: Finfo( "this", c->ftype() ), cinfo_( c )
			{;}

			~ThisFinfo()
			{;}

			/**
			 * This slightly odd looking operation is meant to 
			 * connect any calls made to this dest onward to other
			 * dests. 
			 * For now we ignore it in the DestFinfo.
			 */
			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const 
			{
					return 0;
			}
			
			bool respondToAdd(
					Element* e, Element* dest, const Ftype *destType,
					FuncList& destfl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
			) const;
			
			/*
			virtual bool drop( Element* e, unsigned int i ) const = 0;

			virtual bool respondToDrop( Element* e, unsigned int i )
					const = 0;
					*/

			unsigned int srcList(
					const Element* e, vector< Conn >& list ) const;
			unsigned int destList(
					const Element* e, vector< Conn >& list ) const;


			RecvFunc recvFunc() const {
					return 0;
					// This should refer to a Ftype<T> static function which does a copy of the entire object
			}

			/**
			 * Call the RecvFunc with the arguments in the string.
			 */
			bool strSet( Element* e, const std::string &s )
					const;
			
			/// strGet doesn't work for DestFinfo
			bool strGet( const Element* e, std::string &s ) const {
				return 0;
			}

			const Finfo* match( Element* e, const string& name ) const;
			const Finfo* match( 
					const Element* e, unsigned int connIndex) const;

			// ThisFinfo must go to the cinfo to build up the list.
			void listFinfos( vector< const Finfo* >& flist ) const;

			// ThisFinfo does not allocate any MsgSrc or MsgDest
			// so it does not use this function.
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{
					;
			}

			/**
			 * This cannot be inherited. Returns 0 always.
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return 0;
			}

			bool isTransient() const {
					return 0;
			}

		private:
			const Cinfo* cinfo_;
};

#endif // _THIS_FINFO_H
