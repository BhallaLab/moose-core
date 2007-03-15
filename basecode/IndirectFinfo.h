/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _INDIRECT_FINFO_H
#define _INDIRECT_FINFO_H

/**
 * This Finfo manages indirection. It keeps track of how to look up
 * a portion of a data structure, and thus permits the calling recvFunc
 * to get at something deep inside the data structure. Since we
 * cannot anticipate how to do this, we need the calling recvFunc to
 * do a very stereotyped thing: just look up an IndirectionFinfo
 * whose connIndex matches the Conn argument in the recvFunc.
 * Once it has found the correct indirectionFinfo, the various
 * lookup function pointers can be used along with various other
 * tricks to let us pass the correctly indirected pointer to the
 * recvFunc.
 */
class IndirectFinfo: public Finfo
{
		public:
			IndirectFinfo( unsigned int connIndex, 
							Finfo* lookup,
							Finfo* target,
					: Finfo( name, target->ftype() )
			{;}

			~IndirectFinfo()
			{;}

			/**
			 * Indirect slightly odd looking operation is meant to 
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

			bool isTransient() const {
					return 0;
			}

			/**
			 * This operation makes no sense for the IndirectFinfo
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return 0;
			}

		private:
			Cinfo* cinfo_;
};

#endif // _INDIRECT_FINFO_H
