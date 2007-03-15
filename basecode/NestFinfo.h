/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _NEST_FINFO_H
#define _NEST_FINFO_H

/**
 * Handles nested fields. Requires that the specified field clas has
 * already been declared. 
 */
class NestFinfo: public Finfo
{
		public:
			NestFinfo( const string& name,
						const Cinfo* nestClass,
						void* (*ptrFunc)( void*, unsigned int )
					 )
					: Finfo( name, nestClass->ftype() ),
					nestClass_( nestClass ),
					ptrFunc_( ptrFunc ),
					maxIndex_( 0 )
			{;}

			~NestFinfo()
			{;}

			/**
			 * NestFinfo::add should never happen. All ops take
			 * place on the DynamicFinfo.
			 */
			bool add( 
					Element* e, Element* destElm, const Finfo* destFinfo
			) const {
					assert( 0 );
					return 0;
			}
			
			bool respondToAdd(
					Element* e, Element* src, const Ftype *srcType,
					FuncList& srcFl, FuncList& returnFl,
					unsigned int& destIndex, unsigned int& numDest
			) const {
					assert( 0 );
					return 0;
			}

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Element* e, const std::string &s ) const
			{ 
					return 0;
			}
			
			// The Ftype handles this conversion.
			bool strGet( const Element* e, std::string &s ) const
			{
					return 0;
			}

			unsigned int srcList(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}
			unsigned int destList(
					const Element* e, vector< Conn >& list ) const {
					return 0;
			}

			/**
			 * We don't need to do anything here because NestFinfo
			 * does not deal with messages directly. If we need to
			 * send messages to a NestFinfo, then a DynamicFinfo
			 * must be created
			 */
			void countMessages( 
					unsigned int& srcNum, unsigned int& destNum )
			{ ; }

			/** 
			 * This is the main operation of the NestFinfo. 
			 * It handles indirection as well as indexing.
			 * Indexing is standard using [], but we may have
			 * various separators for indirection. Not clear
			 * how to pick one or another.
			 * This func returns a DynamicFinfo to deal with the
			 * indirection.
			 * It is up to the calling function to decide if the
			 * Dynamic Finfo should stay or be cleared out.
			 */
			const Finfo* match( Element* e, const string& name ) const;
			
			/**
			 * The NestFinfo never has messages going to or from it:
			 * they all go via DynamicFinfo if needed. So it cannot
			 * match any connIndex.
			 */
			const Finfo* match( 
				const Element* e, unsigned int connIndex ) const {
				return 0;
			}

			RecvFunc recvFunc() const {
					return 0;
			}

			bool isTransient() const {
					return 0;
			}

			/**
			 * Returns true only if the other finfo is the same type
			 */
			bool inherit( const Finfo* baseFinfo );

			const Finfo* parseName( 
				vector< IndirectType >& v, const string& path ) const;

		private:
			const Cinfo* nestClass_;
			void* (*ptrFunc_)( void*, unsigned int );
			unsigned int maxIndex_;
};

#endif // _NEST_FINFO_H
