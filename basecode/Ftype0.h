/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FTYPE0_H
#define _FTYPE0_H

extern bool set( Eref e, const Finfo* f );

class Ftype0: public Ftype
{
		public:
			Ftype0()
				: Ftype( "ftype0" )
			{
				addSyncFunc( RFCAST( &( Ftype0::syncFunc ) ) );
				addAsyncFunc( RFCAST( &( Ftype0::asyncFunc ) ) );
				addProxyFunc( RFCAST( &( Ftype0::proxyFunc ) ) );
			}

			unsigned int nValues() const {
				return 0;
			}
			
			bool isSameType( const Ftype* other ) const {
				return ( dynamic_cast< const Ftype0* >( other ) != 0 );
			}

			static bool isA ( const Ftype* other ) {
				return ( dynamic_cast< const Ftype0* >( other ) != 0 );
			}

			size_t size() const
			{
				return 0;
			}

			static const Ftype* global() {
				static Ftype* ret = new Ftype0();
				return ret;
			}

			RecvFunc recvFunc() const {
				return 0;
			}

			RecvFunc trigFunc() const {
				return 0;
			}

			virtual std::string getTemplateParameters() const
			{
				return "none";
			}

			/**
			 * This is a virtual function that calls the function.
			 * It takes a string, but ignores its value.
			 * Returns true on success.
			 */
			bool strSet( Eref e, const Finfo* f, const string& s )
					const
			{
				return set( e, f );
			}
			
			///////////////////////////////////////////////////////
			// Here we define the functions for serializing data
			// for parallel messaging.
			///////////////////////////////////////////////////////

			static void proxyFunc(
				const Conn* c, const void* data, Slot slot )
			{
				extern void send0( Eref e, Slot src );
				send0( c->target(), slot );
			}

			static void syncFunc( const Conn* c )
			{
				; // Don't have to do anything at all here: no data is added
				// Actually data-less sync messages don't make much sense
			}

			static void asyncFunc( const Conn* c )
			{
				// Although we don't add anything to the buffer, this 
				// function adds the info for the presence of this message.
				getAsyncParBuf( c, 0 );
			}
};

#endif // _FTYPE0_H
