/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GENESIS_PARSER_WRAPPER_H
#define _GENESIS_PARSER_WRAPPER_H
// class myFlexLexer: public yyFlexLexer

// ISO C++ requires that these be declared in the innermost enclosing scope.
// Most compilers inject[ed] this declaration automatically, but that's a favor.
// These should be ok in an enclosing namespace.
Element* lookupEchoConn( const Conn* );
Element* lookupCommandConn( const Conn* );
Element* lookupShellInputConn( const Conn* c );

class GenesisParserWrapper: 
	public myFlexLexer, public Element
{
//	friend Element* lookupProcessConn( const Conn* );
	friend Element* lookupEchoConn( const Conn* );
	friend Element* lookupCommandConn( const Conn* );
	friend Element* lookupShellInputConn( const Conn* c );
    public:
		GenesisParserWrapper(const string& n)
		:
		myFlexLexer( this ),
		Element(n),
		readlineConn_( this ),
		processConn_( this ),
		parseConn_( this ),
		echoSrc_( &echoConn_ ),
		commandSrc_( &commandConn_ )
		{
			loadBuiltinCommands();
		}

		// essential functions
		static Element* create(
			const string& name, Element* pa, const Element* proto) {
			// ignore proto
			return new GenesisParserWrapper(name);
			/*
			GenesisParserWrapper* ret = new GenesisParserWrapper(name);
			if ( pa->adoptChild( ret ) ) {
				return ret;
			} else {
				delete ret;
				return 0;
			}
			*/
		}

		bool adoptChild( Element* e ) {
			return 0;
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		static Conn* getReadlineConn( Element* e ) {
			return &( static_cast< GenesisParserWrapper* >( e )->
				readlineConn_ );
		}

		static void readlineFunc( Conn* c, string s ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				AddInput( &s );
		}

		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< GenesisParserWrapper* >( e )->
				processConn_ );
		}

		static void processFunc( Conn* c ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				Process();
		}

		static Conn* getParseConn( Element* e ) {
			return &( static_cast< GenesisParserWrapper* >( e )->
				parseConn_ );
		}

		static void parseFunc( Conn* c, string s ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				ParseInput( &s );
		}

		static SingleMsgSrc* getEchoSrc( Element* e ) {
			return &( static_cast<GenesisParserWrapper *>( e )->
				echoSrc_ );
		}

		static SingleMsgSrc* getCommandSrc( Element* e ) {
			return &( static_cast<GenesisParserWrapper *>( e )->
				commandSrc_ );
		}

		static void aliasFunc( Conn* c, string alias, string old ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				alias( alias, old );
		}

		static void listCommandsFunc( Conn* c ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				listCommands( );
		}

		static Conn* getShellInputConn( Element* e ) {
			return &( static_cast< GenesisParserWrapper* >( e )->
				shellInputConn_ );
		}

		/*
		static SingleMsgSrc1< string >* getEcho( Element* e ) {
			return &( static_cast<GenesisParserWrapper *>( e )->
				echoSrc_ );
		}
		*/
		////////////////////////////////////////////////
		//  Utility functions
		////////////////////////////////////////////////
		static void setShell( Conn* c, string s );
		static string getShell( const Element* e );
		void plainCommand( int argc, const char** argv );
		char* returnCommand( int argc, const char** argv );

    private:
		void loadBuiltinCommands();
		PlainMultiConn readlineConn_;
		// UniConn< lookupProcessConn > processConn_;
		UniConn2 processConn_;
		PlainMultiConn parseConn_;

		SingleMsgSrc1< string > echoSrc_;
		SingleMsgSrc2< int, const char** > commandSrc_;
		UniConn< lookupEchoConn > echoConn_;
		UniConn< lookupCommandConn > commandConn_;
		UniConn< lookupShellInputConn > shellInputConn_;

		string returnCommandValue_;

		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _GENESIS_PARSER_WRAPPER_H
