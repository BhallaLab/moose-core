#ifndef _ShellWrapper_h
#define _ShellWrapper_h
class ShellWrapper: 
	public Shell, public Neutral
{
	friend Element* addInConnLookup( const Conn* );
	friend Element* dropInConnLookup( const Conn* );
	friend Element* setInConnLookup( const Conn* );
	friend Element* createInConnLookup( const Conn* );
	friend Element* deleteInConnLookup( const Conn* );
	friend Element* moveInConnLookup( const Conn* );
	friend Element* copyInConnLookup( const Conn* );
	friend Element* copyShallowInConnLookup( const Conn* );
	friend Element* copyHaloInConnLookup( const Conn* );
	friend Element* ceInConnLookup( const Conn* );
	friend Element* pusheInConnLookup( const Conn* );
	friend Element* popeInConnLookup( const Conn* );
	friend Element* aliasInConnLookup( const Conn* );
	friend Element* quitInConnLookup( const Conn* );
	friend Element* stopInConnLookup( const Conn* );
	friend Element* resetInConnLookup( const Conn* );
	friend Element* stepInConnLookup( const Conn* );
	friend Element* callInConnLookup( const Conn* );
	friend Element* getInConnLookup( const Conn* );
	friend Element* getmsgInConnLookup( const Conn* );
	friend Element* isaInConnLookup( const Conn* );
	friend Element* showInConnLookup( const Conn* );
	friend Element* showmsgInConnLookup( const Conn* );
	friend Element* showobjectInConnLookup( const Conn* );
	friend Element* pweInConnLookup( const Conn* );
	friend Element* leInConnLookup( const Conn* );
	friend Element* listCommandsInConnLookup( const Conn* );
	friend Element* listClassesInConnLookup( const Conn* );
	friend Element* echoInConnLookup( const Conn* );
	friend Element* commandConnLookup( const Conn* );
    public:
		ShellWrapper(const string& n)
		:
			Shell( this), Neutral( n ),
			commandReturnSrc_( &commandConn_ )
			// addInConn uses a templated lookup function,
			// dropInConn uses a templated lookup function,
			// setInConn uses a templated lookup function,
			// createInConn uses a templated lookup function,
			// deleteInConn uses a templated lookup function,
			// moveInConn uses a templated lookup function,
			// copyInConn uses a templated lookup function,
			// copyShallowInConn uses a templated lookup function,
			// copyHaloInConn uses a templated lookup function,
			// ceInConn uses a templated lookup function,
			// pusheInConn uses a templated lookup function,
			// popeInConn uses a templated lookup function,
			// aliasInConn uses a templated lookup function,
			// quitInConn uses a templated lookup function,
			// stopInConn uses a templated lookup function,
			// resetInConn uses a templated lookup function,
			// stepInConn uses a templated lookup function,
			// callInConn uses a templated lookup function,
			// getInConn uses a templated lookup function,
			// getmsgInConn uses a templated lookup function,
			// isaInConn uses a templated lookup function,
			// showInConn uses a templated lookup function,
			// showmsgInConn uses a templated lookup function,
			// showobjectInConn uses a templated lookup function,
			// pweInConn uses a templated lookup function,
			// leInConn uses a templated lookup function,
			// listCommandsInConn uses a templated lookup function,
			// listClassesInConn uses a templated lookup function,
			// echoInConn uses a templated lookup function
			// commandConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setIsInteractive( Conn* c, int value ) {
			static_cast< ShellWrapper* >( c->parent() )->isInteractive_ = value;
		}
		static int getIsInteractive( const Element* e ) {
			return static_cast< const ShellWrapper* >( e )->isInteractive_;
		}
		static void setParser( Conn* c, string value ) {
			static_cast< ShellWrapper* >( c->parent() )->parser_ = value;
		}
		static string getParser( const Element* e ) {
			return static_cast< const ShellWrapper* >( e )->parser_;
		}
		static void setResponse( Conn* c, string value ) {
			static_cast< ShellWrapper* >( c->parent() )->response_ = value;
		}
		static string getResponse( const Element* e ) {
			return static_cast< const ShellWrapper* >( e )->response_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getCommandReturnSrc( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->commandReturnSrc_ );
		}
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void addFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				addFuncLocal( src, dest );
		}
		static void dropFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				dropFuncLocal( src, dest );
		}
		static void setFunc( Conn* c, string field, string value ) {
			static_cast< ShellWrapper* >( c->parent() )->
				setFuncLocal( field, value );
		}
		static void createFunc( Conn* c, string type, string path ) {
			static_cast< ShellWrapper* >( c->parent() )->
				createFuncLocal( type, path );
		}
		static void deleteFunc( Conn* c, string path ) {
			static_cast< ShellWrapper* >( c->parent() )->
				deleteFuncLocal( path );
		}
		static void moveFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				moveFuncLocal( src, dest );
		}
		static void copyFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				copyFuncLocal( src, dest );
		}
		static void copyShallowFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				copyShallowFuncLocal( src, dest );
		}
		static void copyHaloFunc( Conn* c, string src, string dest ) {
			static_cast< ShellWrapper* >( c->parent() )->
				copyHaloFuncLocal( src, dest );
		}
		static void ceFunc( Conn* c, string newpath ) {
			static_cast< ShellWrapper* >( c->parent() )->
				ceFuncLocal( newpath );
		}
		static void pusheFunc( Conn* c, string newpath ) {
			static_cast< ShellWrapper* >( c->parent() )->
				pusheFuncLocal( newpath );
		}
		static void popeFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				popeFuncLocal(  );
		}
		static void aliasFunc( Conn* c, string origfunc, string newfunc ) {
			static_cast< ShellWrapper* >( c->parent() )->
				aliasFuncLocal( origfunc, newfunc );
		}
		static void quitFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				quitFuncLocal(  );
		}
		static void stopFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				stopFuncLocal(  );
		}
		static void resetFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				resetFuncLocal(  );
		}
		static void stepFunc( Conn* c, string steptime, string options ) {
			static_cast< ShellWrapper* >( c->parent() )->
				stepFuncLocal( steptime, options );
		}
		static void callFunc( Conn* c, string args ) {
			static_cast< ShellWrapper* >( c->parent() )->
				callFuncLocal( args );
		}
		static void getFunc( Conn* c, string field ) {
			static_cast< ShellWrapper* >( c->parent() )->
				getFuncLocal( field );
		}
		static void getmsgFunc( Conn* c, string field, string options ){
			static_cast< ShellWrapper* >( c->parent() )->
				getmsgFuncLocal( field, options );
		}
		static void isaFunc( Conn* c, string type, string field ) {
			static_cast< ShellWrapper* >( c->parent() )->
				isaFuncLocal( type, field );
		}
		static void showFunc( Conn* c, string field ) {
			static_cast< ShellWrapper* >( c->parent() )->
				showFuncLocal( field );
		}
		static void showmsgFunc( Conn* c, string field ) {
			static_cast< ShellWrapper* >( c->parent() )->
				showmsgFuncLocal( field );
		}
		static void showobjectFunc( Conn* c, string classname ) {
			static_cast< ShellWrapper* >( c->parent() )->
				showobjectFuncLocal( classname );
		}
		static void pweFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				pweFuncLocal(  );
		}
		static void leFunc( Conn* c, string start ) {
			static_cast< ShellWrapper* >( c->parent() )->
				leFuncLocal( start );
		}
		static void listCommandsFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				listCommandsFuncLocal(  );
		}
		static void listClassesFunc( Conn* c ) {
			static_cast< ShellWrapper* >( c->parent() )->
				listClassesFuncLocal(  );
		}
		static void echoFunc( Conn* c, vector< string >* s, int options ) {
			static_cast< ShellWrapper* >( c->parent() )->
				echoFuncLocal( *s, options );
		}

		static void commandFunc( Conn* c, int argc, const char** argv ){
			static_cast< ShellWrapper* >( c->parent() )->
				commandFuncLocal( argc, argv );
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getAddInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->addInConn_ );
		}
		static Conn* getDropInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->dropInConn_ );
		}
		static Conn* getSetInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->setInConn_ );
		}
		static Conn* getCreateInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->createInConn_ );
		}
		static Conn* getDeleteInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->deleteInConn_ );
		}
		static Conn* getMoveInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->moveInConn_ );
		}
		static Conn* getCopyInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->copyInConn_ );
		}
		static Conn* getCopyShallowInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->copyShallowInConn_ );
		}
		static Conn* getCopyHaloInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->copyHaloInConn_ );
		}
		static Conn* getCeInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->ceInConn_ );
		}
		static Conn* getPusheInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->pusheInConn_ );
		}
		static Conn* getPopeInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->popeInConn_ );
		}
		static Conn* getAliasInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->aliasInConn_ );
		}
		static Conn* getQuitInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->quitInConn_ );
		}
		static Conn* getStopInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->stopInConn_ );
		}
		static Conn* getResetInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->resetInConn_ );
		}
		static Conn* getStepInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->stepInConn_ );
		}
		static Conn* getCallInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->callInConn_ );
		}
		static Conn* getGetInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->getInConn_ );
		}
		static Conn* getGetmsgInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->getmsgInConn_ );
		}
		static Conn* getIsaInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->isaInConn_ );
		}
		static Conn* getShowInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->showInConn_ );
		}
		static Conn* getShowmsgInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->showmsgInConn_ );
		}
		static Conn* getShowobjectInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->showobjectInConn_ );
		}
		static Conn* getPweInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->pweInConn_ );
		}
		static Conn* getLeInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->leInConn_ );
		}
		static Conn* getListCommandsInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->listCommandsInConn_ );
		}
		static Conn* getListClassesInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->listClassesInConn_ );
		}
		static Conn* getEchoInConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->echoInConn_ );
		}

		// Note that this is a shared conn, so no direction pertains.
		static Conn* getCommandConn( Element* e ) {
			return &( static_cast< ShellWrapper* >( e )->commandConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Shell* p = dynamic_cast<const Shell *>(proto);
			// if (p)... and so on. 
			return new ShellWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc1< string > commandReturnSrc_;
		UniConn< addInConnLookup > addInConn_;
		UniConn< dropInConnLookup > dropInConn_;
		UniConn< setInConnLookup > setInConn_;
		UniConn< createInConnLookup > createInConn_;
		UniConn< deleteInConnLookup > deleteInConn_;
		UniConn< moveInConnLookup > moveInConn_;
		UniConn< copyInConnLookup > copyInConn_;
		UniConn< copyShallowInConnLookup > copyShallowInConn_;
		UniConn< copyHaloInConnLookup > copyHaloInConn_;
		UniConn< ceInConnLookup > ceInConn_;
		UniConn< pusheInConnLookup > pusheInConn_;
		UniConn< popeInConnLookup > popeInConn_;
		UniConn< aliasInConnLookup > aliasInConn_;
		UniConn< quitInConnLookup > quitInConn_;
		UniConn< stopInConnLookup > stopInConn_;
		UniConn< resetInConnLookup > resetInConn_;
		UniConn< stepInConnLookup > stepInConn_;
		UniConn< callInConnLookup > callInConn_;
		UniConn< getInConnLookup > getInConn_;
		UniConn< getmsgInConnLookup > getmsgInConn_;
		UniConn< isaInConnLookup > isaInConn_;
		UniConn< showInConnLookup > showInConn_;
		UniConn< showmsgInConnLookup > showmsgInConn_;
		UniConn< showobjectInConnLookup > showobjectInConn_;
		UniConn< pweInConnLookup > pweInConn_;
		UniConn< leInConnLookup > leInConn_;
		UniConn< listCommandsInConnLookup > listCommandsInConn_;
		UniConn< listClassesInConnLookup > listClassesInConn_;
		UniConn< echoInConnLookup > echoInConn_;
		UniConn< commandConnLookup > commandConn_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ShellWrapper_h
