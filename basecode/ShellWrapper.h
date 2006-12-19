#ifndef _ShellWrapper_h
#define _ShellWrapper_h

#include <string>
#include <vector>
#include "Cinfo.h"
#include "ConnFwd.h"
#include "ElementFwd.h"
#include "Neutral.h"
#include "Shell.h"

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
	friend Element* schedNewObjectConnLookup( const Conn* );
    public:
		ShellWrapper(const std::string& n)
		:
			Shell( this), Neutral( n ),
			commandReturnSrc_( &commandConn_ ),
			remoteCommandSrc_( &remoteCommandConn_ ),
			addOutgoingSrc_( &remoteCommandConn_ ),
			addIncomingSrc_( &remoteCommandConn_ ),
			schedNewObjectSrc_( &schedNewObjectConn_ ),
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
			remoteCommandConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setIsInteractive( Conn* c, int value );
		static int getIsInteractive( const Element* e );
		static void setTotalNodes( Conn* c, int value );
		static int getTotalNodes( const Element* e );
		static void setMyNode( Conn* c, int value );
		static int getMyNode( const Element* e );
		static void setParser( Conn* c, std::string value );
		static std::string getParser( const Element* e );
		static void setResponse( Conn* c, std::string value );
		static std::string getResponse( const Element* e );
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getCommandReturnSrc( Element* e );
		static NMsgSrc* getRemoteCommandSrc( Element* e );
		static SingleMsgSrc* getSchedNewObjectSrc( Element* e );
		static NMsgSrc* getAddOutgoingSrc( Element* e );
		static NMsgSrc* getAddIncomingSrc( Element* e );
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void addFunc( Conn* c, std::string src, std::string dest );
		static void dropFunc( Conn* c, std::string src, std::string dest );
		static void setFunc( Conn* c, std::string field, std::string value );
		static void createFunc( Conn* c, std::string type, std::string path );
		static void deleteFunc( Conn* c, std::string path );
		static void moveFunc( Conn* c, std::string src, std::string dest );
		static void copyFunc( Conn* c, std::string src, std::string dest );
		static void copyShallowFunc( Conn* c, std::string src, std::string dest );
		static void copyHaloFunc( Conn* c, std::string src, std::string dest );
		static void ceFunc( Conn* c, std::string newpath );
		static void pusheFunc( Conn* c, std::string newpath );
		static void popeFunc( Conn* c );
		static void aliasFunc( Conn* c, std::string origfunc, std::string newfunc );
		static void quitFunc( Conn* c );
		static void stopFunc( Conn* c );
		static void resetFunc( Conn* c );
		static void stepFunc( Conn* c, std::string steptime, std::string options );
		static void callFunc( Conn* c, std::string args );
		static void getFunc( Conn* c, std::string field );
		static void getmsgFunc( Conn* c, std::string field, std::string options );
		static void isaFunc( Conn* c, std::string type, std::string field );
		static void showFunc( Conn* c, std::string field );
		static void showmsgFunc( Conn* c, std::string field );
		static void showobjectFunc( Conn* c, std::string classname );
		static void pweFunc( Conn* c );
		static void leFunc( Conn* c, std::string start );
		static void listCommandsFunc( Conn* c );
		static void listClassesFunc( Conn* c );
		static void echoFunc( Conn* c, std::vector< std::string >* s, int options );

		static void commandFunc( Conn* c, int argc, const char** argv );
		static void remoteCommandFunc( Conn* c, std::string command );

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getAddInConn( Element* e );
		static Conn* getDropInConn( Element* e );
		static Conn* getSetInConn( Element* e );
		static Conn* getCreateInConn( Element* e );
		static Conn* getDeleteInConn( Element* e );
		static Conn* getMoveInConn( Element* e );
		static Conn* getCopyInConn( Element* e );
		static Conn* getCopyShallowInConn( Element* e );
		static Conn* getCopyHaloInConn( Element* e );
		static Conn* getCeInConn( Element* e );
		static Conn* getPusheInConn( Element* e );
		static Conn* getPopeInConn( Element* e );
		static Conn* getAliasInConn( Element* e );
		static Conn* getQuitInConn( Element* e );
		static Conn* getStopInConn( Element* e );
		static Conn* getResetInConn( Element* e );
		static Conn* getStepInConn( Element* e );
		static Conn* getCallInConn( Element* e );
		static Conn* getGetInConn( Element* e );
		static Conn* getGetmsgInConn( Element* e );
		static Conn* getIsaInConn( Element* e );
		static Conn* getShowInConn( Element* e );
		static Conn* getShowmsgInConn( Element* e );
		static Conn* getShowobjectInConn( Element* e );
		static Conn* getPweInConn( Element* e );
		static Conn* getLeInConn( Element* e );
		static Conn* getListCommandsInConn( Element* e );
		static Conn* getListClassesInConn( Element* e );
		static Conn* getEchoInConn( Element* e );
		static Conn* getSchedNewObjectConn( Element* e );

		// Note that both are shared conns, so no direction pertains.
		static Conn* getCommandConn( Element* e );
		static Conn* getRemoteCommandConn( Element* e );

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const std::string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Shell* p = dynamic_cast<const Shell *>(proto);
			// if (p)... and so on. 
			return new ShellWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

//////////////////////////////////////////////////////////////////
// Parallel Access utility fuctions.
//////////////////////////////////////////////////////////////////

		bool addToRemoteNode( 
						Field& s, const string& dest, int destNode );
		void addFromRemoteNode(
						int srcNode, Field& dest, int tick, int size);

		void sendRemoteCommand( 
						const string& command, int destNode = -1 );

		void schedNewObject( Element* e );
    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc1< std::string > commandReturnSrc_;
		NMsgSrc1< std::string > remoteCommandSrc_;
		NMsgSrc3< Field&, int, int > addOutgoingSrc_;
		NMsgSrc3< Field&, int, int > addIncomingSrc_;
		SingleMsgSrc1< Element* > schedNewObjectSrc_;
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
		MultiConn remoteCommandConn_;
		UniConn< schedNewObjectConnLookup > schedNewObjectConn_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ShellWrapper_h
