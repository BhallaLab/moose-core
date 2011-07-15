// Replication of the GENESIS 2 Script Language Interface parser
// Much stuff here is copied over from GENESIS, which I believe we
// can now GPL.
// Acknowledgements: The original parser code is from Dave Bilitch and
// Matt Wilson, circa 1987, at Caltech.
// Set up for a C++ based parser by Upi Bhalla, 2004, 2005,
// at NCBS.
//
// With the proviso that Caltech has released the code to GPL, this
// code is licensed under the GNU Lesser General Public License v2.
// There is NO WARRANTY for its use. Please see the file
// COPYING.LIB for details.

#if MOOSE_THREADS		
#include <pthread.h>
#endif
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <cstring>

// Upi Bhalla, 24 May 2004:
// I did a little checking up about portability of the setjmp construct
// in the parser code. It looks like it ought to work even in VC++,
// but I'll have to actually compile it to be sure. The consensus
// seems to be that this language construct should probably be
// avoided. Since I want to keep the changes to the SLI parser code
// to a minimum, I'll leave it in.
#include <setjmp.h>

// #include "Shell.h"

#include <FlexLexer.h>
#include "GenesisParser.tab.h"
#include "script.h"
#include "header.h"
#include "GenesisParser.h"
#include "func_externs.h"

using std::cin;
using std::cerr;
using std::cout;
using std::map;
using std::string;

#include "GenesisParserWrapper.h"

#define MOOSE_THREADS 0

/**
 * Decides if the parser automatically should fall back to the system
 * shell (csh) if a command is not found. This is OK for most Unices,
 * but MPI is unhappy with fork calls. 
 */
bool Autoshell() {
#ifdef USE_MPI
	return 0;
#else
	return 1;
#endif
}

extern void do_quit(int argc, const char** argv, Id s);
///////////////////////////////////////////////////////////////////////
//
// Basic GenesisParser functions
//
///////////////////////////////////////////////////////////////////////
bool myFlexLexer::quit = false;

myFlexLexer::myFlexLexer( )
			: yyFlexLexer(), state(LOOKUP)
{
	currstr = "";
	BreakAllowed = 0;
	ReturnIdents = 0;
	Compiling = 0;
	InFunctionDefinition = 0;
	LocalSymbols = 0;
	GlobalSymbols.sym_entlist = 0;
	yybginidx = 0;
	continuation = 0;
	CurLocals = 0;
	script_ptr = -1;
        quit = false;
	// AddFunc("quit", do_quit, "void");
	// AddFunc("echo", do_echo, "void");
	set_float_format("%g");
}

void myFlexLexer::setElement( Id id )
{
	element_ = id;
}

///////////////////////////////////////////////////////////////////////
// Utility function
///////////////////////////////////////////////////////////////////////
int yywrap( void )
{
	return 1;
}

///////////////////////////////////////////////////////////////////////
// Functions for myFlexLexer
///////////////////////////////////////////////////////////////////////


void myFlexLexer::yyerror(char* s)
{
	cerr << "Error: " << s << std::endl;
}

struct winfo {
	char* word_name;
	int word_type;
	struct winfo* next;
};

struct winfo* word_list; 	/* first element in word list */

extern void* malloc();

int myFlexLexer::add_word(int type, char* word)
{
	struct winfo* wp;
	if (lookup_word(word) != LOOKUP) {
		cerr << "Warning: Word " << word << "already defined " << std::endl;
		return 0;
	}

	wp = (struct winfo*) malloc (sizeof(struct winfo) );
	wp->next = word_list;

	wp->word_name = (char *)malloc(strlen(word) + 1);
	strcpy(wp->word_name, word);
	wp->word_type = type;
	word_list = wp;
	return 1;
}

int myFlexLexer::lookup_word(char* word)
{
	struct winfo* wp = word_list;

	for(; wp; wp = wp->next) {
		if (strcmp(wp->word_name, word) == 0)
			return wp->word_type;
	}
	return LOOKUP;	
}

// This becomes a blocking call if threads are not enabled.
void myFlexLexer::Process() {
	int i;
	while (1) {
#if MOOSE_THREADS		
		int status = pthread_mutex_lock(&mutex);
		pthread_cond_wait(&cond, &mutex);
		status = pthread_mutex_unlock(&mutex);
#else
		string s;
		// cin >> s;
		cout << "moose > ";
		std::getline( cin, s );
		currstr += s + "\n";
#endif
		while (currstr.length() > 0) {
                    cerr << "in FlexLexer::Proc: yyparse() = " << currstr << " # " << std::endl;                    
			i =  yyparse();                        
			cerr << "in FlexLexer::Proc: yyparse() = " << currstr << " # " << i << std::endl;
		}
		// yyparse();
		/*
		while ((i = yyparse() ) )
			cerr << "Proc: i = " << i << std::endl;
			*/
	}
}

void myFlexLexer::ParseInput(const string& s) {
	string temp = s;
	unsigned int len = temp.length();
	if ( len > 2 && temp[ len - 2 ] == '\\' ) {
		temp = temp.substr( 0, len - 2 );
		currstr += temp;
		return;
	}
	currstr += temp + "\n";
	while (!quit && currstr.length() > 0 ) {            
            yyparse();
	}
}

void myFlexLexer::AddInput(const string& s) {
#if MOOSE_THREADS
	int status = pthread_mutex_lock(&mutex);
	string temp = s;
	unsigned int len = temp.length();
	if ( len > 2 && temp[ len - 2 ] == '\\' ) {
		temp = temp.substr( 0, len - 2 );
		currstr += temp;
		return;
	}
	currstr += temp + "\n";
	status = pthread_cond_signal(&cond);
	status = pthread_mutex_unlock(&mutex);
#else
	currstr += s + "\n";
#endif
}

// The LexerInput has three states:
// 1. Normal text input
// 2. Input when the Lexer is handling a compound (multiline) statement
// 3. Script file input.
//
// 1. Normal text input. This happens when the object receives an
// AddInput message. It wakes up the parser thread (from Process)
// and calls yyparse, which then examines the LexerInput. It goes
// back to sleep once done.
// 2. Input when Lexer is handling compound statement, for 
// example while, for, if, etc. Here the parser cannot terminate till
// the statement is done. So LexerInput goes to sleep internally,
// waiting for an AddInput message.
// 3. Script file input. When there are script files pending the
// LexerInput reads from these files till they are done. During
// this phase it is a bad idea to accept text input, so that is
// ignored. It can build up in the background, though.

int myFlexLexer::LexerInput( char* buf, int max_size ) {
			// cerr << "In Lexer Input\n";
			// Wait for input
			/*
			int status = pthread_mutex_lock(&mutex);
			pthread_cond_wait(&cond, &mutex);
			status = pthread_mutex_unlock(&mutex);
			*/

	while (1) {
		Script* s = CurrentScript();
		if (s) {
			if (s->file) {
				char* ret = fgets(buf, max_size, s->file);
				if (ret) {
					int len = strlen(buf);
					if (len > 0)
						return len;
				}
				// This cleans up the files and arguments
				NextScript();
				buf[0] = '\200';
				buf[1] = '\0';
				buf[2] = '\0';
				return 1;
			} else {
				// Not really sure when we would run into an empty file
				cerr << "Error: LexerInput: Empty Script file\n";
				script_ptr--;
				continue;
			}
		}
		int ncpy = currstr.length() + 1;
		int remaining = ncpy - max_size;
		if (ncpy > 1) {
			if (max_size < ncpy) {
				ncpy = max_size;
			}
			strncpy(buf, currstr.c_str(), ncpy - 1);
			buf[ncpy - 1] = '\0';
			if (remaining > 0)
				currstr = currstr.substr(ncpy - 1);
			else
				currstr = "";
			// cerr << "In Lexer Input: returning " << ncpy << std::endl;
			return ncpy - 1;
		}
		if (Compiling || InFunctionDefinition) {
			// Don't return, instead get more input.
#if MOOSE_THREADS
			int status = pthread_mutex_lock(&mutex);
			pthread_cond_wait(&cond, &mutex);
			status = pthread_mutex_unlock(&mutex);
#else
			string s;
			// cin >> s;
			std::getline( cin, s );
			currstr += s + "\n";
#endif
		} else {
			return 0;
		}
	}
	// cerr << "In Lexer Input: returning zero\n";
	return 0;
}

const string myFlexLexer::GetOutput() {
	string ret = outstr;
	outstr = "";
	return ret;
}

void myFlexLexer::LexerOutput( const char* buf, int size ) {
			// cerr << "In Lexer Output\n";
			// Cannot simply append buf, because it may have 
			// null terminated chars.
			int maxlen = size + outstr.length() + 1;
			char* s = new char[maxlen];
			strcpy(s, outstr.c_str());
			strncpy(s + outstr.length(), buf, size);
			outstr = s;
			delete[] s;
}



///////////////////////////////////////////////////////////////////

/*
** Routine to handle C comments.
*/

void myFlexLexer::Ccomment()
{
        char c, c1;

loop:
        while ((c = (char)yyinput()) != '*' && c != 0)
                ;

        if ((c1 = (char)yyinput()) != '/' && c != 0)
	{
		yyunput(c1, yytext);
                goto loop;
	}
}

///////////////////////////////////////////////////////////////////
// Here we tap into the GenesisParserWrapper::print command
///////////////////////////////////////////////////////////////////

void myFlexLexer::print( const string& s )
{
	// Element* e = Element::element( element_ );
	Element* e = element_();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data( 0 ) );
	gpw->print( s );
}

///////////////////////////////////////////////////////////////////
// 
// Here we do some commands that do need to be altered from SLI
// to fit in the MOOSE framework
//
///////////////////////////////////////////////////////////////////

void myFlexLexer::AddFunc(const char* name, slifunc func, const char* type)
{
	func_entry *fe = new func_entry(func, type);
	const string sname(name);
	if (fe) {
		func_map[sname] = fe;
	}
}

void myFlexLexer::alias(const string& alias, const string& old )
{
	if ( old.length() > 0 && alias.length() > 0 ) {
		Func_map::iterator i;
		i = func_map.find( old );
		if ( i != func_map.end() ) {
			func_map[ alias ] = i->second;
			alias_map[ alias ] = old;
		}
	} else if ( alias.length() > 0 ) { // Show this alias.
		map< string, string >::iterator i;
		i = alias_map.find( alias );
		if ( i != alias_map.end() )
			print( i->second );
	} else { // list all aliases
		map< string, string >::iterator i;
		for ( i = alias_map.begin(); i != alias_map.end(); i++ )
			print( i->first + "	" + i->second );
	}
}

void myFlexLexer::listCommands( )
{
	Func_map::iterator i;

	for ( i = func_map.begin(); i != func_map.end(); i++ ) {
		print( i->first );
	}
}

int myFlexLexer::IsCommand(const char* name)
{
	Func_map::iterator i;
	i = func_map.find(name);
	// cout << "In IsCommand for '" << name << "'... ";
	if (i != func_map.end()) {
		// cout << "Found it!!\n";
		return 1;
	}
	/*
	cout << "Error: in IsCommand: Did not find command '" <<
		name << "'\n";
	*/
	return 0;
}

func_entry* myFlexLexer::GetCommand(const char* name)
{
	Func_map::iterator i;
	i = func_map.find(name);
	if (i != func_map.end()) {
		return (*i).second;
	}
	return 0;
}

		void ListCommands();


/*
static int bGCTrace = FALSE;


int GetCommandTraceLevel(void)
{
    return(bGCTrace);
}

int SetCommandTraceLevel(int iLevel)
{
    bGCTrace = !!iLevel;
}
*/

Result func_entry::Execute(int argc, const char** argv, Id s)
{
	Result		result;
	assert( s != Id() );
	if ( type == "int") {
	    result.r_type = IntType();
	    result.r.r_int = ((PFI)func)( argc, argv, s );
	} else
	if (type == "float") {
	    // ffunc = (PFF)func;
	    result.r_type = FloatType();
	    result.r.r_float = ((PFF)func)(argc, argv, s);
	} else
	if(type == "double") {
	    // dfunc = (PFD)func;
	    result.r_type = FloatType();
	    result.r.r_float = ((PFD)func)( argc, argv, s );
	} else
	if(type == "char*") {
	    // cfunc = (PFC)func;
	    result.r_type = StrType();
	    result.r.r_str = ((PFC)func)( argc, argv, s );
	} else
	if(type == "char**") {
	    // cfunc = (PFC)func;
	    result.r_type = ArgListType();
	    result.r.r_str = ((PFC)func)( argc, argv, s );
	} else {
	    func( argc, argv, s );
            if (func == do_quit){
                myFlexLexer::doQuit(true);
            }
            result.r_type = IntType();
            result.r.r_int = 0;
            
	}
	return(result);
}

Result myFlexLexer::ExecuteCommand(int argc, char** argv)
{
FILE		*pfile;
// int 		code;
short 		redirect_mode = 0;
int 		start = 0;
int 		i;
const char*	mode = "a";
Result		result;
// int		ival;
func_entry	*command;

    result.r_type = IntType();
    result.r.r_int = 0;
    if(argc < 1){
	return(result);
    }
    /*
    ** is this an simulator shell function?
    */
    command = GetCommand(argv[0]);
	if (command && command->HasFunc() ) {
	/*
	** check through the arg list for stdout
	** redirection
	*/
	for(i=0;i<argc;i++){
	    /*
	    ** check for stdout redirection
	    */
	    if(i+1 < argc && ((strcmp(argv[i],"|") == 0) ||
	   (strcmp(argv[i],">") == 0) ||
	   (strcmp(argv[i],">>") == 0))){
		start = i+1;
		if(strcmp(argv[i],"|") == 0){
		    /*
		    ** pipe
		    */
		    redirect_mode = 1;
		    mode = "w";
		}
		if(strcmp(argv[i],">") == 0){
		    /*
		    ** redirect stdout to a file
		    */
		    redirect_mode = 2;
		    mode = "w";
		}
		if(strcmp(argv[i],">>") == 0){
		    /*
		    ** append stdout to a file
		    */
		    redirect_mode = 2;
		    mode = "a";
		}
		break;
	    }
	}
	if(redirect_mode){
		cerr << "Error: Redirection not yet working in MOOSE\n";
	// Here we have a lot of OS-specific stuff. Will deal with later.
#if 0
	    FILE *fp = NULL;
	    FILE *output;
	    int   savefd;
	    int   stdoutfd = fileno(stdout);

	    normal_tty();
	    /*
	    ** save the stdout FILE structure
	    */
	    switch(redirect_mode){
	    case 1:
		/*
		** open up the pipe
		*/
		fp = G_popen(ArgListToString(argc-start,argv+start),mode);
		break;
	    case 2:
		/*
		** open up the file
		*/
		fp = fopen(argv[start],mode);
		break;
	    }
	    if (fp == NULL){ /* G_popen or fopen failed!!! */
		genesis_tty();
		perror(argv[start]);
		return(result);
	    }
	    /*
	    ** flush the existing data in stdout
	    */
	    fflush(stdout);
	    savefd = dup(stdoutfd);
	    close(stdoutfd);
	    dup(fileno(fp));
	    /*
	    ** call the function
	    */
		command->Execute(start - 1, argv);
	    // func(start-1,argv);
	    /*
	    ** flush the output and close it
	    */
	    fflush(stdout);
	    close(stdoutfd);
	    switch(redirect_mode){
	    case 1:
		G_pclose(fp);
		break;
	    case 2:
		fclose(fp);
		break;
	    }
	    /*
	    ** restore the stdout file structure
	    */
	    dup(savefd);
	    close(savefd);
	    genesis_tty();
#endif
	} else {
	/*
	** call the function
	*/
            result = command->Execute( argc, (const char**)argv, element_ );
        }
        if (myFlexLexer::quit){
            EndScript();
        }
        return result;

    } else 
    /*
    ** is it a simulator shell script?
    */
    if(ValidScript(pfile = SearchForScript(argv[0],"r"))){
	AddScript(NULL, pfile, argc, argv, FILE_TYPE);
    } else {
	/*
	** if the function is not locally defined then
	** check to see if system functions should be
	** tried
	*/
	/**
	 * May 2008. USB
	 * This is risky in MPI environments, because they tend to barf
	 * at system and fork calls. Autoshell should return 0 for these
	 * cases.
	 * Also these calls don't work in non-Unix OSs.
	 */
		if( Autoshell() ) {
			int code;
	    	if((code = ExecFork(argc,argv)) != 0){
				printf("Error: System call returned code %d\n", code);
	    	};
	    		result.r_type = IntType();
	    	result.r.r_int = code;
	    	return(result);
		} 
	}
    result.r_type = IntType();
    result.r.r_int = 0;
    return(result);
}



void myFlexLexer::AddScript(
	char* ptr, FILE* fp, int argc, char** argv, short type)
{
    if ((type == FILE_TYPE && fp == NULL)
	|| (type == STR_TYPE && ptr == NULL)) {
	return;
    }

    if (script_ptr+1 < MAXSCRIPTS) {
	script_ptr++;
	script[script_ptr].ptr = ptr;
	script[script_ptr].file = fp;
	script[script_ptr].current = ptr;
	script[script_ptr].type = type;
	script[script_ptr].argc = argc;
	script[script_ptr].argv = CopyArgv(argc,argv);
	script[script_ptr].line = 0;
    } else {
	// Error();
	cerr << "file script stack overflow" << std::endl;
    }
    if(argc > 0)
	PushLocalVars(argc,argv,NULL);
}

Script *myFlexLexer::NextScript()
{
	// Note subtle but important change from GENESIS version:
	// Here the script ptr is >= 0, since we do not need to
	// reserve the zeroth one for the tty io.
	if (script_ptr >= 0) {
	/*
	** close the current file
	*/
		if (script[script_ptr].type == FILE_TYPE){
			fclose(script[script_ptr].file);
		} else {
			script[script_ptr].current = script[script_ptr].ptr;
		}
		if (script[script_ptr].argc > 0) {
			PopLocalVars();
		}
		FreeArgv(script[script_ptr].argc, script[script_ptr].argv);
		return(&(script[--script_ptr]));
	} else {
		return(NULL);
	}
}

int myFlexLexer::IncludeScript(int argc, char** argv)
{
FILE	*pfile;

    /*
    ** try to open it as a script
    */
    if((pfile = SearchForScript(argv[0],"r")) != NULL){
	AddScript((char *)NULL, pfile, argc, argv, FILE_TYPE);
	return(1);
    } else {
	return(0);
    }
}

void myFlexLexer::doQuit(bool quit)
{
    myFlexLexer::quit = quit;
}
    
