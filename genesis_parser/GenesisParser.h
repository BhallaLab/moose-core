/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
** Large parts of this individual file have been copied from files
** in the genesis/src/ss directory. These are authored by 
** Dave Bilitch, and copyright Caltech
** under terms which permit free redistribution. Note that these
** files are NOT GPL.
** Additional modifications and conversion to C++ are
**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
**********************************************************************/

#ifndef _GENESIS_PARSER_H
#define _GENESIS_PARSER_H

#define LOOKUP 0
#define BGINSTKSIZE 100
#define MAXSCRIPTS 40

// typedef unsigned int Id;

#include <map>
#include <string>
#include "basecode/Id.h"
#include "script.h"

// Particularly ugly macro definition 
//#define BEGIN yy_start = 1 + 2 *

typedef struct _localval
  {
    short   l_type;
    short   l_offs;
  } LocalVal;
    
typedef union _resultvalue
  { 
    int     r_int;
    double  r_float;
    char    *r_str;
    LocalVal    r_loc;
  }ResultValue;

typedef struct _result
  {
    int     r_type;
    ResultValue r;
  } Result;

typedef struct _pn
  {
    Result  pn_val;
    struct _pn  *pn_left;
    struct _pn  *pn_right;
  } ParseNode;


typedef union _yyvals
  {
    int     iconst;
    double  fconst; 
    char    *str;
    ParseNode   *pn;
  } YYSTYPE;
#define YYSTYPE_IS_DECLARED 1


typedef struct _symtabent
  {
    char        *sym_ident;
    Result      sym_val;
    struct _symtabent   *sym_next;
  } SymtabEnt;

typedef struct _symtab 
  {
    SymtabEnt   *sym_entlist;
  } Symtab; 


typedef struct _script_info
  {
    int     si_lineno;
    char    *si_script;
    int     si_refcnt;
  } ScriptInfo;


typedef struct _localvars
  {
    int         lv_nargs;
    int         lv_nlocals;
    Result      *lv_vars;
    struct _localvars   *lv_prev;
    char        *lv_function;
    ScriptInfo      *lv_si; 
  } LocalVars;

extern Result* SymtabLook(Symtab* symtab, const char* sym);
extern "C" int yywrap( void );
typedef void (*slifunc)(int argc, const char** argv, Id s );
typedef int (*PFI)(int argc, const char** argv, Id s );
typedef char* (*PFC)(int argc, const char** argv, Id s );
typedef float (*PFF)(int argc, const char** argv, Id s );
typedef double (*PFD)(int argc, const char** argv, Id s );

class func_entry 
{
	public:
		func_entry(slifunc func_arg, const char* type_arg)
			: func(func_arg), type(type_arg)
		{
			;
		}

		Result Execute(int argc, const char** argv, Id s );
		bool HasFunc() {
			return (func != 0);
		}

	private:
		slifunc func;
		std::string type;
};

typedef std::map< std::string, func_entry* > Func_map;

class myFlexLexer: public yyFlexLexer
{
	public:
		myFlexLexer();
		
		int yylex();

		int state;
		int yyparse();
		static void yyerror(char* s);

		int add_word(int type, char* word);
		int lookup_word(char* word);
		int yychar; // Lookahead symbol
		YYSTYPE yylval; // Semantic value of lookahead symbol
		int yynerrs; // Number of parse errors.
		int yylloc; // Location data for lookahead symbol

		void AddInput(const std::string& s);
		void Process();
		void ParseInput(const std::string& s);

		const std::string GetOutput();
		void alias(const std::string& alias, const std::string& old );
		void listCommands();

		void setElement( Id id );
                static void doQuit(bool quit);
	protected:
		int LexerInput( char* buf, int max_size );

		void LexerOutput( const char* buf, int size );
		void Ccomment();
		void Pushyybgin(int start);
		void Popyybgin();
		void ParseInit();
		int	NestedLevel();
		ParseNode* vardef(char* ident, int type,
			int castop, ParseNode* init);
		int nextchar(int flush);
		ScriptInfo* MakeScriptInfo();
		int CurrentScriptLine() {
			if (script_ptr >= 0)
				return script[script_ptr].line;
			return -1;
		}
		char* CurrentScriptName() {
			if(script_ptr >= 0 && script[script_ptr].argc > 0)
				return(script[script_ptr].argv[0]);
			return 0;
		}
		Script* CurrentScript()
		{
			if (script_ptr >= 0)
				return(&script[script_ptr]);
			return 0;
		}
		void do_listglobals(int argc, char* argv[]);
		void PushLocalVars(int argc, char* argv[], Symtab* symtab);
		void PopLocalVars();
		void do_include_args(char* script, ParseNode* args);
		int ExecuteFunction(int argc, char* argv[]);
		char* ExecuteStrFunction(int argc, char* argv[]);
		Result do_func(ParseNode* func, ParseNode*  args);
		Result do_funcwithnode(char* cmdleader, ParseNode* args,
			ParseNode* argcomplist);
		char** PTArgv(int arg);
		int PTArgc();
		void TraceScript();
		void SetLine(ScriptInfo* si);
		Result* locref(int offset);
		Result dollarref(int offset);
		Result PTEval(ParseNode* pn);
		void PTCall(ParseNode* pn);
		Result do_cmd(
			char* cmdname, ParseNode* args, short do_autoshell);
		char* do_cmd_args(ParseNode* arg, int* argc, char* argv[]);

		double GetScriptDouble(const char*);
		void SetScriptDouble(char*, double);
		int GetScriptInt(char*);
		void SetScriptInt(char*, int);
		char* GetScriptStr(char*);
		void SetScriptStr(char*, char*);
		void CreateScriptFloat(char*);
		void CreateScriptInt(char*);
		void CreateScriptString(char*);
		float get_script_float(char* name); // dunno why it differs
		void set_script_float(char* name, float); //dunno why it differs
		char *get_glob_val(char* name);
		int do_addglobal(int argc, char* argv[]);
		int do_setglobal(int argc, char* argv[]);
		char* do_getglobal(int argc, char* argv[]);
		Result do_foreach_arg(
			ParseNode* arg, ParseNode* body, Result* rp);
		void do_foreach(ParseNode* arg, ParseNode* body, Result* rp);

		void CompileScriptVars(FILE* fs, char* leader);

		void PTInit();
		int IsSilent() {
			return 0; // later put in a value.
		}

		void AddFunc(const char* name, slifunc func, const char* type);
		int IsCommand(const char* name);
		func_entry* GetCommand(const char* name);
		virtual Result ExecuteCommand(int argc, char** argv);
		void AddScript(char* ptr, FILE* fp,
			int argc, char** argv, short type);
		Script *NextScript();
		void EndScript()
		{
    		NextScript();
		}
		int IncludeScript(int argc, char** argv);
		void print( const std::string& s );

		Id element() const {
			return element_;
		}

		Id element_;

		Symtab* getGlobalSymbols() {
			return &GlobalSymbols;
		}

	private:
		std::string currstr;
		std::string outstr;
#if MOOSE_THREADS
		pthread_mutex_t mutex;
		pthread_cond_t  cond;
#endif
 
//		jmp_buf  BreakBuf;   /* Used to break out of a loop */ 
//		jmp_buf  ReturnBuf;  /* Used to return out of a script */

		int  DefType;    /* Remembers type in a var decl */ 
		int  DefCast;    /* Remembers cast in a var decl */ 
		int  BreakAllowed; /* In a loop control structure */
		int  ReturnIdents; /* 1 ==> lexer doesn't classify IDENTs */
		int  Compiling;  /* Make a statement list rather than execute */
		int  InFunctionDefinition;
		int  NextLocal;  /* Next local symbol offset */
		int  ArgMatch;   /* Matches argument number to name in func */
		Symtab  GlobalSymbols;	/* Symbols defined outside a function */
		Symtab*	LocalSymbols;   /* Symbols local to a function */ 
		LocalVars* CurLocals;
		ResultValue RV;          /* Dummy ReturnValue for PTNew */
		int yybginstk[BGINSTKSIZE];
		int yybginidx;
		int continuation;
		jmp_buf BreakBuf; 	/* Used to break out of a loop */
		jmp_buf ReturnBuf; 	/* Used to break out of a script func */
		Func_map func_map;
		std::map< std::string, std::string > alias_map;
		short script_ptr;
		Script script[MAXSCRIPTS];
                static bool quit;
};

#endif // _GENESIS_PARSER_H
