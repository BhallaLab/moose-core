// Taken from genesis/ss_func_ext.h

#ifndef _FUNC_EXTERNS_H
#define _FUNC_EXTERNS_H

/*
** Script variables
*/

extern double GetScriptDouble(const char* name);
extern int GetScriptInt(char* name);
extern char* GetScriptStr(char* name);

extern void SetScriptDouble(char* name, double value);
extern void SetScriptInt(char* name, int value);
extern void SetScriptStr(char* name, char* value);

extern void CreateScriptFloat(char* name);
extern void CreateScriptInt(char* name);
extern void CreateScriptString(char* name);

/*
** Various functions used in scripts and a few other places.
*/

// extern int        PTArgc();
// extern void       PTCall(ParseNode* pn);
// extern Result     PTEval(ParseNode* pn);
extern void        PTFree(ParseNode* pn);
// extern void        PTInit();
extern ParseNode *PTNew(
	int type, ResultValue data, ParseNode* left, ParseNode*  right);
extern void        FreePTValues(ParseNode* pn);

extern char *do_cmd_args(ParseNode* arg, int* argc, char* argv[]);
extern char *strsave(const char *cp);
extern char* ExecuteStrFunction(int argc, char* argv[]);
extern int   ArgListType();
extern void   CastToFloat(Result* rp);
extern void   CastToInt(Result* rp);
extern void   CastToStr(Result* rp);
extern Result ExecuteCommand(int argc, char* argv[]);
extern int   FloatType();
extern void   FreeScriptInfo(ScriptInfo* si);
extern int   InControlStructure();
extern int   IntType();
extern int   StrType();
extern void   set_float_format(const char* format);
extern char *ConvertBuf();
extern void CompileScriptVars();
extern int IsCommand(char* name);
extern int IsInclude(char* name);

extern Symtab *SymtabCreate();
extern void SymtabDestroy(Symtab* symtab);
extern Result* SymtabLook(Symtab* symtab, const char* sym);
extern Result* SymtabNew(Symtab* symtab, char* sym);
extern char* SymtabKey(Symtab* symtab, Result* rp);

extern int IncludeScript(int argc, char* argv[]);
extern void EndScript();

extern char **CopyArgv(int argc, char** argv);
extern void FreeArgv(int argc, char* argv[]);

/**
 * From exec_fork.cpp
 */
int ExecFork( int argc, char** argv);
/*
** Obsolete functions from tools library

extern float get_script_float(char* name);
extern int set_script_float(char* name, float value);
extern char* get_glob_val(char* name);

*/

#endif
