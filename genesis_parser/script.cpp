// Slightly altered from shell_script.c from the Genesis 2 
// distribution. The original version is by Dave Bilitch with
// some fixes by Mike Hucka

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "utility/utility.h"

#include "script.h"

/* mds3 changes */
/*
** Argument to 'fclose()' cast to (FILE*)
*/

/*
#define MAXSCRIPTS	20

short	script_ptr = -1;
Script	script[MAXSCRIPTS];
*/

// extern char **CopyArgv(int argc, char** argv);
// extern void FreeArgv(int argc, char* argv[]);
extern char *getenv(char* name);

int IsWhiteSpace(char c)
{
	return((c == ' ') || (c == '\t') || (c == '\n'));
}

int EmptyLine(char* s)
{
    while(*s != '\0' && (*s == ' ' || *s == '\t' || *s == '\n')) s++;
    if(*s == '\0') return(1);
    return(0);
}

int ValidScript(FILE* fp)
{
static const unsigned int SIZE = 80;
char line[80];
// char *lineptr;
unsigned int i;

    if(fp == NULL) return(0);
    /*
    ** check for the magic word at the beginning of the file
    */
    line[0] = '\0';
    while(EmptyLine(line) && !feof(fp)){
	fgets(line,SIZE,fp);
    }
    /*
    ** skip white space
    */
    i = 0;
    while(line[i] == ' ' && i < SIZE-1) i++;
    if((i < SIZE-2) && (strncmp(line+i,"//",2) == 0)){
	i += 2;
	/*
	** skip more white space
	*/
	while(line[i] == ' ' && i < SIZE-1) i++;
	if((i <= SIZE - strlen("genesis")) && 
	(strncmp(line+i,"genesis", strlen("genesis")) ==  0)) {
	    fseek(fp,0L,0);
	    return(1);
	}
    }
    fclose(fp);
    return(0);
}
#if 0
/*
** file script utilities (used by the 'source' command)
*/
void SetScript(char* ptr, FILE* fp, int argc, char** argv, short type)
{
    char **newargv;

    if ((type == FILE_TYPE && fp == NULL)
	|| (type == STR_TYPE && ptr == NULL)) {
	return;
    }

    ClearScriptStack();
    script_ptr = 0;
    script[script_ptr].ptr = ptr;
    script[script_ptr].file = fp;
    script[script_ptr].current = ptr;
    script[script_ptr].type = type;
    script[script_ptr].argc = argc;
    /*
    ** make a copy of the argv list
    */
    script[script_ptr].argv = CopyArgv(argc, argv);
    script[script_ptr].line = 0;
#ifdef NEW
    if (argc > 0)
	PushLocalVars(argc,argv,NULL);
#endif
}
#endif


#if 0
char* TopLevelNamedScriptFileName()
{
    int i;

    for (i = 0; i <= script_ptr; i++) {
	if (script[i].type == FILE_TYPE && script[i].file != stdin) {
	    return script[i].argv[0];
	}
    }
    return NULL;
}
#endif

#if 0
#ifdef OLDTRACE
TraceScript()
{
    int i;
    int once = 0;

    for (i = 0; i <= script_ptr; i++) {
	if (script[i].type == FILE_TYPE && script[i].file == stdin) {
	    continue;
	} else {
	    once = 1;
	    printf("-> ");
	    if (script[i].type == STR_TYPE){
		printf("MACRO ");
	    } 
	    if (script[i].argc > 0){
		printf(" line %d of %s ", script[i].line, script[i].argv[0]);
	    } else {
		printf(" line %d of ??? ", script[i].line);
	    }
	}
    }
    if (once) {
	printf("\n");
    }
}
#endif
#endif

#if 0
FILE *NextScriptFp()
{
    if (script_ptr > 0) {
	/*
	** close the current file
	*/
	if (script[script_ptr].type == FILE_TYPE) {
	    fclose(script[script_ptr].file);
	} else {
	    script[script_ptr].current = script[script_ptr].ptr;
	}
#ifdef NEW
	if (script[script_ptr].argc > 0) {
	    PopLocalVars();
	}
#endif
	FreeArgv(script[script_ptr].argc, script[script_ptr].argv);
	return(script[--script_ptr].file);
    } else {
	return(NULL);
    }
}
#endif

#if 0
ClearScriptStack()
{
    while(script_ptr > 0){
	NextScript();
    }
}

ScriptDepth()
{
    return(script_ptr);
}

Script *CurrentScript()
{
    if(script_ptr >= 0)
	return(&(script[script_ptr]));
    else
	return(NULL);
}

FILE *CurrentScriptFp()
{
    if (script_ptr >= 0)
	return(script[script_ptr].file);
    else
	return(NULL);
}

char *CurrentScriptName()
{
    if(script_ptr >= 0 && script[script_ptr].argc > 0)
	return(script[script_ptr].argv[0]);
    else
	return(NULL);
}

int CurrentScriptLine()
{
    if(script_ptr >= 0)
	return(script[script_ptr].line);
    else
	return(-1);
}

ScriptEnded()
{
    if (script_ptr >= 0) {
	switch(script[script_ptr].type) {
	case FILE_TYPE:
#ifdef NEW
	    if (script[script_ptr].argc > 0) {
		PopLocalVars();
	    }
#endif
	    return(feof(script[script_ptr].file));
	    /* NOTREACHED */
	    break;
	case STR_TYPE:
#ifdef NEW
	    if (script[script_ptr].argc > 0) {
		PopLocalVars();
	    }
#endif
	    return(*(script[script_ptr].current) == '\0');
	    /* NOTREACHED */
	    break;
	}
    } else
	return(0);
}

ScriptArgc()
{
    if(script_ptr >= 0)
	return(script[script_ptr].argc);
    else
	return(0);
}

char *ScriptArgv(arg)
int arg;
{
    if((script_ptr >= 0) && (arg < script[script_ptr].argc)){
	return(script[script_ptr].argv[arg]);
    } else {
	return("");
    }
}
#endif

/* 
// Moved to tp.h
void EndScript()
{
    NextScript();
}
*/

int IsFile(FILE* fp)
{
struct stat buf;
    if(!fp) return(0);
    if(fstat(fileno(fp),&buf) == -1 || buf.st_mode & S_IFDIR){
	return(0);
    } else {
	return(1);
    }
}

FILE *OpenScriptFile(const char* name, const char* mode)
{
  FILE *fp;
  string extname(name);
  size_t len;

    /*
    ** try to open the file.  Append script file extension if not present.
    */

    len = strlen(name);
    if (len < 2 || name[len-1] != 'g' || name[len-2] != '.')
      {
	extname.append(".g");
      }

    if((fp = fopen(extname.c_str(), mode)) != NULL){
	/*
	** is it a real file?
	*/
	if(!IsFile(fp)){
	    return(NULL);
	}
    }
    return(fp);
}

/*
** search the paths given in SIMPATH environment variable
** if the SIMPATH variable is not found then just search the
** current directory
*/
FILE *SearchForScript(const char* name, const char* mode)
{    
    FILE *fp;
    
    
    if(name == NULL) return(NULL);

    PathUtility pathHandler(Property::getProperty(Property::SIMPATH));
    
    fp = NULL;
    string file_name = string(name);
    fp = OpenScriptFile(name, mode);
    if (fp == NULL)
    {
        for( unsigned int i = 0; i < pathHandler.size(); ++i )
        {
            string path = pathHandler.makeFilePath(file_name, i);
            fp = OpenScriptFile(path.c_str(), mode);
            if (fp != NULL )
            {
                break;                
            }            
        }
    }
    return fp;    
}

#if 0
void do_where(argc,argv)
int argc;
char **argv;
{
char prefix[300];
char tmp[300];
char *prefixptr;
char *path;
char *env;
FILE *fp;
char *getenv();
char *name;

    initopt(argc, argv, "script");
    if (G_getopt(argc, argv) != 0)
      {
	printoptusage(argc, argv);
	return;
      }

    name = optargv[1];
    env = path = getenv("SIMPATH");
    fp = NULL;
    if(path == NULL){
	/*
	** just look it up in current directory
	*/
	fp = fopen(name,"r");
	if(fp != NULL){
	    fclose(fp);
	    printf("'%s' found in current directory \n",name);
	    return;
	} else {
	    printf("could not find '%s' in current directory\n",name);
	    return;
	}
    } else {
	/*
	** follow the paths in order trying to find the file
	*/
	while(fp == NULL && path != NULL && *path != '\0'){
	    /*
	    ** skip white space
	    */
	    while(IsWhiteSpace(*path) && *path != '\0') path++;
	    /*
	    ** copy the path up to a space
	    */
	    prefixptr = prefix;
	    while(!IsWhiteSpace(*path) && *path != '\0') {
		*prefixptr++ = *path++;
	    }
	    *prefixptr = '\0';
	    if(prefix[0] != '\0'){
		strcpy(tmp,prefix);
		strcat(tmp,"/");
		strcat(tmp,name);
		fp = fopen(tmp,"r");
	    }
	    if(fp != NULL){
		printf("'%s' found in %s\n",name,prefix);
		fclose(fp);
		return;
	    }
	}
    }
    printf("could not find '%s' in %s\n",name,env);
}

void do_source(argc,argv)
int argc;
char **argv;
{
FILE	*pfile;

    if(argc < 2){
	printf("usage: %s script [script-arguments]\n",argv[0]);
	return;
    }
    /*
    ** try to open the named script
    */
    if((pfile = SearchForScript(argv[1],"r")) != NULL){
	if(debug > 0){
	    printf("source %s\n",argv[1]);
	}
	/*
	** activate the script and pass it the remaining
	** arguments on the command line
	*/
	AddScript((char *)NULL, pfile, argc - 1, argv + 1, FILE_TYPE);
    } else {
	Error();
	printf("unable to open file %s\n",
	argv[1]);
    }
}
#endif

int IsInclude(char* s)
{
FILE	*pfile;

    /*
    ** try to open it as a script
    */
    if((pfile = SearchForScript(s,"r")) != NULL){
	fclose(pfile);
	return(1);
    } else {
	return(0);
    }
}


#if 0
/*
** search the paths given in PATH environment variable
** if the PATH variable is not found then just search the
** current directory
*/
FILE *SearchForExecutable(name,mode,pathname)
char *name;
char *mode;
char **pathname;
{
char prefix[300];
char *prefixptr;
char *path;
FILE *fp;
char *getenv();

    if(name == NULL) return(NULL);
    path = getenv("PATH");
    fp = NULL;
    /*
    ** if there is no search path defined or it is an absolute
    ** or relative path then dont search
    */
    if((path == NULL) || (strchr(name, '/') != NULL)){
	/*
	** just look it up in current directory
	*/
	if(!IsFile(fp = fopen(name,mode))){
	    if (fp != NULL){
		fclose(fp);
		fp = NULL;
	    }
	} else {
	    *pathname = CopyString(name);
	}
    } else {
	/*
	** follow the paths in order trying to find the file
	*/
	while(fp == NULL && path != NULL && *path != '\0'){
	    /*
	    ** skip white space
	    */
	    while(IsWhiteSpace(*path) && *path != '\0') path++;
	    /*
	    ** copy the path up to a space
	    */
	    prefixptr = prefix;
	    while(!IsWhiteSpace(*path) && *path != '\0' &&
	    *path !=':') {
		*prefixptr++ = *path++;
	    }
	    if(*path == ':'){
		path++;
	    }
	    *prefixptr = '\0';
	    /*
	    ** now try to open the file
	    */
	    if(prefix[0] != '\0'){
		strcat(prefix,"/");
	    }
	    strcat(prefix,name);
	    if(!IsFile(fp = fopen(prefix,mode))){
		if(fp != NULL){
		    fclose(fp);
		    fp = NULL;
		}
	    } else {
		*pathname = CopyString(prefix);
	    }
	}
    }
    return(fp);
}

/*
** Searching for non-script files like the colorscales.
** search the paths given in SIMPATH environment variable
** if the SIMPATH variable is not found then just search the
** current directory
*/
FILE *SearchForNonScript(char* name, char* mode)
{
char prefix[300];
char *prefixptr;
char *path;
FILE *fp;
char *getenv();

    if(name == NULL) return(NULL);
    path = getenv("SIMPATH");
    fp = NULL;
    /*
    ** if there is no search path defined or it is an absolute
    ** or relative path then dont search
    */
    if((path == NULL) || (strchr(name, '/') != NULL)){
	/*
	** just look it up in current directory
	*/
	fp = fopen(name,mode);
    } else {
	/*
	** follow the paths in order trying to find the file
	*/
	while(fp == NULL && path != NULL && *path != '\0'){
	    /*
	    ** skip white space
	    */
	    while(IsWhiteSpace(*path) && *path != '\0') path++;
	    /*
	    ** copy the path up to a space
	    */
	    prefixptr = prefix;
	    while(!IsWhiteSpace(*path) && *path != '\0') {
		*prefixptr++ = *path++;
	    }
	    *prefixptr = '\0';
	    /*
	    ** now try to open the file
	    */
	    if(prefix[0] != '\0'){
		strcat(prefix,"/");
		strcat(prefix,name);
		fp = fopen(prefix,mode);
	    }
	}
    }
    return(fp);
}


/*
** Searching for a file on the genesis path, returns the
** full pathname of the file
*/
char *FindFileOnPath(char* name)
{
static char prefix[300];
char *prefixptr;
char *path;
FILE *fp;
char *getenv();
 
    if(name == NULL) return(NULL);
    path = getenv("SIMPATH");
    fp = NULL;
    /*
    ** if there is no search path defined or it is an absolute
    ** path then dont search
    */
    if((path == NULL) || (name[0] == '/')){
    /*
    ** just look it up in current directory
    */
    fp = fopen(name,"r");
    if (fp) {
        /* This is stupid, opening a file and then closing it,
        ** but it is easy */
        fclose(fp);
        return(name);
    }
    } else {
    /*
    ** follow the paths in order trying to find the file
    */
    while(fp == NULL && path != NULL && *path != '\0'){
        /*
        ** skip white space
        */
        while(IsWhiteSpace(*path) && *path != '\0') path++;
        /*
        ** copy the path up to a space
        */
        prefixptr = prefix;
        while(!IsWhiteSpace(*path) && *path != '\0') {
        *prefixptr++ = *path++;
        }
        *prefixptr = '\0';
        /*
        ** now try to open the file
        */
        if(prefix[0] != '\0'){
        strcat(prefix,"/");
        strcat(prefix,name);
        fp = fopen(prefix,"r");
        if (fp) {
            fclose(fp);
            return(prefix);
        }
        }
    }
    }
    return(NULL);
}
#endif



void FreeArgv(int argc, char* argv[])
{	/* FreeArgv --- Free arg list memory and args */

	int	arg;

	if (argv == NULL)
	    return;

	for (arg = 0; arg < argc; arg++)
	    if (argv[arg] != NULL)
		free(argv[arg]);

	free(argv);

}	/* FreeArgv */






char **CopyArgv(int argc, char** argv)
{
char **newargv;
int i;
size_t	length;

    if(argc == 0) return(NULL);
    newargv = (char **)calloc(argc +1,sizeof(char *));
    for(i=0;i<argc;i++){
	length = strlen(argv[i]);
	newargv[i] = (char *)malloc(length + 1);
	strcpy(newargv[i],argv[i]);
    }
    return(newargv);
}

