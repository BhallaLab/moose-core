// Based on shell-struct.h from GENESIS, by Dave Bilitch.


#ifndef SLI_SCRIPT_H
#define SLI_SCRIPT_H

#define         FILE_TYPE       0
#define         STR_TYPE        1


typedef struct script_type {
    char	*ptr;
    char	*current;
    short	type;
    int		argc;
    char	**argv;
    short	line;
    FILE        *file;
} Script;

extern FILE *SearchForScript(const char* name, const char* mode);
extern int ValidScript(FILE* fp);
extern void AddScript(char* ptr, FILE* fp, int argc, char** argv, short type);

/*
typedef struct func_table_type {
	char 	*name;
	PFI	adr;
	char 	*type;
} FuncTable;

typedef struct {
    char 	*name;
    char 	*type;
    short 	offset;
    short 	type_size;
    short 	field_indirection;
    short 	function_type;
    short 	dimensions;
    int 	dimension_size[4];
} Info;

typedef struct {
    PFI		function;
    int		count;
    int		priority;
} Job; 
*/


#endif // SLI_SCRIPT_H
