// Getopt.cpp
// Derived very closely from genesis/src/sys/getopt.c
// Written by Dave Bilitch at Caltech.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// #include "header.h"

#define OPTLIKE(arg) \
	(arg[0] == '-' && 'a' <= arg[1] && arg[1] <= 'z')

int	G_optind;
int	G_opterror;
char*	G_optopt;
int	optargc;
char**	optargv;

typedef struct _opt
  {
    char*	name;
    int		staticargs;
    int		additionalargs;
    struct _opt	*next;
  } Opt;

typedef struct _cmd
  {
    char*	mem;
    char*	optstring;
    int		staticargs;
    int		additionalargs;
    Opt*	options;
  } Cmd;

static Cmd optcmd;

static void uninitopt()

{	/* uninitopt --- Free memory from previous options processing */

	if (optcmd.mem != NULL)
	  {
	    free(optcmd.mem);
	    optcmd.mem = NULL;
	  }

	while (optcmd.options != NULL)
	  {
	    Opt*	opt;

	    opt = optcmd.options;
	    optcmd.options = opt->next;
	    free(opt);
	  }

}	/* uninitopt */



int initopt(int argc, char* argv[], char* optstring)
{	/* initopt --- Digest optstring and prepare to scan options */

	char*	mem;
	char*	arg;
	int	argcount;

	uninitopt();

	mem = strdup(optstring);
	if (mem == NULL)
	  {
	    perror("initopt");
	    return -1;
	  }

	optcmd.optstring = optstring;
	optcmd.mem = mem;
	optcmd.options = NULL;
	optcmd.staticargs = 0;
	optcmd.additionalargs = 0;

	for (argcount = 1; argcount < argc; argcount++)
	  {
	    if (OPTLIKE(argv[argcount]))
		break;
	  }

	optargv = argv;
	optargc = argcount;

	arg = strtok(mem, " \t\n");
	while (arg != NULL)
	  {
	    if (OPTLIKE(arg))
	      {
		Opt*	opt;

		opt = (Opt *) malloc(sizeof(Opt));
		if (opt == NULL)
		  {
		    perror("initopt");
		    uninitopt();
		    return -1;
		  }

		opt->name = arg;
		opt->staticargs = 0;
		opt->additionalargs = 0;
		opt->next = optcmd.options;
		optcmd.options = opt;

		arg = strtok(NULL, " \t\n");
		while (arg != NULL)
		  {
		    if (OPTLIKE(arg))
			break;

		    if (strcmp(arg, "...") == 0 || arg[0] == '[')
			opt->additionalargs = 1;
		    else
			opt->staticargs++;

		    arg = strtok(NULL, " \t\n");
		  }
	      }
	    else
	      {
		if (strcmp(arg, "...") == 0 || arg[0] == '[')
		    optcmd.additionalargs = 1;
		else
		    optcmd.staticargs++;

		arg = strtok(NULL, " \t\n");
	      }
	  }

	G_optind = 1;
	while (G_optind < argc && !OPTLIKE(argv[G_optind]))
	    G_optind++;
	G_opterror = 0;

	return 0;

}	/* initopt */


/*
** G_getopt
**
**	Gets the next option and associated arguments from argv.  Return
**	values are:
**
**		-1	Unknown or ambiguous option
**		-2	Incorrect number of option arguments
**		-3	Incorrect number of command arguments
**		-4	G_getopt called before initopt or after all options
**			  have been processed
**		-5	G_getopt found -help and does not match an option
**		-6	G_getopt found -usage and does not match an option
**		 0	End of command options, optargv contains the
**			  command arguments
**		 1	Valid option and arguments, optargv contains the
**			  option arguments, G_optopt contains the full option
**			  name
*/

int G_getopt(int argc, char** argv)
{	/* G_getopt --- Get next option with associated args from argv */
    
	Opt*	found;
	Opt*	opt;
	size_t	len;

	if (optcmd.mem == NULL)
	  {
	    fprintf(stderr, "G_getopt: called before initopt or after all arguments have been processed (this is a bug)\n");
	    fprintf(stderr, "        command == '%s'\n", argv[0]);
	    return -4;
	  }

	if (G_optind >= argc)
	  {
	    uninitopt();

	    optargv = argv;
	    optargc = 1;
	    while (optargc < argc && !OPTLIKE(argv[optargc]))
		optargc++;

	    if (optargc-1 < optcmd.staticargs)
	      {
		fprintf(stderr, "%s: Too few command arguments for %s\n",
		    argv[0], G_optopt);
		return -2;
	      }

	    if (optargc-1 > optcmd.staticargs && !optcmd.additionalargs)
	      {
		fprintf(stderr, "%s: Too many command arguments for %s\n",
			argv[0], G_optopt);
		return -2;
	      }

	    return 0;
	  }

	if (!OPTLIKE(argv[G_optind]))
	  {
	    fprintf(stderr, "%s: Expecting a command option, found '%s'\n",
			argv[0], argv[G_optind]);
	    return -1;
	  }

	len = strlen(argv[G_optind]);
	found = NULL;
	for (opt = optcmd.options; opt != NULL; opt = opt->next)
	  {
	    if (strncmp(opt->name, argv[G_optind], len) == 0) {
		if (found != NULL)
		  {
		    fprintf(stderr, "%s: Ambiguous command option '%s'\n",
			    argv[0], argv[G_optind]);
		    return -1;
		  }
		else
		    found = opt;
	    }
	  }

	if (found == NULL)
	  {
	    if (strncmp("-help", argv[G_optind], len) == 0)
		return -5;

	    if (strncmp("-usage", argv[G_optind], len) == 0)
		return -6;

	    fprintf(stderr, "%s: Unknown command option '%s'\n",
		    argv[0], argv[G_optind]);
	    return -1;
	  }

	G_optopt = found->name;
	optargv = argv+G_optind;

	optargc = 0;
	for (G_optind++; G_optind < argc && !OPTLIKE(argv[G_optind]); G_optind++)
	    optargc++;

	if (optargc < found->staticargs)
	  {
	    fprintf(stderr, "%s: Too few arguments to command option '%s'\n",
		    argv[0], G_optopt);
	    return -2;
	  }

	if (optargc > found->staticargs && !found->additionalargs)
	  {
	    fprintf(stderr, "%s: Too many arguments to command option '%s'\n",
		    argv[0], G_optopt);
	    return -2;
	  }

	optargc++; /* for the command option itself */

	return 1;

}	/* G_getopt */

void printoptusage(int argc, char** argv)
{	/* printoptusage --- Print a usage statement */

	if (optcmd.optstring != NULL)
	    fprintf(stderr, "usage: %s %s\n", argv[0], optcmd.optstring);

}	/* printoptusage */

typedef struct
  {
    int		optind;
    int		opterror;
    char*	optopt;
    int		optargc;
    char**	optargv;
    Cmd		cmd;
  } SaveOpts;

void *savopt()
{
        SaveOpts*	so;

	so = (SaveOpts*) malloc(sizeof(SaveOpts));
	if (so != NULL)
	  {
	    so->optind = G_optind;
	    so->opterror = G_opterror;
	    so->optopt = G_optopt;
	    so->optargc = optargc;
	    so->optargv = optargv;
	    so->cmd = optcmd;
	  }

	optcmd.mem = NULL;
	optcmd.options = NULL;

	return (void*) so;
}

void restoropt(SaveOpts* so)
{
	if (so != NULL)
	  {
	    G_optind = so->optind;
	    G_opterror = so->opterror;
	    G_optopt = so->optopt;
	    optargc = so->optargc;
	    optargv = so->optargv;
	    optcmd = so->cmd;

	    free(so);
	  }
}
