////////////////////////////////////////////////////////////////////
//
// exec_fork.cpp: Based closely on system.c from GENESIS
//
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>
// #include "system_deps.h"
#if defined(unix) || defined(__unix__) || defined(__unix)
#include <unistd.h>
#include <sys/wait.h>
#endif
extern int debug;

void Beep(){
    putchar('\007');
}

int ExecFork( int argc, char** argv)
{
#if defined(unix) || defined(__unix__) || defined(__unix)
int pid;
int status;
char	*newargv[4];
char	string[1000];
int	i;

    /*
    ** prepare the argument list for the fork
    */
    char *shell = getenv("SHELL");
    if (shell)
        newargv[0] = shell;
    else
        newargv[0] = "sh"; // I believe sh is the most commonly
                           // encountered shell in Unices. - Subha
    newargv[1] = "-c";
    string[0] = '\0';
    for(i=0;i<argc;i++){
	strcat(string,argv[i]);
	strcat(string," ");
    }
    newargv[2] = string;
    newargv[3] = NULL;
    /*
    ** do the fork
    */
    pid = vfork();
	// could be fork if vfork doesn't work
    if(pid == -1){
	printf("run: fork unsuccessful in ExecFork() for %s\n",string);
	_exit(0);
    } else
    if(pid ==0){
	/*
	** pid = 0 indicates that this is the child resulting
	** from the fork so execute the program
	** which overlays the current process and therefore
	** does not return
	*/
	execvp(newargv[0],newargv);
	/*
	** if the exec fails then exit the child
	*/
	printf("unable to execute '%s'\n",argv[0]);
	_exit(0);
    }
#ifdef DEBUG
    printf("waiting for child process %d\n",pid);
#endif
    /*
    ** pid > 0 indicates successful child has been forked
    ** so wait for it to complete execution and return the status
    */
    while(wait(&status) != pid);
    // if(debug){
	// printf("child process %d done. Status = %d\n", pid, status);
    // }
    return(status);

#else
  return -1; // not a unix system - do nothing
#endif // unix
}
