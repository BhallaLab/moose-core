// Based on genesis/src/ss/parse.c

#include <iostream>
#include <map>
#include <vector>
#include <setjmp.h>
using namespace std;
#include <FlexLexer.h>
#include "header.h"
#include "script.h"
#include "GenesisParser.h"

/*
** parse.cpp
**
**	Routines involved in parse tree construction.
*/

ParseNode *PTNew(int type, ResultValue data, ParseNode* left, ParseNode* right)
{	/* PTNew --- Make a new parse tree node and initialze fields */

//        extern SIGTYPE sig_msg_restore_context();
	ParseNode	*pn;

	pn = (ParseNode *) malloc(sizeof(ParseNode));
	if (pn == NULL)
	  {
	    perror("PTNew");
	    // sig_msg_restore_context(0, errno);
	    /* No Return */
	  }

	pn->pn_val.r_type = type;
	pn->pn_val.r = data;
	pn->pn_left = left;
	pn->pn_right = right;

	return(pn);

}	/* PTNew */


void PTFree(ParseNode* pn)
{	/* PTFree --- Free an entire parse tree given root */

     //   int PTFree();

	if (pn != NULL)
	  {
	    PTFree(pn->pn_left);
	    PTFree(pn->pn_right);
	    free(pn);
	  }

}	/* PTFree */
