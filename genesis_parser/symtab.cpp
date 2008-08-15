#include <iostream>
#include <map>
#include <vector>
using namespace std;

#include <setjmp.h>
#include <FlexLexer.h>
#include "header.h"
#include "script.h"
// #include "Shell.h"
#include "GenesisParser.h"

/*
** symtab.cpp
** basec closely on symtab.c from GENESIS.
**
**	Symbol table routines.
*/

Symtab *SymtabCreate()
{	/* SymtabCreate --- Create an empty symbol table */
	Symtab	*symtab = new Symtab();
	symtab->sym_entlist = 0;
	return(symtab);
}	/* SymtabCreate */



void SymtabDestroy(Symtab *symtab)
{	/* SymtabDestroy --- Destroy symbol table */

	SymtabEnt	*symtabent;

	symtabent = symtab->sym_entlist;
	while (symtabent != 0)
	  {
	    SymtabEnt	*next;

	    next = symtabent->sym_next;
	    delete symtabent->sym_ident;
	    delete symtabent;

	    symtabent = next;
	  }
	 delete symtab;
}	/* SymtabDestroy */


Result *SymtabLook(Symtab* symtab, const char* sym)
{	/* SymtabLook --- Look for symbol table entry for sym */

	SymtabEnt	*symtabent;

	if (symtab == 0)
	    return(0);

	symtabent = symtab->sym_entlist;
	while (symtabent != 0)
	  {
	    if (strcmp(symtabent->sym_ident, sym) == 0)
		return(&symtabent->sym_val);

	    symtabent = symtabent->sym_next;
	  }
	return(0);

}	/* SymtabLook */



char *SymtabKey(Symtab* symtab, Result* rp)
{	/* SymtabKey --- Return sym_ident for given result address */

	SymtabEnt	*symtabent;

	if (symtab == NULL)
	    return(NULL);

	symtabent = symtab->sym_entlist;
	while (symtabent != NULL)
	  {
	    if (&symtabent->sym_val == rp)
		return(symtabent->sym_ident);

	    symtabent = symtabent->sym_next;
	  }
	return(NULL);

}	/* SymtabKey */



Result *SymtabNew(Symtab* symtab, char* sym)
{	/* SymtabNew --- Enter new symbol table entry */

	SymtabEnt	*symtabent;
	Result		*rp;

	rp = SymtabLook(symtab, sym);
	if (rp == 0)
	  {
	  	symtabent = new SymtabEnt();

	    symtabent->sym_ident = (char *) strdup(sym);
	    symtabent->sym_next = symtab->sym_entlist;
	    symtab->sym_entlist = symtabent;
	    rp = &symtabent->sym_val;
	    rp->r_type = 0;
	  }

	return(rp);
}	/* SymtabNew */


#ifdef COMMENT

SymtabPush(SymtabScope scope)
{	/* SymtabPush --- Allocate a new symbol table */

	SymtabStack	*symtab = new SymtabStack();

	symtab->sym_scope = scope;
	symtab->sym_tab = 0;
	symtab->sym_prev = CurSymtab;
	CurSymtab = symtab;
}	/* SymtabPush */



SymtabPop()

{	/* SymtabPop --- Pop the current symbol table */

	SymtabEnt	*symtabent;

	if (CurSymtab == 0)
	  {
	  	cerr << "SymtabPop: Symbol table stack is empty!\n";
	    return;
	  }

	symtabent = CurSymtab->sym_tab;
	while (symtabent != 0)
	  {
	    delete symtabent->sym_ident;
	    delete symtabent;
	    symtabent = symtabent->sym_next;
	  }
	delete CurSymtab;

	CurSymtab = CurSymtab->sym_prev;
}	/* SymtabPop */
#endif
