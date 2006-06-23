%option c++
%option always-interactive
%option yyclass="myFlexLexer"

%p 5000
%a 5000
%o 7500


%{
// Word recognizer with a symbol table, C++ implementation
#include <fstream>
#include <setjmp.h>
#include "GenesisParser.h"
#include "GenesisParser.tab.hpp"


#    define lextoken(tok)   return(tok)
#    define lexchar(ch) return(ch)

%}

S   ("+"|"-")?
E   ((e|E){S}[0-9]+)


/*
** C comment handling taken from ANSI C Lex specification published by
** Jeff Lee and Jutta Degener <jutta@pobox.com>.
*/

%Start	LIT FUNCLIT CCOMMENT CPPCOMMENT

%{
/* I use this area to hold things that I might want back */

#ifdef COMMENT
<CPPCOMMENT>[^\n]*	Popyybgin();
"//"			Pushyybgin(CPPCOMMENT);

\\\n			continuation = 0;

<CCOMMENT>[^*\n]*	;
<CCOMMENT>"*"		;
<CCOMMENT>"\n"		;
<CCOMMENT>"*/"		Popyybgin();
"/*"/[^\n]*		Pushyybgin(CCOMMENT);
#endif
%}

%%

"/*"                    Ccomment();

"//".*			;

<LIT,FUNCLIT>\n		|
<LIT,FUNCLIT>\r		|
<LIT,FUNCLIT>"//"\r	|
<LIT,FUNCLIT>"//"\n	{
			  if (!continuation)
			      lexchar('\n');
			  else
			    {
			      continuation = 0;
			      lexchar(WHITESPACE);
			    }
			}

\n			|
\r			|
"//"\r			|
"//"\n			{
			  if (!continuation)
			      lexchar('\n');

			  continuation = 0;
			}

\\.*			continuation = 1;

\"[^\r"]*		|
\"[^\n"]*		{
			  char	ch;

			  ch = (char)yyinput();
			  if (ch == '\n')
			    {
			      yyerror
			      ("Missing closing quote in string constant");
			      /* No Return */
			    }

			  if (yytext[yyleng-1] == '\\')
			    {
			      yytext[yyleng-1] = '"';
			      yyless(yyleng-1);
			      yymore();
			    }
			  else
			    {
			      yylval.str = (char *) strdup(yytext+1);
			      lextoken(STRCONST);
			    }
			};

"$"[0-9]+		{
			  sscanf(yytext+1, " %d ", &yylval.iconst);
			  lextoken(DOLLARARG);
			}

<FUNCLIT>[^ \t\r\n{}(),"$\\]+ { 
			  yylval.str = (char *) strdup(yytext);
			  lextoken(LITERAL);
			};

<LIT>[^ \t\r\n{}"$;\\]+      { 
			  yylval.str = (char *) strdup(yytext);
			  lextoken(LITERAL);
			};

<FUNCLIT,LIT>[ \t]+	lextoken(WHITESPACE);
[ \t]+			;

include		lextoken(INCLUDE);

"*ENDSCRIPT*"	lextoken(ENDSCRIPT);

break		lextoken(BREAK);

return		lextoken(RETURN);

while		lextoken(WHILE);

if		lextoken(IF);

else		lextoken(ELSE);

elif		lextoken(ELIF);

for		lextoken(FOR);

foreach		lextoken(FOREACH);

end		lextoken(END);

function	lextoken(FUNCTION);

extern		lextoken(EXTERN);

int		lextoken(INT);

float		lextoken(FLOAT);

str		lextoken(STR);

"<"		lextoken(LT);
"<="		lextoken(LE);
">"		lextoken(GT);
">="		lextoken(GE);
"=="		lextoken(EQ);
"!="		lextoken(NE);

"||"		lextoken(OR);
"&&"		lextoken(AND);

"**"		lextoken(POW);

[_A-Za-z][_A-Za-z0-9]*	{
			  Result	*rp;

			  if (!ReturnIdents &&
			      ((rp=SymtabLook(LocalSymbols, yytext)) != NULL ||
			       (rp=SymtabLook(&GlobalSymbols, yytext)) != NULL))
			    {
			      yylval.str = (char *) rp;
			      if (rp->r_type == FUNCTION)
				  { lextoken(FUNCREF); }
			      else
				  lextoken(VARREF);
			    }
			  else
			    {
			      yylval.str = (char *) strdup(yytext);
			      lextoken(IDENT);
			    }
			};

[0-9]+			{
			  sscanf(yytext, " %d ", &yylval.iconst);
			  lextoken(INTCONST);
			}

[0-9]+{E}		|
[0-9]+\.[0-9]*{E}?	|
[0-9]*\.[0-9]+{E}?	{
			  sscanf(yytext, " %lf ", &yylval.fconst);
			  lextoken(FLOATCONST);
			}

"{"			{ Pushyybgin(0); lexchar('{'); };
"}"			{ Popyybgin(); lexchar('}'); };

.			lexchar(yytext[0]);
%%




/*
** Lexer routines to push and pop the lexer begin state
** They have to be here rather than GenesisParser.cpp because certain
** ugly #defines are
** made by flex and referred to in these functions.
*/

void myFlexLexer::Pushyybgin(int start)
{	/* Pushyybgin --- Push to another lexer begin state */

	if (yybginidx == BGINSTKSIZE)
	    yyerror("Pushyybgin: lexer begin state stack overflow");
	    /* No Return */

	yybginstk[yybginidx++] = YY_START;
	BEGIN start;

}	/* Pushyybgin */


void myFlexLexer::Popyybgin()
{	/* Popyybgin --- Pop to previous lexer begin state */

	if (yybginidx == 0)
	    yyerror("Popyybgin: lexer begin state stack underflow");
	    /* No Return */

	BEGIN yybginstk[--yybginidx];

}	/* Popyybgin */


int myFlexLexer::nextchar(int flush)
{
	if (flush)
	  {
	    yybginidx = 0;
	    BEGIN 0;
	    yyrestart(yyin); /* flushes flex input buffer */
	  }

	return 0;
}
