
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 1 "GenesisParser.ypp"

// SLI parser
#include <iosfwd>
#include <string>
#include <vector>
#include <map>


// Upi Bhalla, 24 May 2004:
// I did a little checking up about portability of the setjmp construct
// in the parser code. It looks like it ought to work even in VC++,
// but I'll have to actually compile it to be sure. The consensus
// seems to be that this language construct should probably be
// avoided. Since I want to keep the changes to the SLI parser code
// to a minimum, I'll leave it in.
#include <setjmp.h>

// #include "../basecode/Shell.h"
#include <FlexLexer.h>
#include "header.h"
#include "script.h"
#include "GenesisParser.h"
#include "GenesisParser.tab.h"
#include "func_externs.h"

using namespace std;

/*
** Parser routines which return something other than int.
*/

extern char *TokenStr(int token);


/* Line 189 of yacc.c  */
#line 72 "GenesisParser.ypp"

#include "GenesisParser.yy.cpp"


/* Line 189 of yacc.c  */
#line 114 "GenesisParser.tab.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     AND = 258,
     OR = 259,
     NE = 260,
     EQ = 261,
     GE = 262,
     GT = 263,
     LE = 264,
     LT = 265,
     POW = 266,
     UMINUS = 267,
     WHILE = 268,
     IF = 269,
     ELSE = 270,
     ELIF = 271,
     FOR = 272,
     FOREACH = 273,
     END = 274,
     INCLUDE = 275,
     ENDSCRIPT = 276,
     BREAK = 277,
     QUIT = 278,
     INT = 279,
     FLOAT = 280,
     STR = 281,
     RETURN = 282,
     WHITESPACE = 283,
     FUNCTION = 284,
     INTCONST = 285,
     DOLLARARG = 286,
     FLOATCONST = 287,
     STRCONST = 288,
     LITERAL = 289,
     IDENT = 290,
     VARREF = 291,
     FUNCREF = 292,
     EXTERN = 293,
     SL = 294,
     COMMAND = 295,
     EXPRCALL = 296,
     ARGUMENT = 297,
     ARGLIST = 298,
     LOCREF = 299,
     ICAST = 300,
     FCAST = 301,
     SCAST = 302
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 202 "GenesisParser.tab.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   378

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  68
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  60
/* YYNRULES -- Number of rules.  */
#define YYNRULES  132
/* YYNRULES -- Number of states.  */
#define YYNSTATES  220

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   302

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      59,     2,     2,    60,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    11,     2,     2,     2,    20,    14,     2,
      62,    63,    18,    12,    67,    13,     2,    19,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    61,
       2,    64,     2,     2,    17,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,    16,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    65,    15,    66,    21,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     7,    12,    14,    16,    18,
      20,    22,    24,    26,    28,    30,    32,    34,    36,    38,
      40,    42,    44,    46,    48,    50,    52,    54,    55,    63,
      64,    76,    77,    78,    90,    91,   100,   101,   104,   105,
     113,   117,   118,   122,   128,   132,   133,   135,   136,   140,
     141,   145,   146,   151,   153,   154,   160,   161,   165,   167,
     171,   172,   174,   176,   179,   180,   182,   184,   187,   189,
     191,   193,   194,   195,   201,   203,   205,   207,   210,   213,
     216,   219,   220,   221,   228,   229,   233,   235,   239,   241,
     243,   245,   248,   250,   251,   256,   257,   261,   262,   265,
     267,   270,   272,   273,   277,   281,   285,   288,   292,   296,
     300,   304,   308,   312,   316,   319,   323,   327,   330,   334,
     338,   342,   346,   350,   354,   356,   358,   360,   362,   364,
     368,   372,   376
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      69,     0,    -1,    70,    -1,    -1,    -1,    70,    71,    73,
      72,    -1,    59,    -1,    60,    -1,    61,    -1,    76,    -1,
      78,    -1,    80,    -1,    83,    -1,    87,    -1,    90,    -1,
      74,    -1,    75,    -1,    95,    -1,    98,    -1,   112,    -1,
     110,    -1,   118,    -1,   124,    -1,   125,    -1,   126,    -1,
      32,    -1,    34,    -1,    -1,    24,    62,   127,    63,    77,
      70,    30,    -1,    -1,    28,    62,    87,    61,   127,    61,
      87,    63,    79,    70,    30,    -1,    -1,    -1,    29,    47,
      62,    81,   102,   101,   102,    82,    63,    70,    30,    -1,
      -1,    25,    62,   127,    63,    84,    70,    85,    30,    -1,
      -1,    26,    70,    -1,    -1,    27,    62,   127,    63,    86,
      70,    85,    -1,    47,    64,   127,    -1,    -1,    31,    89,
     104,    -1,    88,    45,    39,   101,   104,    -1,    88,    45,
     104,    -1,    -1,   105,    -1,    -1,    46,    93,    91,    -1,
      -1,    47,    94,   105,    -1,    -1,    92,    96,   100,   104,
      -1,    48,    -1,    -1,    97,    91,    99,   100,   104,    -1,
      -1,   100,   103,   105,    -1,   105,    -1,   101,   103,   105,
      -1,    -1,   103,    -1,    39,    -1,   103,    39,    -1,    -1,
      39,    -1,   106,    -1,   105,   106,    -1,    45,    -1,    44,
      -1,    42,    -1,    -1,    -1,    65,   107,   109,   108,    66,
      -1,    98,    -1,    95,    -1,   127,    -1,    49,    46,    -1,
      49,    48,    -1,    40,    46,    -1,    40,    48,    -1,    -1,
      -1,   111,   113,   115,   114,    70,    30,    -1,    -1,    62,
     116,    63,    -1,    46,    -1,   116,    67,    46,    -1,    35,
      -1,    36,    -1,    37,    -1,   117,   119,    -1,   121,    -1,
      -1,   119,    67,   120,   121,    -1,    -1,    46,   122,   123,
      -1,    -1,    64,   127,    -1,    33,    -1,    38,   127,    -1,
      38,    -1,    -1,   127,    15,   127,    -1,   127,    14,   127,
      -1,   127,    16,   127,    -1,    21,   127,    -1,   127,    17,
     127,    -1,   127,    12,   127,    -1,   127,    13,   127,    -1,
     127,    18,   127,    -1,   127,    19,   127,    -1,   127,    20,
     127,    -1,   127,    22,   127,    -1,    13,   127,    -1,   127,
       4,   127,    -1,   127,     3,   127,    -1,    11,   127,    -1,
     127,    10,   127,    -1,   127,     9,   127,    -1,   127,     8,
     127,    -1,   127,     7,   127,    -1,   127,     6,   127,    -1,
     127,     5,   127,    -1,    47,    -1,    43,    -1,    41,    -1,
      44,    -1,    42,    -1,    65,   127,    66,    -1,    65,    95,
      66,    -1,    65,    98,    66,    -1,    62,   127,    63,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    80,    80,    84,    88,    87,   123,   124,   125,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   146,   166,   177,   176,   194,
     193,   216,   222,   215,   240,   239,   257,   260,   263,   262,
     279,   295,   294,   304,   347,   388,   391,   405,   404,   419,
     418,   450,   449,   493,   501,   500,   522,   525,   531,   535,
     541,   542,   545,   546,   549,   550,   553,   559,   567,   574,
     581,   589,   593,   588,   602,   606,   610,   619,   638,   644,
     675,   713,   717,   712,   733,   734,   738,   749,   762,   768,
     774,   782,   788,   790,   789,   803,   802,   814,   815,   819,
     831,   837,   846,   849,   851,   853,   855,   858,   861,   863,
     865,   867,   869,   871,   873,   876,   878,   880,   883,   885,
     887,   889,   891,   893,   896,   916,   923,   930,   938,   946,
     949,   952,   955
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "AND", "OR", "NE", "EQ", "GE", "GT",
  "LE", "LT", "'!'", "'+'", "'-'", "'&'", "'|'", "'^'", "'@'", "'*'",
  "'/'", "'%'", "'~'", "POW", "UMINUS", "WHILE", "IF", "ELSE", "ELIF",
  "FOR", "FOREACH", "END", "INCLUDE", "ENDSCRIPT", "BREAK", "QUIT", "INT",
  "FLOAT", "STR", "RETURN", "WHITESPACE", "FUNCTION", "INTCONST",
  "DOLLARARG", "FLOATCONST", "STRCONST", "LITERAL", "IDENT", "VARREF",
  "FUNCREF", "EXTERN", "SL", "COMMAND", "EXPRCALL", "ARGUMENT", "ARGLIST",
  "LOCREF", "ICAST", "FCAST", "SCAST", "'\\n'", "'\\r'", "';'", "'('",
  "')'", "'='", "'{'", "'}'", "','", "$accept", "script", "statement_list",
  "@1", "stmnt_terminator", "statement", "endscript_marker", "quit_stmnt",
  "while_stmnt", "@2", "for_stmnt", "@3", "foreach_stmnt", "$@4", "$@5",
  "if_stmnt", "@6", "else_clause", "@7", "assign_stmnt", "include_hdr",
  "$@8", "include_stmnt", "opt_node", "cmd_name", "$@9", "$@10",
  "cmd_stmnt", "$@11", "funcref", "func_call", "$@12", "opt_arg_list",
  "arg_list", "optwslist", "wslist", "ws", "arg_component_list",
  "arg_component", "$@13", "$@14", "ac_func_cmd_expr", "ext_func",
  "func_hdr", "func_def", "$@15", "$@16", "func_args", "func_arg_list",
  "decl_type", "decl_stmnt", "decl_list", "$@17", "decl_ident", "$@18",
  "init", "break_stmnt", "return_stmnt", "null_stmnt", "expr", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,    33,    43,    45,    38,   124,    94,    64,    42,    47,
      37,   126,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   295,   296,   297,   298,   299,   300,   301,   302,    10,
      13,    59,    40,    41,    61,   123,   125,    44
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    68,    69,    70,    71,    70,    72,    72,    72,    73,
      73,    73,    73,    73,    73,    73,    73,    73,    73,    73,
      73,    73,    73,    73,    73,    74,    75,    77,    76,    79,
      78,    81,    82,    80,    84,    83,    85,    85,    86,    85,
      87,    89,    88,    90,    90,    91,    91,    93,    92,    94,
      92,    96,    95,    97,    99,    98,   100,   100,   101,   101,
     102,   102,   103,   103,   104,   104,   105,   105,   106,   106,
     106,   107,   108,   106,   109,   109,   109,   110,   110,   111,
     111,   113,   114,   112,   115,   115,   116,   116,   117,   117,
     117,   118,   119,   120,   119,   122,   121,   123,   123,   124,
     125,   125,   126,   127,   127,   127,   127,   127,   127,   127,
     127,   127,   127,   127,   127,   127,   127,   127,   127,   127,
     127,   127,   127,   127,   127,   127,   127,   127,   127,   127,
     127,   127,   127
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     0,     4,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     7,     0,
      11,     0,     0,    11,     0,     8,     0,     2,     0,     7,
       3,     0,     3,     5,     3,     0,     1,     0,     3,     0,
       3,     0,     4,     1,     0,     5,     0,     3,     1,     3,
       0,     1,     1,     2,     0,     1,     1,     2,     1,     1,
       1,     0,     0,     5,     1,     1,     1,     2,     2,     2,
       2,     0,     0,     6,     0,     3,     1,     3,     1,     1,
       1,     2,     1,     0,     4,     0,     3,     0,     2,     1,
       2,     1,     0,     3,     3,     3,     2,     3,     3,     3,
       3,     3,     3,     3,     2,     3,     3,     2,     3,     3,
       3,     3,     3,     3,     1,     1,     1,     1,     1,     3,
       3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     4,     1,   102,     0,     0,     0,     0,    41,
      25,    99,    26,    88,    89,    90,   101,     0,    47,    49,
      53,     0,     0,    15,    16,     9,    10,    11,    12,    13,
       0,    14,    51,    17,    45,    18,    20,    81,    19,     0,
      21,    22,    23,    24,     0,     0,     0,     0,    64,     0,
       0,     0,   126,   128,   125,   127,   124,     0,     0,   100,
      79,    80,    45,     0,     0,    77,    78,     6,     7,     8,
       5,    64,    56,    70,    69,    68,    71,    54,    46,    66,
      84,    95,    91,    92,     0,     0,     0,     0,    31,    65,
      42,   117,   114,   106,     0,   124,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    40,    50,
      65,    44,    64,     0,    56,    67,     0,    82,    97,    93,
      27,    34,     0,    60,   132,   130,   131,   129,   116,   115,
     123,   122,   121,   120,   119,   118,   108,   109,   104,   103,
     105,   107,   110,   111,   112,   113,    64,    58,    62,     0,
      52,    75,    74,    72,    76,    64,    86,     0,     3,     0,
      96,     0,     3,     3,     0,    62,     0,    61,     0,    43,
      63,    57,     0,    55,    85,     0,     4,    98,    94,     4,
       4,     0,    60,    59,    73,    87,    83,    28,     3,     0,
       0,     0,    32,    61,     4,     0,    35,    29,     0,     0,
       3,     3,    38,     4,     4,     3,    30,    33,     4,    39
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,     4,    70,    22,    23,    24,    25,   172,
      26,   210,    27,   133,   208,    28,   173,   200,   215,    29,
      30,    48,    31,    77,    32,    62,    64,    33,    72,    34,
      35,   124,   122,   156,   176,   159,    90,    78,    79,   123,
     182,   163,    36,    37,    38,    80,   168,   127,   167,    39,
      40,    82,   171,    83,   128,   170,    41,    42,    43,    59
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -104
static const yytype_int16 yypact[] =
{
    -104,    23,    28,  -104,   242,   -33,    16,    34,   -13,  -104,
    -104,  -104,  -104,  -104,  -104,  -104,    93,   -30,  -104,    21,
    -104,   -22,   -28,  -104,  -104,  -104,  -104,  -104,  -104,  -104,
      45,  -104,  -104,  -104,    42,  -104,  -104,  -104,  -104,    51,
    -104,  -104,  -104,  -104,    93,    93,    52,    40,    61,    93,
      93,    93,  -104,  -104,  -104,  -104,  -104,    93,    80,   309,
    -104,  -104,    42,    93,    42,  -104,  -104,  -104,  -104,  -104,
    -104,    69,  -104,  -104,  -104,  -104,  -104,  -104,    42,  -104,
      47,  -104,    44,  -104,   159,   179,    21,    55,  -104,  -104,
    -104,   356,  -104,    91,   199,    50,    54,    64,    32,    93,
      93,    93,    93,    93,    93,    93,    93,    93,    93,    93,
      93,    93,    93,    93,    93,    93,    93,  -104,   309,    42,
      42,  -104,    79,    80,  -104,  -104,    86,  -104,    74,  -104,
    -104,  -104,    93,    94,  -104,  -104,  -104,  -104,   327,   327,
     345,   345,   345,   345,   345,   345,   131,   131,   131,   131,
     131,   131,    91,    91,    91,  -104,    79,    42,   -39,    38,
    -104,  -104,  -104,  -104,   309,    79,  -104,   -59,  -104,    93,
    -104,    51,  -104,  -104,   241,  -104,    42,   100,    38,  -104,
    -104,    42,    75,  -104,  -104,    97,   114,   309,  -104,   116,
     -15,    52,    94,    42,  -104,  -104,  -104,  -104,  -104,    85,
     122,   107,  -104,    38,   124,    93,  -104,  -104,   117,   221,
    -104,  -104,  -104,   127,   129,  -104,  -104,  -104,   -15,  -104
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -104,  -104,    96,  -104,  -104,  -104,  -104,  -104,  -104,  -104,
    -104,  -104,  -104,  -104,  -104,  -104,  -104,   -58,  -104,   -43,
    -104,  -104,  -104,   128,  -104,  -104,  -104,   -49,  -104,  -104,
     -48,  -104,    76,    56,    18,  -103,   -46,   -47,   -76,  -104,
    -104,  -104,  -104,  -104,  -104,  -104,  -104,  -104,  -104,  -104,
    -104,  -104,  -104,    49,  -104,  -104,  -104,  -104,  -104,   -44
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -66
static const yytype_int16 yytable[] =
{
      84,    85,   125,    87,   184,    91,    92,    93,   185,    96,
      97,   198,   199,    94,    98,   -36,    60,   119,    61,   118,
     -65,   -65,   -65,     3,    65,   121,    66,   -65,    -2,    44,
     177,    67,    68,    69,    47,    99,   100,   101,   102,   103,
     104,   105,   106,   125,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   178,   116,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   157,   161,   162,   160,   180,    45,   164,
      73,   125,    74,    75,    73,    63,    74,    75,   174,   203,
      71,    49,   -49,    50,   -49,   -49,    46,    81,   137,    86,
      89,    51,    88,    76,    49,   125,    50,    76,   120,   126,
     179,   129,   181,   116,    51,   -49,   132,   125,   158,   183,
     135,    52,    53,    54,    55,   187,    18,    95,    20,   157,
     136,   193,   166,   175,    52,    53,    54,    55,   169,   180,
      56,   194,    57,   195,   196,    58,   197,   205,   201,   113,
     114,   115,   206,   116,   -37,    57,   193,   216,    58,   217,
     219,   209,    99,   100,   101,   102,   103,   104,   105,   106,
     207,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     211,   116,    99,   100,   101,   102,   103,   104,   105,   106,
     117,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     165,   116,    99,   100,   101,   102,   103,   104,   105,   106,
     202,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     188,   116,   130,     0,    99,   100,   101,   102,   103,   104,
     105,   106,   192,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   131,   116,    99,   100,   101,   102,   103,   104,
     105,   106,     0,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   134,   116,   186,     0,     5,     6,   189,   190,
       7,     8,     0,     9,    10,    11,    12,    13,    14,    15,
      16,     0,    17,     0,   212,     0,     0,     0,    18,    19,
      20,    21,     0,     0,   204,     0,     0,     0,     0,     0,
       0,     0,   191,     0,     0,     0,   213,   214,     0,     0,
       0,   218,    99,   100,   101,   102,   103,   104,   105,   106,
       0,   107,   108,   109,   110,   111,   112,   113,   114,   115,
       0,   116,   101,   102,   103,   104,   105,   106,     0,   107,
     108,   109,   110,   111,   112,   113,   114,   115,     0,   116,
     -66,   -66,   -66,   -66,   -66,   -66,     0,   107,   108,   109,
     110,   111,   112,   113,   114,   115,     0,   116,   107,   108,
     109,   110,   111,   112,   113,   114,   115,     0,   116
};

static const yytype_int16 yycheck[] =
{
      44,    45,    78,    46,    63,    49,    50,    51,    67,    58,
      58,    26,    27,    57,    58,    30,    46,    64,    48,    63,
      59,    60,    61,     0,    46,    71,    48,    66,     0,    62,
     133,    59,    60,    61,    47,     3,     4,     5,     6,     7,
       8,     9,    10,   119,    12,    13,    14,    15,    16,    17,
      18,    19,    20,   156,    22,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   120,   123,   123,   122,    39,    62,   123,
      42,   157,    44,    45,    42,    64,    44,    45,   132,   192,
      45,    11,    42,    13,    44,    45,    62,    46,    66,    47,
      39,    21,    62,    65,    11,   181,    13,    65,    39,    62,
     156,    67,   159,    22,    21,    65,    61,   193,    39,   165,
      66,    41,    42,    43,    44,   169,    46,    47,    48,   176,
      66,   178,    46,    39,    41,    42,    43,    44,    64,    39,
      47,    66,    62,    46,    30,    65,    30,    62,   191,    18,
      19,    20,    30,    22,    30,    62,   203,    30,    65,    30,
     218,   205,     3,     4,     5,     6,     7,     8,     9,    10,
      63,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      63,    22,     3,     4,     5,     6,     7,     8,     9,    10,
      62,    12,    13,    14,    15,    16,    17,    18,    19,    20,
     124,    22,     3,     4,     5,     6,     7,     8,     9,    10,
     192,    12,    13,    14,    15,    16,    17,    18,    19,    20,
     171,    22,    63,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,   176,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    63,    22,     3,     4,     5,     6,     7,     8,
       9,    10,    -1,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    63,    22,   168,    -1,    24,    25,   172,   173,
      28,    29,    -1,    31,    32,    33,    34,    35,    36,    37,
      38,    -1,    40,    -1,    63,    -1,    -1,    -1,    46,    47,
      48,    49,    -1,    -1,   198,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    61,    -1,    -1,    -1,   210,   211,    -1,    -1,
      -1,   215,     3,     4,     5,     6,     7,     8,     9,    10,
      -1,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      -1,    22,     5,     6,     7,     8,     9,    10,    -1,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    -1,    22,
       5,     6,     7,     8,     9,    10,    -1,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    -1,    22,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    -1,    22
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    69,    70,     0,    71,    24,    25,    28,    29,    31,
      32,    33,    34,    35,    36,    37,    38,    40,    46,    47,
      48,    49,    73,    74,    75,    76,    78,    80,    83,    87,
      88,    90,    92,    95,    97,    98,   110,   111,   112,   117,
     118,   124,   125,   126,    62,    62,    62,    47,    89,    11,
      13,    21,    41,    42,    43,    44,    47,    62,    65,   127,
      46,    48,    93,    64,    94,    46,    48,    59,    60,    61,
      72,    45,    96,    42,    44,    45,    65,    91,   105,   106,
     113,    46,   119,   121,   127,   127,    47,    87,    62,    39,
     104,   127,   127,   127,   127,    47,    95,    98,   127,     3,
       4,     5,     6,     7,     8,     9,    10,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    22,    91,   127,   105,
      39,   104,   100,   107,    99,   106,    62,   115,   122,    67,
      63,    63,    61,    81,    63,    66,    66,    66,   127,   127,
     127,   127,   127,   127,   127,   127,   127,   127,   127,   127,
     127,   127,   127,   127,   127,   127,   101,   105,    39,   103,
     104,    95,    98,   109,   127,   100,    46,   116,   114,    64,
     123,   120,    77,    84,   127,    39,   102,   103,   103,   104,
      39,   105,   108,   104,    63,    67,    70,   127,   121,    70,
      70,    61,   101,   105,    66,    46,    30,    30,    26,    27,
      85,    87,   102,   103,    70,    62,    30,    63,    82,   127,
      79,    63,    63,    70,    70,    86,    30,    30,    70,    85
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
myFlexLexer::yyparse (void *YYPARSE_PARAM)
#else
int
myFlexLexer::yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
myFlexLexer::yyparse (void)
#else
int
myFlexLexer::yyparse ()

#endif
#endif
{


    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 3:

/* Line 1455 of yacc.c  */
#line 84 "GenesisParser.ypp"
    { 
		    (yyval.pn) = NULL;
 		  ;}
    break;

  case 4:

/* Line 1455 of yacc.c  */
#line 88 "GenesisParser.ypp"
    {
		    (yyval.str) = (char *) MakeScriptInfo();
		    SetLine((ScriptInfo *) (yyval.str));
		  ;}
    break;

  case 5:

/* Line 1455 of yacc.c  */
#line 93 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    if (InFunctionDefinition || Compiling)
			if ((yyvsp[(3) - (4)].pn) != NULL)
			  {
			    v.r_str = (yyvsp[(2) - (4)].str);
			    (yyval.pn) = PTNew(SL, v, (yyvsp[(1) - (4)].pn), (yyvsp[(3) - (4)].pn));
			  }
			else
			  {
			    FreeScriptInfo((ScriptInfo *)(yyvsp[(2) - (4)].str));
			    (yyval.pn) = (yyvsp[(1) - (4)].pn);
			  }
		    else
		      {
		        /* execute statement */
		        if (setjmp(BreakBuf) == 0) {
			    if (setjmp(ReturnBuf) == 0)
				PTCall((yyvsp[(3) - (4)].pn));
			    else
				EndScript();
		        }
			FreeScriptInfo((ScriptInfo *)(yyvsp[(2) - (4)].str));
			FreePTValues((yyvsp[(3) - (4)].pn));
			PTFree((yyvsp[(3) - (4)].pn));
		      }
		  ;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 147 "GenesisParser.ypp"
    {
		    /*
		    ** When the end of a script is encountered, the simulator
		    ** sgets routine returns NULL.  The nextchar routine in the
		    ** lexer returns a special character '\200' which is lexed
		    ** as ENDSCRIPT.  We need this when we include a script
		    ** in a function or control structure so that the script
		    ** local variable storage is allocated and deallocated.
		    */

		    if (Compiling || InFunctionDefinition)
		      {
			(yyval.pn) = PTNew(ENDSCRIPT, RV, NULL, NULL);
		      }
		    else
			(yyval.pn) = NULL;
		  ;}
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 167 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    char * argv[] = {"quit"};
		    (yyval.pn) = PTNew(QUIT, v, NULL, NULL);
		    doQuit(true);
		    ExecuteCommand(1, argv);
		    YYACCEPT;	
	          ;}
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 177 "GenesisParser.ypp"
    {
		    Compiling++;
		    BreakAllowed++;
		    (yyval.str) = (char *) MakeScriptInfo();
		  ;}
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 183 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_str = (yyvsp[(5) - (7)].str);
		    (yyval.pn) = PTNew(WHILE, v, (yyvsp[(3) - (7)].pn), (yyvsp[(6) - (7)].pn));
		    Compiling--;
		    BreakAllowed--;
		  ;}
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 194 "GenesisParser.ypp"
    {
		      Compiling++;
		      BreakAllowed++;
		      (yyval.str) = (char *) MakeScriptInfo();
		    ;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 201 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    ParseNode	*forbody, *whilepart;

		    v.r_str = (char *) MakeScriptInfo();
		    forbody = PTNew(SL, v, (yyvsp[(10) - (11)].pn), (yyvsp[(7) - (11)].pn));
		    v.r_str = (yyvsp[(9) - (11)].str);
		    whilepart = PTNew(FOR, v, (yyvsp[(5) - (11)].pn), forbody);
		    (yyval.pn) = PTNew(SL, v, (yyvsp[(3) - (11)].pn), whilepart);
		    Compiling--;
		    BreakAllowed--;
		  ;}
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 216 "GenesisParser.ypp"
    {
			BEGIN FUNCLIT;
			Compiling++;
			BreakAllowed++;
		    ;}
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 222 "GenesisParser.ypp"
    {
			BEGIN 0;
		    ;}
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 226 "GenesisParser.ypp"
    {
		    Result	*rp;
		    ResultValue	v;
		    // char        buf[100];

		    rp = (Result *) (yyvsp[(2) - (11)].str);
		    v.r_str = (char *) rp;
		    (yyval.pn) = PTNew(FOREACH, v, (yyvsp[(6) - (11)].pn), (yyvsp[(10) - (11)].pn));
		    Compiling--;
		    BreakAllowed--;
		  ;}
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 240 "GenesisParser.ypp"
    {
		    Compiling++;
		    (yyval.str) = (char *) MakeScriptInfo();
		  ;}
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 245 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    ParseNode	*stmntlists;

		    stmntlists = PTNew(0, v, (yyvsp[(6) - (8)].pn), (yyvsp[(7) - (8)].pn));
		    v.r_str = (yyvsp[(5) - (8)].str);
		    (yyval.pn) = PTNew(IF, v, (yyvsp[(3) - (8)].pn), stmntlists);
		    Compiling--;
		  ;}
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 257 "GenesisParser.ypp"
    {
 		    (yyval.pn) = NULL;
 		  ;}
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 261 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (2)].pn); ;}
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 263 "GenesisParser.ypp"
    {
		    Compiling++;
		    (yyval.str) = (char *) MakeScriptInfo();
		  ;}
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 268 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    ParseNode	*stmntlists;

		    stmntlists = PTNew(0, v, (yyvsp[(6) - (7)].pn), (yyvsp[(7) - (7)].pn));
		    v.r_str = (yyvsp[(5) - (7)].str);
		    (yyval.pn) = PTNew(IF, v, (yyvsp[(3) - (7)].pn), stmntlists);
		    Compiling--;
		  ;}
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 280 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    Result	*rp;
		    // char        buf[100];

		    (yyval.pn) = NULL;
		    rp = (Result *) (yyvsp[(1) - (3)].str);
			  {
			    v.r_str = (char *) rp;
		            (yyval.pn) = PTNew('=', v, (yyvsp[(3) - (3)].pn), NULL);
			  }
		  ;}
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 295 "GenesisParser.ypp"
    {
		    Pushyybgin(LIT);
		  ;}
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 299 "GenesisParser.ypp"
    {
		    (yyval.str) = NULL;
		  ;}
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 305 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    // Result	*rp;
		    int		argc;
		    char	*argv[100];
		    char	argbuf[1000];
			bool	doneFree = 0;
		    // jmp_buf	save;
		    // Result	r;

		    Popyybgin();
		    sprintf(argbuf, "%s", (yyvsp[(2) - (5)].str));
		    argc = 1;
		    argv[0] = argbuf;
		    do_cmd_args((yyvsp[(4) - (5)].pn), &argc, argv);
		    argv[argc] = NULL;

		    if (!IncludeScript(argc, argv))
		      {
			sprintf(argbuf, "Script '%s' not found", (yyvsp[(2) - (5)].str));
			FreePTValues((yyvsp[(4) - (5)].pn));
			PTFree((yyvsp[(4) - (5)].pn));
			free((yyvsp[(2) - (5)].str));
			doneFree = 1;
			yyerror(argbuf);
		      }

		    if (Compiling || InFunctionDefinition)
		      {
			v.r_str = (yyvsp[(2) - (5)].str);
			(yyval.pn) = PTNew(INCLUDE, v, (yyvsp[(4) - (5)].pn), NULL);
		      }
		    else
		      {
			  if ( doneFree == 0 ) {
				FreePTValues((yyvsp[(4) - (5)].pn));
				PTFree((yyvsp[(4) - (5)].pn));
				free((yyvsp[(2) - (5)].str));
				}
			(yyval.pn) = NULL;
		      }
		  ;}
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 348 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    // Result	*rp;
		    int		argc;
		    char	*argv[100];
		    char	argbuf[1000];
			bool	doneFree = 0;
		    // jmp_buf	save;
		    // Result	r;

		    Popyybgin();
		    sprintf(argbuf, "%s", (yyvsp[(2) - (3)].str));
		    argc = 1;
		    argv[0] = argbuf;
		    argv[argc] = NULL;

		    if (!IncludeScript(argc, argv))
		      {
			sprintf(argbuf, "Script '%s' not found", (yyvsp[(2) - (3)].str));
			free((yyvsp[(2) - (3)].str));
			doneFree = 1;
			yyerror(argbuf);
		      }

		    if (Compiling || InFunctionDefinition)
		      {
			v.r_str = (yyvsp[(2) - (3)].str);
			(yyval.pn) = PTNew(INCLUDE, v, NULL, NULL);
		      }
		    else
		      {
			  if ( doneFree == 0 ) {
				free((yyvsp[(2) - (3)].str));
			}
			(yyval.pn) = NULL;
		      }
		  ;}
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 388 "GenesisParser.ypp"
    {
		    (yyval.pn) = (ParseNode*) NULL;
		  ;}
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 392 "GenesisParser.ypp"
    {
		    (yyval.pn) = (yyvsp[(1) - (1)].pn);
		  ;}
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 405 "GenesisParser.ypp"
    {
		    Pushyybgin(LIT);
		  ;}
    break;

  case 48:

/* Line 1455 of yacc.c  */
#line 409 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_str = (yyvsp[(1) - (3)].str);
		    if ((yyvsp[(3) - (3)].pn) == NULL)
			(yyval.pn) = PTNew(COMMAND, v, NULL, NULL);
		    else
			(yyval.pn) = PTNew(FUNCTION, v, NULL, (yyvsp[(3) - (3)].pn));
		  ;}
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 419 "GenesisParser.ypp"
    {
		    Pushyybgin(LIT);
		  ;}
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 423 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    char*	varname;

		    varname = NULL;
		    if (LocalSymbols != NULL)
			varname = SymtabKey(LocalSymbols, (Result *)(yyvsp[(1) - (3)].str));
		    if (varname == NULL)
			varname = SymtabKey(&GlobalSymbols, (Result *)(yyvsp[(1) - (3)].str));
		    v.r_str = (char*) strsave(varname);

		    (yyval.pn) = PTNew(FUNCTION, v, NULL, (yyvsp[(3) - (3)].pn));
		  ;}
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 450 "GenesisParser.ypp"
    {
		    BEGIN LIT;
		  ;}
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 454 "GenesisParser.ypp"
    {
		    // ResultValue	v;
		    // Result	*rp;
		    int		argc;
		    char	*argv[100];
		    char	argbuf[1000];
		    // jmp_buf	save;
		    // Result	r;

		    (yyval.pn) = (yyvsp[(1) - (4)].pn);
		    (yyvsp[(1) - (4)].pn)->pn_left = (yyvsp[(3) - (4)].pn);
		    Popyybgin();
		    if ((yyvsp[(1) - (4)].pn)->pn_val.r_type != EXPRCALL && (yyvsp[(1) - (4)].pn)->pn_right == NULL &&
				!IsCommand((yyvsp[(1) - (4)].pn)->pn_val.r.r_str))
		      {
			if (IsInclude((yyvsp[(1) - (4)].pn)->pn_val.r.r_str))
			  {
			    sprintf(argbuf, "%s", (yyvsp[(1) - (4)].pn)->pn_val.r.r_str);
			    argc = 1;
			    argv[0] = argbuf;
			    do_cmd_args((yyvsp[(3) - (4)].pn), &argc, argv);
			    argv[argc] = NULL;
			    IncludeScript(argc, argv);

			    if (Compiling || InFunctionDefinition)
			      {
				(yyvsp[(1) - (4)].pn)->pn_val.r_type = INCLUDE;
			      }
			    else
			      {
				FreePTValues((yyvsp[(1) - (4)].pn));
				PTFree((yyvsp[(1) - (4)].pn));
				(yyval.pn) = NULL;
			      }
			  }
		      }
		  ;}
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 494 "GenesisParser.ypp"
    {
		    Pushyybgin(LIT);
		    (yyval.str) = (yyvsp[(1) - (1)].str);
		  ;}
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 501 "GenesisParser.ypp"
    {
		    BEGIN LIT;
		  ;}
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 505 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    Result	*rp;

		    Popyybgin();
		    rp = (Result *) (yyvsp[(1) - (5)].str);
		    if ((yyvsp[(2) - (5)].pn) == NULL)
			(yyval.pn) = PTNew(FUNCTION, rp->r, (yyvsp[(4) - (5)].pn), NULL);
		    else
		      {
			v.r_str = (char*) strsave(SymtabKey(&GlobalSymbols, rp));
			(yyval.pn) = PTNew(FUNCTION, v, (yyvsp[(4) - (5)].pn), (yyvsp[(2) - (5)].pn));
		      }
		  ;}
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 522 "GenesisParser.ypp"
    {
		    (yyval.pn) = NULL;
		  ;}
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 526 "GenesisParser.ypp"
    {
		    (yyval.pn) = PTNew(ARGLIST, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn));
		  ;}
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 532 "GenesisParser.ypp"
    {
		    (yyval.pn) = PTNew(ARGLIST, RV, NULL, (yyvsp[(1) - (1)].pn));
		  ;}
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 536 "GenesisParser.ypp"
    {
		    (yyval.pn) = PTNew(ARGLIST, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn));
		  ;}
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 554 "GenesisParser.ypp"
    {
			    ResultValue	v;

			    (yyval.pn) = PTNew(ARGUMENT, v, NULL, (yyvsp[(1) - (1)].pn));
			  ;}
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 560 "GenesisParser.ypp"
    {
			    ResultValue	v;

			    (yyval.pn) = PTNew(ARGUMENT, v, (yyvsp[(1) - (2)].pn), (yyvsp[(2) - (2)].pn));
			  ;}
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 568 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_str = (yyvsp[(1) - (1)].str);
		    (yyval.pn) = PTNew(LITERAL, v, NULL, NULL);
		  ;}
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 575 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_str = (yyvsp[(1) - (1)].str);
		    (yyval.pn) = PTNew(LITERAL, v, NULL, NULL);
		  ;}
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 582 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_int = (yyvsp[(1) - (1)].iconst);
		    (yyval.pn) = PTNew(DOLLARARG, v, NULL, NULL);
		  ;}
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 589 "GenesisParser.ypp"
    {
		    Pushyybgin(0);
		  ;}
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 593 "GenesisParser.ypp"
    {
		    Popyybgin();
		  ;}
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 597 "GenesisParser.ypp"
    {
		    (yyval.pn) = (yyvsp[(3) - (5)].pn);
		  ;}
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 603 "GenesisParser.ypp"
    {
		    (yyval.pn) = (yyvsp[(1) - (1)].pn);
		  ;}
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 607 "GenesisParser.ypp"
    {
		    (yyval.pn) = (yyvsp[(1) - (1)].pn);
		  ;}
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 611 "GenesisParser.ypp"
    {
		    if ((yyvsp[(1) - (1)].pn)->pn_val.r_type == STRCONST)
			(yyvsp[(1) - (1)].pn)->pn_val.r_type = LITERAL;

		    (yyval.pn) = (yyvsp[(1) - (1)].pn);
		  ;}
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 620 "GenesisParser.ypp"
    {
		    ParseNode	*funcpn;
		    ResultValue	v;
		    Result	*rp;

		    rp = SymtabNew(&GlobalSymbols, (yyvsp[(2) - (2)].str));
		    if (rp->r_type != 0 && rp->r_type != FUNCTION)
			fprintf(stderr, "WARNING: function name '%s' is redefining a variable!\n", (yyvsp[(2) - (2)].str));

		    rp->r_type = FUNCTION;

		    v.r_str = (char *) NULL;
		    funcpn = PTNew(SL, v, NULL, NULL);
		    rp->r.r_str = (char *) funcpn;

		    free((yyvsp[(2) - (2)].str));
		    (yyval.pn) = NULL;
		  ;}
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 639 "GenesisParser.ypp"
    {
		    (yyval.pn) = NULL;
		  ;}
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 645 "GenesisParser.ypp"
    {
		    ParseNode	*funcpn;
		    ResultValue	v;
		    Result	*rp;
		    // char	*script;

		    if (InFunctionDefinition)
		      {
			fprintf(stderr, "Function definition within another function or\n");
			fprintf(stderr, "within a control structure (FUNCTION %s).\n", (yyvsp[(2) - (2)].str));
			yyerror("");
			/* No Return */
		      }

		    InFunctionDefinition++;
		    NextLocal = 0;
		    rp = SymtabNew(&GlobalSymbols, (yyvsp[(2) - (2)].str));
		    if (rp->r_type != 0 && rp->r_type != FUNCTION)
			fprintf(stderr, "WARNING: function name '%s' is redefining a variable!\n", (yyvsp[(2) - (2)].str));

		    rp->r_type = FUNCTION;

		    LocalSymbols = SymtabCreate();
		    v.r_str = (char *) LocalSymbols;
		    funcpn = PTNew(SL, v, NULL, NULL);
		    rp->r.r_str = (char *) funcpn;

		    free((yyvsp[(2) - (2)].str));
		    (yyval.pn) = funcpn;
		  ;}
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 676 "GenesisParser.ypp"
    {
		    ParseNode	*funcpn;
		    // ResultValue	v;
		    Result	*rp;
		    // char	*script;

		    rp = (Result *) (yyvsp[(2) - (2)].str);
		    if (InFunctionDefinition)
		      {
			fprintf(stderr, "Function definition within another function or\n");
			fprintf(stderr, "within a control structure (FUNCTION %s).\n", (yyvsp[(2) - (2)].str));
			yyerror("");
			/* No Return */
		      }

		    /*
		    ** Free old function parse tree and symtab
		    */

		    funcpn = (ParseNode *) rp->r.r_str;
		    if (funcpn->pn_val.r.r_str != NULL)
			SymtabDestroy((Symtab *)(funcpn->pn_val.r.r_str));
		    FreePTValues(funcpn->pn_left);
		    PTFree(funcpn->pn_left);
		    FreePTValues(funcpn->pn_right);
		    PTFree(funcpn->pn_right);

		    InFunctionDefinition++;
		    NextLocal = 0;
		    LocalSymbols = SymtabCreate();
		    funcpn->pn_val.r.r_str = (char *) LocalSymbols;

		    (yyval.pn) = funcpn;
		  ;}
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 713 "GenesisParser.ypp"
    {
		    ReturnIdents = 1;
		  ;}
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 717 "GenesisParser.ypp"
    {
		    ReturnIdents = 0;
		  ;}
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 721 "GenesisParser.ypp"
    {
		    InFunctionDefinition--;

		    (yyvsp[(1) - (6)].pn)->pn_left = (yyvsp[(3) - (6)].pn);
		    (yyvsp[(1) - (6)].pn)->pn_right = (yyvsp[(5) - (6)].pn);

		    LocalSymbols = NULL;
		    (yyval.pn) = NULL;
		  ;}
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 733 "GenesisParser.ypp"
    { (yyval.pn) = NULL; ;}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 735 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (3)].pn); ;}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 739 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    ParseNode	*init;

		    ArgMatch = 1;
		    v.r_int = ArgMatch++;
		    init = PTNew(DOLLARARG, v, NULL, NULL);
		    (yyval.pn) = vardef((yyvsp[(1) - (1)].str), STR, SCAST, init);
		    free((yyvsp[(1) - (1)].str));
		  ;}
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 750 "GenesisParser.ypp"
    {
		    ResultValue	v;
		    ParseNode	*init;

		    v.r_int = ArgMatch++;
		    init = PTNew(DOLLARARG, v, NULL, NULL);
		    v.r_str = (char *) MakeScriptInfo();
		    (yyval.pn) = PTNew(SL, v, (yyvsp[(1) - (3)].pn), vardef((yyvsp[(3) - (3)].str), STR, SCAST, init));
		    free((yyvsp[(3) - (3)].str));
		  ;}
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 763 "GenesisParser.ypp"
    {
		    ReturnIdents = 1;
		    DefType = INT;
		    DefCast = ICAST;
		  ;}
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 769 "GenesisParser.ypp"
    {
		    ReturnIdents = 1;
		    DefType = FLOAT;
		    DefCast = FCAST;
		  ;}
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 775 "GenesisParser.ypp"
    {
		    ReturnIdents = 1;
		    DefType = STR;
		    DefCast = SCAST;
		  ;}
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 783 "GenesisParser.ypp"
    {
		    (yyval.pn) = (yyvsp[(2) - (2)].pn);
		  ;}
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 790 "GenesisParser.ypp"
    {
		    ReturnIdents = 1;
		  ;}
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 794 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_str = (char *) MakeScriptInfo();
		    (yyval.pn) = PTNew(SL, v, (yyvsp[(1) - (4)].pn), (yyvsp[(4) - (4)].pn));
		  ;}
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 803 "GenesisParser.ypp"
    {
		    ReturnIdents = 0;
		  ;}
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 807 "GenesisParser.ypp"
    {
		    (yyval.pn) = vardef((yyvsp[(1) - (3)].str), DefType, DefCast, (yyvsp[(3) - (3)].pn));
		    free((yyvsp[(1) - (3)].str));
		  ;}
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 814 "GenesisParser.ypp"
    { (yyval.pn) = NULL; ;}
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 816 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (2)].pn); ;}
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 820 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    if (BreakAllowed)
			(yyval.pn) = PTNew(BREAK, v, NULL, NULL);
		    else
			yyerror("BREAK found outside of a loop");
			/* No Return */
		  ;}
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 832 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    (yyval.pn) = PTNew(RETURN, v, (yyvsp[(2) - (2)].pn), NULL);
		  ;}
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 838 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    (yyval.pn) = PTNew(RETURN, v, NULL, NULL);
		  ;}
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 846 "GenesisParser.ypp"
    { (yyval.pn) = NULL; ;}
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 850 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('|', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 852 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('&', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 854 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('^', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 856 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('~', RV, (yyvsp[(2) - (2)].pn), NULL); ;}
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 859 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('@', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 862 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('+', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 864 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('-', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 866 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('*', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 868 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('/', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 870 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('%', RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 872 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(POW, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 874 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(UMINUS, RV, (yyvsp[(2) - (2)].pn), NULL); ;}
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 877 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(OR, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 879 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(AND, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 881 "GenesisParser.ypp"
    { (yyval.pn) = PTNew('!', RV, (yyvsp[(2) - (2)].pn), NULL); ;}
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 884 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(LT, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 886 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(LE, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 888 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(GT, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 890 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(GE, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 892 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(EQ, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 894 "GenesisParser.ypp"
    { (yyval.pn) = PTNew(NE, RV, (yyvsp[(1) - (3)].pn), (yyvsp[(3) - (3)].pn)); ;}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 897 "GenesisParser.ypp"
    { 
		    Result	*rp;
		    ResultValue	v;

		    /*
		    ** Variable reference
		    */

		    rp = (Result *) (yyvsp[(1) - (1)].str);
		      {
			if (rp->r_type == FUNCTION || rp->r_type == LOCREF)
			    v = rp->r;
			else /* Global Variable */
			    v.r_str = (char *) rp;

		        (yyval.pn) = PTNew(rp->r_type, v, NULL, NULL);
		      }
 		  ;}
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 917 "GenesisParser.ypp"
    { 
		    ResultValue	v;

		    v.r_float = (yyvsp[(1) - (1)].fconst);
		    (yyval.pn) = PTNew(FLOATCONST, v, NULL, NULL);
 		  ;}
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 924 "GenesisParser.ypp"
    { 
		    ResultValue	v;

		    v.r_int = (yyvsp[(1) - (1)].iconst);
		    (yyval.pn) = PTNew(INTCONST, v, NULL, NULL);
 		  ;}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 931 "GenesisParser.ypp"
    { 
		    ResultValue	v;

		    v.r_str = (yyvsp[(1) - (1)].str);
		    (yyval.pn) = PTNew(STRCONST, v, NULL, NULL);
 		  ;}
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 939 "GenesisParser.ypp"
    {
		    ResultValue	v;

		    v.r_int = (yyvsp[(1) - (1)].iconst);
		    (yyval.pn) = PTNew(DOLLARARG, v, NULL, NULL);
		  ;}
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 947 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (3)].pn); ;}
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 950 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (3)].pn); ;}
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 953 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (3)].pn); ;}
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 956 "GenesisParser.ypp"
    { (yyval.pn) = (yyvsp[(2) - (3)].pn); ;}
    break;



/* Line 1455 of yacc.c  */
#line 2904 "GenesisParser.tab.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
        ;// Deleted YYSTACK_FREE( yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 960 "GenesisParser.ypp"



/*
** TokenStr
**
**	Return the token string for the given token.
*/

char *TokenStr(int token)
{	/* TokenStr --- Return token string for token */

	static char	buf[100];

	switch (token)
	  {

	  case LT: return("<");
	  case LE: return("<=");
	  case GT: return(">");
	  case GE: return(">=");
	  case EQ: return("==");
	  case NE: return("!=");

	  case OR: return("||");
	  case AND: return("&&");

#define	RET(tok)	case tok: return("tok")

	  RET(UMINUS);

	  RET(WHILE);
	  RET(IF);
	  RET(ELSE);
	  RET(FOR);
	  RET(FOREACH);
	  RET(END);
	  RET(INCLUDE);
	  RET(BREAK);
	  RET(INT);
	  RET(FLOAT);
	  RET(STR);
	  RET(RETURN);
	  RET(WHITESPACE);
	  RET(FUNCTION);
	  RET(INTCONST);
	  RET(DOLLARARG);
	  RET(FLOATCONST);
	  RET(STRCONST);
	  RET(LITERAL);
	  RET(IDENT);
	  RET(VARREF);
	  RET(FUNCREF);
	  RET(SL);
	  RET(COMMAND);
	  RET(ARGUMENT);
	  RET(ARGLIST);
	  RET(LOCREF);
	  RET(ICAST);
	  RET(FCAST);
	  RET(SCAST);

	  }

	if (token < 128)
	    if (token < ' ')
		sprintf(buf, "^%c", token+'@');
	    else
		sprintf(buf, "%c", token);
	else
	    sprintf(buf, "%d", token);

	return(buf);

}	/* TokenStr */


ParseNode * myFlexLexer::vardef(char* ident, int type, int castop, ParseNode* init)
{	/* vardef --- Define a variable */

	ParseNode	*pn;
	Result		*rp;
	// Result		*r;
	ResultValue	v, slv;

	if (InFunctionDefinition && LocalSymbols != NULL)
	  {
	    rp = SymtabNew(LocalSymbols, ident);
	    if (rp->r_type == 0)
	      {
	        rp->r_type = LOCREF;
		rp->r.r_loc.l_type = type;
		rp->r.r_loc.l_offs = NextLocal++;
	      }

	    v.r_str = (char *) rp;
	    pn = PTNew(castop, v, NULL, NULL);
	    if (init)
	      {
		slv.r_str = (char *) MakeScriptInfo();
		pn = PTNew(SL, slv, pn, PTNew('=', v, init, NULL));
	      }
	  }
	else
	  {
	    rp = SymtabNew(&GlobalSymbols, ident);
	    switch(type)
	      {

	      case INT:
	        if (rp->r_type == 0)
	            rp->r.r_int = 0;
	        else
		    CastToInt(rp);
	        break;

	      case FLOAT:
	        if (rp->r_type == 0)
	            rp->r.r_float = 0.0;
	        else
		    CastToFloat(rp);
	        break;

	      case STR:
	        if (rp->r_type == 0)
	            rp->r.r_str = (char *) strsave("");
	        else
		    CastToStr(rp);
	        break;

	      }

	    rp->r_type = type;
	    v.r_str = (char *) rp;
	    if (init)
	        pn = PTNew('=', v, init, NULL);
	    else
	        pn = NULL;
	  }

	return(pn);

}	/* vardef */


void myFlexLexer::ParseInit()

{    /* ParseInit --- Initialize parser variables */

        InFunctionDefinition = 0;
	Compiling = 0;
	BreakAllowed = 0;
	LocalSymbols = NULL;
	nextchar(1);	/* Flush lexer input */
	PTInit();	/* Reinit parse tree evaluation */

}    /* ParseInit */


int myFlexLexer::NestedLevel()

{    /* NestedLevel --- Return TRUE if in func_def or control structure */

        return(InFunctionDefinition || Compiling);

}    /* NestedLevel */


