/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

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
     INT = 278,
     FLOAT = 279,
     STR = 280,
     RETURN = 281,
     WHITESPACE = 282,
     FUNCTION = 283,
     INTCONST = 284,
     DOLLARARG = 285,
     FLOATCONST = 286,
     STRCONST = 287,
     LITERAL = 288,
     IDENT = 289,
     VARREF = 290,
     FUNCREF = 291,
     EXTERN = 292,
     SL = 293,
     COMMAND = 294,
     EXPRCALL = 295,
     ARGUMENT = 296,
     ARGLIST = 297,
     LOCREF = 298,
     ICAST = 299,
     FCAST = 300,
     SCAST = 301
   };
#endif
/* Tokens.  */
#define AND 258
#define OR 259
#define NE 260
#define EQ 261
#define GE 262
#define GT 263
#define LE 264
#define LT 265
#define POW 266
#define UMINUS 267
#define WHILE 268
#define IF 269
#define ELSE 270
#define ELIF 271
#define FOR 272
#define FOREACH 273
#define END 274
#define INCLUDE 275
#define ENDSCRIPT 276
#define BREAK 277
#define INT 278
#define FLOAT 279
#define STR 280
#define RETURN 281
#define WHITESPACE 282
#define FUNCTION 283
#define INTCONST 284
#define DOLLARARG 285
#define FLOATCONST 286
#define STRCONST 287
#define LITERAL 288
#define IDENT 289
#define VARREF 290
#define FUNCREF 291
#define EXTERN 292
#define SL 293
#define COMMAND 294
#define EXPRCALL 295
#define ARGUMENT 296
#define ARGLIST 297
#define LOCREF 298
#define ICAST 299
#define FCAST 300
#define SCAST 301




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

