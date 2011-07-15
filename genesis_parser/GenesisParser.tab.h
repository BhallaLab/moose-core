
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
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



