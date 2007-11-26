/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
** Large parts of this individual file have been copied from files
** in the genesis/src/ss directory. These are authored by 
** Dave Bilitch, and copyright Caltech
** under terms which permit free redistribution. Note that these
** files are NOT GPL.
** Additional modifications and conversion to C++ are
**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
**********************************************************************/

#ifndef _PARGENESIS_PARSER_H
#define _PARGENESIS_PARSER_H


class ParGenesisParserWrapper:public GenesisParserWrapper
{
public:
	ParGenesisParserWrapper();
	void MyFunc();
	void loadBuiltinCommands();
};

#endif	//_PARGENESIS_PARSER_H

