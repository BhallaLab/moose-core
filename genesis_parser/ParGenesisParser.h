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

static const int MAX_COMMAND_ARGS = 25;
static const int MAX_COMMAND_SIZE = 1024;

struct stCommand
{
	int iSize;
	char arrCommand[MAX_COMMAND_ARGS][MAX_COMMAND_SIZE];

	stCommand()
	{
		clear();
	};

	void clear()
	{
		iSize = 0;
		memset(arrCommand, 0, MAX_COMMAND_ARGS*MAX_COMMAND_SIZE*sizeof(char));
	};

};


class ParGenesisParserWrapper:public GenesisParserWrapper
{
public:
	ParGenesisParserWrapper();
	void MyFunc();
	void loadBuiltinCommands();

	static void parseFunc( const Conn& c, string s );
	virtual Result ExecuteCommand(int argc, char** argv);
	int SendCommand(int argc);
	struct stCommand objCommand_; 

private:
	int	processrank_;
	int	processcount_;
};

#endif	//_PARGENESIS_PARSER_H

