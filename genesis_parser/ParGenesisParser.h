/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This class is a parallel version of Moose parser.
 * This file is compiled only when the parallelizatin flag is enabled.
 * This class derives from GenesisParserWrapper, the parser class for serial Moose. 
 *
 * 
 * This class refers to the base class, GenesisParserWrapper, for all parser functionality. 
 * Parallel moose parser requires overriding of some of the base class functionality. 
 * Such functions are overridden in this class. Except the overridden functions the parallel parser
 * refers to the base serial parser for all it's functionality.
 *
 * 
 * Parallel moose parser parses the input script file only on the root process. Each parsed command 
 * is then sent using MPI to each of the non-root processes. 
 * 
 */

#ifndef _PARGENESIS_PARSER_H
#define _PARGENESIS_PARSER_H

static const int MAX_COMMAND_ARGS = 25;
static const int MAX_COMMAND_SIZE = 1024;

struct stCommand
{
	int iSize;
	int iRank;
	char arrCommand[MAX_COMMAND_ARGS][MAX_COMMAND_SIZE];

	stCommand()
	{
		clear();
	};

	void clear()
	{
		iSize = 0;
		iRank = 0;
		memset(arrCommand, 0, MAX_COMMAND_ARGS*MAX_COMMAND_SIZE*sizeof(char));
	};

};


class ParGenesisParserWrapper:public GenesisParserWrapper
{
public:
	ParGenesisParserWrapper();
	/**
	 * This function overrides functions to be called for a moose command. A map is populated by the serial parser
	 * which maintains the functions to be called for every moose command. This function overrides some of these 		 * functions with for parallel moose. 
	 */
	void loadBuiltinCommands();


	/**
	 * This function overrides the "GenesisParserWrapper::parserFunc" function for parallel moose. 
	 * This function is called by both the root and non root processes. 
	 * The script file is parsed and broadcast on the root process and commands are 
	 * received and executed on the non-root processes.
	 */
	static void parseFunc( const Conn& c, string s );
	
	/**
	 * This virtual function overrides the "myFlexLexer::ExecuteCommand" function for parallel moose.
	 * When this function is called, parsed commands are sent to non-root processes on the root process
	 * and the command is executed on the non-root processes. The function is virtual to ensure that the overridded
	 * function gets called when invoked from the "myFlexLexer" class.
	 */
	virtual Result ExecuteCommand(int argc, char** argv);

	/**
	 * This function broadcasts the parsed command to all non-root processes
	 */
	int SendCommand(int argc);
	
	/**
	 * This function handles visulalization functionality after a step command has been sent
	 */
	void PostStepCommand();

	/**
	 * This function broadcasts the neuronal connections to all neurons
	 */
	void BCastConnections();

	/**
	 * This function randomly connects all neurons
	 */
        void generateRandomConnections();


	/**
	 * This function checks if a connection with a neuron repeats
	 */
        bool checkUnique(int randomVar, int spikeIndex);

	/**
	 * This function checks if a command should be executed on the root process
	 */
        bool RootCommand(char **argv);

	/**
	 * This function accepts visualization data and displays it using gnuplot
	 */
        void DisplayData(void* pArgs);

	/**
	 * This object contains the parsed command to be sent to the non-root processes on MPI
	 */
	struct stCommand objCommand_; 

private:
	/**
	 * The rank of the executing process.
	 */
	int	processrank_;

	/**
	 * The total count of processes.
	 */
	int	processcount_;

	/**
	 * The rank to which a command must be sent
	 */
	int	sendrank_;
};

#endif	//_PARGENESIS_PARSER_H

