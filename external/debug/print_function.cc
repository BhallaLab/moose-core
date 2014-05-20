/*
 * =====================================================================================
 *
 *       Filename:  print_function.cc
 *
 *    Description:  Print function.
 *
 *        Version:  1.0
 *        Created:  07/25/2013 02:48:49 AM
 *       Revision:  none
 *       Compiler:  gcc/g++
 *
 *         Author:  Dilawar Singh (), dilawar@ee.iitb.ac.in
 *   Organization:  IIT Bombay
 *
 * =====================================================================================
 */

#include "print_function.h"

#include <ctime>
#include <algorithm>

string colored(string msg)
{
  stringstream ss;
  ss << T_RED << msg << T_RESET;
  return ss.str();
}

string colored(string msg, string colorName)
{
  stringstream ss;
  ss << colorName << msg << T_RESET;
  return ss.str();
}

string debugPrint(string msg, string prefix, string color, unsigned debugLevel) 
{
  stringstream ss; ss.str("");
  if(debugLevel <= DEBUG_LEVEL)
  {
    ss << setw(debugLevel/2) << "[" << prefix << "] " 
      << color << msg << T_RESET;
  }
  return ss.str();
}

/*-----------------------------------------------------------------------------
 *  This function dumps a message onto console. Fills appropriate colors as
 *  needed. What can I do, I love colors.
 *-----------------------------------------------------------------------------*/
void dump(string msg, string type, bool autoFormat)
{
    stringstream ss;
    ss << "[" << type << "] ";
    bool set = false;
    bool reset = true;
    string color = T_GREEN;
    if(type == "WARNING" || type == "WARN" || type == "FIXME")
        color = T_YELLOW;
    else if(type == "DEBUG" || type == "EXPECT_FAILURE" || type == "EXPECT")
        color = T_CYAN;
    else if(type == "ERROR" || type == "FAIL")
        color = T_RED;
    else if(type == "INFO")
        color = T_BLUE;
    else if(type == "LOG")
       color = T_MAGENTA;

    for(unsigned int i = 0; i < msg.size(); ++i)
    {
        if('`' == msg[i])
        {
            if(!set and reset) 
            {
                set = true;
                reset = false;
                ss << color;
            }
            else if(set && !reset)
            {
                reset = true;
                set = false;
                ss << T_RESET;
            }
        }
        else if('\n' == msg[i])
            ss << "\n + ";
        else
            ss << msg[i];
    }

    /*  Be safe than sorry */
    if(!reset)
        ss << T_RESET;
    cerr << ss.str() << endl;
}

/*-----------------------------------------------------------------------------
 *  Log to a file, and also to console.
 *-----------------------------------------------------------------------------*/
bool isBackTick(char a)
{
    if('`' == a)
        return true;
    return false;
}

void log(string msg, string type, bool redirectToConsole, bool removeTicks)
{
    if(redirectToConsole)
        dump(msg, type, true);

    /* remove any backtick from the string. */
    if(removeTicks)
        remove_if(msg.begin(), msg.end(), isBackTick);
    
    fstream logF;
    logF.open("__moose__.log", ios::app);

    time_t rawtime; time(&rawtime);
    struct tm* timeinfo;
    timeinfo = localtime(&rawtime);

    logF << asctime(timeinfo) << ": " << msg;

    logF.close();
}


