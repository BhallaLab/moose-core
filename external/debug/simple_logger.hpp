/*
 * ==============================================================================
 *
 *       Filename:  moose_logger.h
 *
 *    Description:  A simple XML based logger for moose.
 *
 *        Version:  1.0
 *        Created:  Saturday 24 May 2014 06:25:10  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawar@ee.iitb.ac.in
 *   Organization:  
 *
 * ==============================================================================
 */

#ifndef  MOOSE_LOGGER_INC
#define  MOOSE_LOGGER_INC

#include <sstream>
#include <string>
#include <ctime>
#include <numeric>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

class SimpleLogger {

    public:
        SimpleLogger()
        {
            outputFile = "";
            startTime = timeStamp();
        }

        ~SimpleLogger()
        {
        }

        /**
         * @brief Get current timestamp.
         *
         * @return  A string represeting current timestamp.
         */
        const std::string timeStamp() 
        {
            time_t     now = time(0);
            struct tm  tstruct;
            char       buf[80];
            tstruct = *localtime(&now);
            strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
            return buf;
        }

        /**
         * @brief When an element is created in moose, log its presense in this
         * map.
         *
         * @param type Type of element.
         * @param path Path of the element.
         */
        void updateGlobalCount(string type)
        {
            if(elementsMap.find(type) == elementsMap.end())
                elementsMap[type] = 1;
            else
                elementsMap[type] = elementsMap[type] + 1;
        }

        template<typename A, typename B>
        string mapToString(const map<A, B>& m, string title = "") const
        {
            unsigned width = 50;
            stringstream ss;

            ss << title;
            for(unsigned i = 0; i < width - title.size(); ++i) 
                ss << "~";
            ss << endl;

            typename map<A, B>::const_iterator it;
            for( it = m.begin(); it != m.end(); it++)
                ss << setw(width/2) << it->first << setw(width/2) << it->second << endl;

            for(unsigned i = 0; i < width; ++i) 
                ss << "=";

            ss << endl;
            return ss.str();
        }

        /**
         * @brief Dump statistics onto console.
         *
         * @param which If which is 0 then print elements inside moose, if it is
         * 1 then print total time taken during simulation.
         *
         * @return 
         */
        string dumpStats( int which )
        {
            stringstream ss;
            unsigned width = 50;
            ss << endl;
            if(which == 0)
                ss << mapToString<string, unsigned long>(elementsMap, "Elements");

            else if( which == 1)
            {
                timekeeperMap["Simulation"] = accumulate(
                        simulationTime.begin()
                        , simulationTime.end()
                        , 0.0
                        );
                timekeeperMap["Initialization"] = accumulate(
                        initializationTime.begin()
                        , initializationTime.end()
                        , 0.0
                        );
                timekeeperMap["Creation"] = accumulate(
                        creationTime.begin()
                        , creationTime.end()
                        , 0.0
                        );

                ss << mapToString<string, float>( timekeeperMap, "Simulation stats" );
            }
            return ss.str();
        }


        template<typename A, typename B>
        void mapToXML(ostringstream& ss
                , const map<A, B>& m
                , const char* tagName
                , unsigned indent
                ) const
        {
            string prefix = "";
            for(int i = 0; i < indent; ++i)
                prefix += "\t";
            ss << prefix << "<" << tagName << ">" << endl;

            typename map<A, B>::const_iterator it;
            for(it = m.begin(); it != m.end(); it++)
            {
                ss << prefix << prefix 
                    << "<" << it->first << ">" 
                    << it->second 
                    << "</" << it->first << ">" << endl;
            }

            ss << prefix << "</" << tagName << ">" << endl;
        }

        /**
         * @brief Convert this logger to XML.
         *
         * @return A XML string.
         */
        string loggerToXML( const char* outFile = "moose_logger.log" )
        {
            logSS.str("");
            logSS << "<simulation simulator=\"moose\">" << endl;
            logSS << "\t<start_time>" << startTime << "</start_time>" << endl;
            
            mapToXML<string, unsigned long>(logSS, elementsMap, "elements", 1);
            mapToXML<string, float>(logSS, timekeeperMap, "times", 1);

            logSS << "\t<end_time>" << timeStamp() << "</end_time>" << endl;
            logSS << "</simulation>" << endl;
            fstream logF;
            logF.open(outFile, std::fstream::out | std::fstream::app);
            logF << logSS.str();
            logF.close();
            return logSS.str();
        }
        
    private:
        map<string, unsigned long> elementsMap;
        map<string, float> timekeeperMap;

    public:

        string outputFile;
        string startTime;
        string endTime;

        ostringstream logSS;

        /* Map to keep simulation run-time data. */
        vector<float> simulationTime;
        vector<float> initializationTime;
        vector<float> creationTime;
};

extern SimpleLogger logger;

#endif   /* ----- #ifndef MOOSE_LOGGER_INC  ----- */
