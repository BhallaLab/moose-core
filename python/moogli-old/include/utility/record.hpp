#ifndef __RECORD__
#define __RECORD__

#include <string>
#include <iostream>

using namespace std;

enum class record_t { ERROR   =   0
                    , INFO    =   1
                    };

void
record_error( const string & file_name
            , const string & function_name
            , const unsigned int line_number
            , const string & message
            );

void
record_info( const string & file_name
           , const string & function_name
           , const unsigned int line_number
           , const string & message
);



void
record( record_t record_type
      , const string & file_name
      , const string & function_name
      , const unsigned int line_number
      , const string & message
      );

#define RECORD(type, message) (record(type, __FILE__, __FUNCTION__, __LINE__, message))

#define RECORD_ERROR(message) RECORD(record_t::ERROR, message)

#define RECORD_INFO(message) RECORD(record_t::INFO, message)

#endif  /* __RECORD__ */
