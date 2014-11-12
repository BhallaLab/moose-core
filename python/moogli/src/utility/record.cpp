#include "utility/record.hpp"

using namespace std;

void
record( record_t record_type
      , const string & file_name
      , const string & function_name
      , const unsigned int line_number
      , const string & message
      )
{
    switch(record_type)
    {
        case record_t::ERROR      :   record_error( file_name
                                        , function_name
                                        , line_number
                                        , message
                                        );
                            break;
        case record_t::INFO       :   record_info( file_name
                                       , function_name
                                       , line_number
                                       , message
                                       );
                            break;
        default         :   break;
    }

}

void
record_error( const string & file_name
            , const string & function_name
            , const unsigned int line_number
            , const string & message
            )
{
    cerr << "[ERROR]";
    cerr << "-";
    cerr << "[" + file_name     + "]";
    cerr << "-";
    cerr << "[" + function_name + "]";
    cerr << "-";
    cerr << "[" + to_string(line_number)   + "]";
    cerr << "-";
    cerr << message;
    cerr << endl;
}

void
record_info( const string & file_name
           , const string & function_name
           , const unsigned int line_number
           , const string & message
)
{
    cout << "[INFO]";
    cout << "-";
    cout << "[" + file_name     + "]";
    cout << "-";
    cout << "[" + function_name + "]";
    cout << "-";
    cout << "[" + to_string(line_number)   + "]";
    cout << "-";
    cout << message;
    cout << endl;
}

void
record_benchmark()
{

}
