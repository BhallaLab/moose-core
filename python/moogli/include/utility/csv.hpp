#ifndef _CSV_HPP_
#define _CSV_HPP_

#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <climits>

using namespace std;

typedef vector<string> CsvRow;
typedef list<CsvRow> Csv;

bool
read_csv( ifstream& file
        , CsvRow&   row
        , char      line_delimiter  = '\n'
        , char      field_delimiter = ','
        );

unsigned int
read_csv( const string& filename
        , Csv& csv
        , char field_delimiter           = ','
        , char line_delimiter            = '\n'
        , unsigned int line_count        = UINT_MAX
        , unsigned int column_count_hint = 0
        , unsigned int row_count_hint    = 0
        );

void
write_csv( const Csv& csv
         , ostream& os
         , char field_delimiter = ','
         , char line_delimiter  = '\n'
         );

unsigned int
read_csv_with_stats( const string& filename
                   , Csv&  csv
                   );

#endif /* _CSV_HPP_ */
