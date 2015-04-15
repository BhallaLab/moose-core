#include "utility/conversions.hpp"
#include "utility/stringutils.hpp"
#include "utility/fileutils.hpp"
#include "utility/csv.hpp"

#include <fstream>
#include <string>
#include <streambuf>
#include <vector>
#include <list>
#include <chrono>
#include <iostream>

using namespace std;

typedef vector<string> CsvRow;
typedef list<CsvRow> Csv;

bool
read_csv( ifstream& file
        , CsvRow&   row
        , char      line_delimiter
        , char      field_delimiter
        )
{
    string line;
    if(getline(file, line, line_delimiter))
    {
        tokenize(line, field_delimiter, row);
        return true;
    }
    else
    {
        return false;
    }
}

// unsigned int
// read_csv( const string& filename
//         , Csv& csv
//         , char field_delimiter           = ','
//         , char line_delimiter            = '\n'
//         , unsigned int line_count        = UINT_MAX
//         , unsigned int column_count_hint = 0
//         , unsigned int row_count_hint    = 0
//         )
// {

//     ifstream file(filename.c_str(), ifstream::in );
//     if(!file)
//     {
//         cerr << "[I/O Error] Unable to open: " << filename;
//         exit(1);
//     }

//     CsvRow csv_row;

//     while(line_count && read_csv_row(file, csv_row))
//     {
//         csv.push_back(csv_row);
//         --line_count;
//     }

//     file.close();
//     return csv.size();
// }


unsigned int
read_csv( const string& filename
        , Csv& csv
        , char field_delimiter
        , char line_delimiter
        , unsigned int line_count
        , unsigned int column_count_hint
        , unsigned int row_count_hint
        )
{
    string buffer;

    unsigned int row_index    = 0;
    unsigned int column_start = 0;
    unsigned int file_size    = file_get_contents(filename, buffer);

    CsvRow csv_row;

    for(unsigned int i = 0; row_index < line_count && i < file_size; ++i)
    {
        if(buffer[i] == line_delimiter)
        {
            string row;
            row.assign(buffer, column_start, i - column_start);
            csv_row.push_back(move(row));
            csv.push_back(move(csv_row));
            csv_row = CsvRow();
            ++ row_index;
            column_start = i + 1;
            continue;
        }
        if(buffer[i] == field_delimiter)
        {
            string row;
            row.assign( buffer, column_start, i - column_start);
            csv_row.push_back(move(row));
            column_start = i + 1;
            continue;
        }
    }
    return row_index;
}

void
write_csv( const Csv& csv
         , ostream& os
         , char field_delimiter
         , char line_delimiter
         )
{
    unsigned int row_index = 0;
    unsigned int col_index = 0;

    if(csv.size() == 0) { return; }

    Csv::const_iterator iter = csv.begin();

    for(row_index = 0; row_index < csv.size() - 1; ++row_index)
    {
        const CsvRow& row = *iter;

        if(row.size() == 0) { os << line_delimiter; continue; }

        for(col_index = 0; col_index < row.size() - 1; ++col_index)
        {
            os << row[col_index] << field_delimiter;
        }
        os << row[col_index] << line_delimiter;
        ++iter;
    }

    const CsvRow& row = *iter;

    if(row.size() == 0) { return; }

    for(col_index = 0; col_index < row.size() - 1; ++col_index)
    {
        os << row[col_index] << field_delimiter;
    }
    os << row[col_index];
}

unsigned int
read_csv_with_stats( const string& filename
                   , Csv&  csv
                   )
{

    if(!file_exists(filename))
    {
        cerr << "[I/O Error] Unable to open: "
             << filename
             << endl;
        exit(1);
    }

    unsigned int row_count;

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    row_count = read_csv(filename, csv);
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();

    cerr << "Read "
         << row_count
         << " rows in "
         << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
         << " nanoseconds from "
         << filename
         << endl;

    return row_count;
}
