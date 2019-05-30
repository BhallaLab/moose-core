/***
 *    Description:  MooseParser class. 
 *
 *        Created:  2019-05-30

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */

#ifndef MOOSEPARSER_H
#define MOOSEPARSER_H

#include "../external/muparser/include/muParser.h"

class MooseParser : public mu::Parser
{
    public:
        MooseParser();
        ~MooseParser();

    private:
        /* data */
};

#endif /* end of include guard: MOOSEPARSER_H */
