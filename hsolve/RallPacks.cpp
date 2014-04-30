/*
 * ==============================================================================
 *
 *       Filename:  RallPacks.cpp
 *
 *    Description:  This contains Rallpacks benchmarks. These benchmarks are
 *    published here 
 *
 *    http://link.springer.com/content/pdf/10.1007%2F978-1-4615-3254-5_21.pdf
 *
 *        Version:  1.0
 *        Created:  Wednesday 30 April 2014 01:06:26  IST
 *       Revision:  0.1
 *       Compiler:  g++
 *
 *         Author:  Dilawar Singh
 *
 *   Organization:  NCBS Bangalore
 *
 * ==============================================================================
 */


#include "RallPacks.h"


/*
 *-------------------------------------------------------------------------------
 *       Class:  RallPacks
 *      Method:  RallPacks
 * Description:  constructor
 *-------------------------------------------------------------------------------
 */
RallPacks::RallPacks ()
{
    
}  /* -----  end of method RallPacks::RallPacks  (constructor)  ----- */

/*
 *-------------------------------------------------------------------------------
 *       Class:  RallPacks
 *      Method:  RallPacks
 * Description:  copy constructor
 *-------------------------------------------------------------------------------
 */
RallPacks::RallPacks ( const RallPacks &other )
{
}  /* -----  end of method RallPacks::RallPacks  (copy constructor)  ----- */

/*
 *-------------------------------------------------------------------------------
 *       Class:  RallPacks
 *      Method:  ~RallPacks
 * Description:  destructor
 *-------------------------------------------------------------------------------
 */
RallPacks::~RallPacks ()
{
}  /* -----  end of method RallPacks::~RallPacks  (destructor)  ----- */

/*
 *-------------------------------------------------------------------------------
 *       Class:  RallPacks
 *      Method:  operator =
 * Description:  assignment operator
 *-------------------------------------------------------------------------------
 */
RallPacks& RallPacks::operator = ( const RallPacks &other )
{
    if ( this != &other )
    {
    }
    return *this;
}  /* -----  end of method RallPacks::operator =  (assignment operator)  ----- */


/* 
 * ===  FUNCTION  ===============================================================
 *         Name:  runRallpackBenchmarks
 *  Description:  Function which runs Rallpack.
 * ==============================================================================
 */
void runRallpackBenchmarks()
{
    tbegin;
    tend;
}
