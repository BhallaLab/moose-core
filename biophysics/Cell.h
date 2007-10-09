/*******************************************************************
 * File:            Cell.h
 * Description:      Generic container for interconnected objects
 *                   that can be handled by a single solver.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-02 13:36:46
 ********************************************************************/

#ifndef _CELL_H
#define _CELL_H
class Cell
{
  public:
    Cell();
    static void processFunc( const Conn& c, ProcInfo info );
    void processFuncLocal( Element* e, ProcInfo info );
    static void reinitFunc( const Conn& c, ProcInfo info );
    void reinitFuncLocal(  Element* e, ProcInfo info );

  private:
    

};


#endif
