/*******************************************************************
 * File:            Cell.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-02 13:38:29
 ********************************************************************/
#ifndef _CELL_CPP
#define _CELL_CPP

#include "moose.h"
#include "../element/Wildcard.h"
#include "HSolveStructure.h"
#include "NeuroHub.h"
#include "NeuroScanBase.h"
#include "NeuroScan.h"
#include "HSolveBase.h"
#include "HSolve.h"
#include "ThisFinfo.h"
#include "Cell.h"
#include "header.h"
#include "../element/Neutral.h"
#include <iostream>
#include <sstream>
const Cinfo * initCellCinfo()
{
    static Finfo* processShared[] =
	{
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                           RFCAST( &Cell::processFunc ) ),
            new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                           RFCAST( &Cell::reinitFunc ) ),
	};
    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof( processShared ) / sizeof( Finfo* ) );
    static Finfo* cellFinfos[] = 
        {            
            process,
        };
    static SchedInfo schedInfo[] = { { process, 0, 0 } };
    static Cinfo cellCinfo("Cell",
                           "Subhasis Ray",
                           "Cell object. Container for applying solvers.",
                           initNeutralCinfo(), // Cell is an extension of Neutral
                           cellFinfos,
                           sizeof( cellFinfos)/sizeof(Finfo*),
                           ValueFtype1 < Cell > :: global(),
                           schedInfo,
                           1
        );
    return &cellCinfo;    
}

static const Cinfo* cellInfo = initCellCinfo();

Cell::Cell()
{
}

void Cell::processFunc( const Conn& c, ProcInfo info )
{
    Element* e = c.targetElement();
    static_cast< Cell* >( e->data() )->processFuncLocal( e, info );
}

void Cell::processFuncLocal( Element* e, ProcInfo info )
{   
}

void Cell::reinitFunc( const Conn& c, ProcInfo info )
{
    Element* e = c.targetElement();
    assert(e!=0);    
    static_cast< Cell* >( e->data() )->reinitFuncLocal( e, info );
}

void Cell::reinitFuncLocal( Element* e, ProcInfo info)
{
    static Id nsolvers( "/solvers/neuronal" );
    if (nsolvers.bad())
    {
        cerr << "Error: Cell::reinitFuncLocal - \"/solvers/neuronal\" does not exist!" << endl;
        return;        
    }
    
    vector <Id> children = Neutral::getChildList(e);
    if ( children.empty() )
    {
        return;
    }
    else // there are compartments under this, create solver if not already created
    {
        
        Element* seed = 0;
        
        for ( vector <Id>::iterator i = children.begin(); i != children.end(); ++i )
        {
            Id seedId = *i;
            seed = seedId();            
            if( seed->cinfo()->isA( Cinfo::find( "Compartment" ) ) )
            {
                // Where to create solver? Under /solvers/neuronal/
                // The name is integ<Id of this cell>
                std::stringstream solverName(std::stringstream::out);
                solverName << "integ" << e->id();
                std::string solverPath = "/solvers/neuronal/";
                solverPath += solverName.str();
        
                // See if neuronal solver exists for this Cell
                Id nsolve(solverPath);
        
                if ( !nsolve.bad() ) { // alter existing nsolve
                    set( nsolve(), "path", (seed->id()).path() );
                } else {
                    // make a new solver
                    Element* ni = Neutral::create( "HSolve", solverName.str(), nsolvers() );
                    set( ni, "path", (seed->id()).path());
                    
                    // TODO: Check this part - each cell should set the clock
                    // corresponding to its SchedInfo to the associated
                    // solver.  What is tick#4 for ??
                    // Id cj("/sched/cj");
//                     vector <Id> childList = Neutral::getChildList(cj());
            
//                     childList[4]()->findFinfo( "process" )->add(
//                         childList[4](), ni, ni->findFinfo( "process" ) );
                }
                return;
            }        
        }        
    }    
}

#endif
