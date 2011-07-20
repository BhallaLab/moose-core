// Efield.h --- 
// 
// Filename: Efield.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Wed Jul 20 14:40:35 2011 (+0530)
// Version: 
// Last-Updated: Wed Jul 20 17:19:56 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 14
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Port of efield class in GENESIS.
// This object calculates the field potential due to compartments connected to it.
// 

// Change log:
// 
// 
// 

// Code:

#ifndef _EFIELD_H
#define _EFIELD_H

class Efield
{
  public:
    Efield();

    ////////////////////////////////////////
    // Field functions
    ////////////////////////////////////////
    static void setScale(const Conn * c, double value);
    static double getScale(Eref e);
    static double getPotential(Eref e);
    static void setX(const Conn * c, double value);
    static void setY(const Conn * c, double value);
    static void setZ(const Conn * c, double value);
    static double getX(Eref e);
    static double getY(Eref e);
    static double getZ(Eref e);
    static void currentFunc(const Conn * conn, double value);
    static void processFunc(const Conn * conn, ProcInfo proc);
    static void reinitFunc(const Conn * conn, ProcInfo proc);
    void innerReinitFunc(const Conn * conn, ProcInfo proc);
    void updateDistances(Eref eref);
  protected:
    double x_;
    double y_;
    double z_;
    double scale_;
    double pot_;
    double innerPot_;
    vector <double> distance_;
};

#endif
// 
// Efield.h ends here
