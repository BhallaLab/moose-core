// HDF5DataWriter.h --- 
// 
// Filename: HDF5DataWriter.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 15:47:23 2012 (+0530)
// Version: 
// Last-Updated: Sat Feb 25 20:07:01 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 27
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Specialization of HDF5WriterBase to save Table objects in
// MOOSE. The table can be regular table or Stimulus table. The data
// is appended to an existing dataset at each process step. An
// explicit writing is also allowed via the flush command.  As soon as
// the data from a table is written to file, the table is cleared.
// 

// Change log:
// 
// 2012-02-25 15:50:03 (+0530) Subha - started coding this class
// 

// Code:
#ifdef USE_HDF5
#ifndef _HDF5DATAWRITER_H

#include "HDF5WriterBase.h"

class HDF5DataWriter: public HDF5WriterBase
{
  public:
    static const hssize_t CHUNK_SIZE;
    HDF5DataWriter();
    virtual ~HDF5DataWriter();
    void flush();
    void process(const Eref &e, ProcPtr p);
    void reinit(const Ered &e, ProcPtr p);
    virtual void addObject(string path);
    static const Cinfo* initCinfo();
  protected:
    hid_t get_dataset(ObjId id);
    hid_t create_dataset(hid_t parent, string name, ObjId oid);
};
#endif // _HDF5DATAWRITER_H
#endif // USE_HDF5

// 
// HDF5DataWriter.h ends here
