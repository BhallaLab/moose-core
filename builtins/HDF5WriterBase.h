// HDF5WriterBase.h --- 
// 
// Filename: HDF5WriterBase.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 14:39:19 2012 (+0530)
// Version: 
// Last-Updated: Mon Feb 27 23:05:31 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 48
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// HDF5WriterBase provides a common interface for writing data/model to file.
// 
// 

// Change log:
// 
// 2012-02-25 14:39:36 (+0530) subha - started initial implementation
// 

// Code:

#ifdef USE_HDF5
#ifndef _HDF5IO_H
#define _HDF5IO_H
class HDF5WriterBase
{
  public:
    HDF5WriterBase();
    virtual ~HDF5WriterBase();
    void setFilename(string filename);
    string getFilename() const;
    bool isOpen() const;
    void setMode(unsigned int mode);
    unsigned int getMode() const;
    virtual void flush();
    
    static const Cinfo* initCinfo();
    
  protected:
    herr_t openFile();
    
    map <string, hid_t> nodemap_;
    hid_t filehandle_;
    string filename_;
    unsigned int openmode_;
};

#endif // _HDF5IO_H
#endif // USE_HDF5



// 
// HDF5WriterBase.h ends here
