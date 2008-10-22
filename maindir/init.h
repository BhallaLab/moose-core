/*******************************************************************
 * File:            init.h
 * Description:     Functions to do initialization for moose
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 14:52:47
 ********************************************************************/

#ifndef _INIT_H
#define _INIT_H

void initMoose();
void initSched();
void setupDefaultSchedule(Element* t0, Element* t1, Element* cj);

#endif
