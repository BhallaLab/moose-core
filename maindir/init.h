/*******************************************************************
 * File:            init.h
 * Description:     Functions to do initialization for moose
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 14:52:47
 ********************************************************************/

#ifndef _INIT_H
#define _INIT_H
#include <iostream>
#include "../basecode/header.h"
#include "../basecode/moose.h"
#include "../element/Neutral.h"
#include "../basecode/IdManager.h"

int mooseInit();

void setupDefaultSchedule(Element* t0, Element* t1, Element* cj);


#endif
