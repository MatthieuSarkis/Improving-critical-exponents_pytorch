// -*- coding: utf-8 -*-
//
// Written by Hor Dashti(Ebi), https://github.com/h-dashti
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef PROGRESS_H_
#define PROGRESS_H_

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>

class Progress{
    
protected:
    int total_iterations;
    int curr;
    int prev;
    long long int start_time;
    
public:
    Progress();
    Progress(const int total_iters);
    void Assign(const int total_iters);
    void ReAssign(const int total_iters);
    void Reset();
    void Reset(const int total_iters);
    void Next(const int idx, std::ostream& stm = std::cout);
    void PrintTimePercentage(const int smp, std::ostream& stm);
};


Progress::Progress()
{}

Progress::Progress(const int total_iters):
    total_iterations(total_iters),
    curr(0),
    prev(-1),
    start_time(clock())
{}
    
void Progress::Assign(const int total_iters)
{
    Reset(total_iters);
}

void Progress::ReAssign(const int total_iters)
{
    total_iterations = total_iters;
}

void Progress::Reset()
{
    curr = 0;
    prev = -1;
    start_time = clock();
}

void Progress::Reset(const int total_iters)
{
    total_iterations = total_iters;
    curr = 0;
    prev = -1;
    start_time = clock();
}

void Progress::Next(const int idx, std::ostream& stm)
{
    curr = (int)((idx + 1.0) * (1.0 / total_iterations) * 100);
    if (curr != prev)
    {
        PrintTimePercentage(idx, stm);
        if (curr >= 100)
        {
            stm << std::endl;
        }
    }
}

void Progress::PrintTimePercentage(const int smp, std::ostream& stm)
{
    stm << "\r";
    stm << curr << "%";
    stm.flush();
    prev = curr;
}



#endif 
