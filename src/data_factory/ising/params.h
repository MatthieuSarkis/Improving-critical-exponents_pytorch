// -*- coding: utf-8 -*-
//
// Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef params_h
#define params_h

#include <math.h>

float T_critical(int L)
{
    float Tc = 2 / log(1 + sqrt(2));
    return Tc / (1 + 5 / (4.0 * L));
}

int L = 64;
int n_steps_initial = 1000;
int n_steps_thermalize = 100;
int n_steps_generation = 10;
int n_data_per_temp = 50000;
float T_min = 1.8;
float T_max = 3.0;
float dT = 0.010;
bool binary = true;

//float Tc = T_critical(L);
float Tc = 2 / log(1 + sqrt(2));

#endif /* params_h */
