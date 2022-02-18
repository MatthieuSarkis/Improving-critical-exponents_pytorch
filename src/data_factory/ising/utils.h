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

#ifndef utils_h
#define utils_h

#include <cmath>

bool isSame(float x, float y)
{
    return fabs(x - y) < 10e-6;
}

#endif /* utils_h */