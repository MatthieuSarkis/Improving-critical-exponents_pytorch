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

#ifndef RANDOMCLASS_H_
#define RANDOMCLASS_H_

#include <iostream>
#include <random>
#include <ctime>

class Rand
{
protected:
	std::mt19937 m_rgen;

public:

	Rand();
	Rand(unsigned int seed);

	void set_Seed(unsigned int seed);
	void set_Seed();
	double nextDouble();
        double nextNormal(double = 0, double = 1);
	int nextInt(int first, int last);
	int nextInt(int last);

	std::mt19937& get_Rgen();
};

Rand::Rand()
{
	set_Seed();
}

Rand::Rand(unsigned int seed)
{
	set_Seed(seed);
}

inline void Rand::set_Seed(unsigned int seed)
{
	m_rgen.seed(seed);
}

inline void Rand::set_Seed()
{
#ifdef _WIN32
    m_rgen.seed((unsigned int)time(nullptr));
#else
    m_rgen.seed(std::random_device{}());
#endif
}

inline double Rand::nextDouble()
{
	return std::uniform_real_distribution<>(0, 1.0)(m_rgen);
}

inline double Rand::nextNormal(double mean, double std)
{
	return std::normal_distribution<>(mean, std)(m_rgen);
}

inline int Rand::nextInt(int first, int last)
{
	return std::uniform_int_distribution<>(first, last)(m_rgen);
}

inline int Rand::nextInt(int last)
{
	return    std::uniform_int_distribution<>(0, last)(m_rgen);
}

inline std::mt19937& Rand::get_Rgen()
{
	return m_rgen;
}

#endif
