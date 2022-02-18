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

#ifndef metropolis_class_h
#define metropolis_class_h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <vector>
#include <cstdint>

#include "random_class.h"

using namespace std;

class Metropolis
{
protected:
    Rand m_rnd;
    int m_L;
    int m_E, m_M;
    vector<int> m_E_history, m_M_history;
    float m_T;
    int8_t* m_spin;
    int m_history;

public:
    Metropolis(int L, float T);
    ~Metropolis();
    void set_seed(unsigned int seed);
    void set_temperature(float T);
    void initialize_spins();
    void set_magnetization(int M);
    void compute_and_set_magnetization();
    void set_energy(int E);
    void compute_and_set_energy();
    void one_step_evolution();
    int get_energy();
    int get_magnetization();
    void save_spin_lattice(ofstream& file, bool row=false, bool binary=false);
};

Metropolis::Metropolis(int L, float T)
{
    m_L = L;
    m_T = T;
    m_spin = new int8_t[m_L * m_L];
}

Metropolis::~Metropolis()
{
    delete m_spin;
}

inline void Metropolis::set_seed(unsigned int seed)
{
    m_rnd.set_Seed(seed);
}

void Metropolis::initialize_spins()
{
    for (int i = 0; i < m_L * m_L; i++)
    {
	m_spin[i] = 2 * m_rnd.nextInt(0, 1) - 1;
    }

    compute_and_set_magnetization();
    compute_and_set_energy();
    m_history = 0;
}

inline void Metropolis::set_temperature(float T)
{
    m_T = T;
}

inline void Metropolis::set_magnetization(int M)
{
    m_M = M;
}

void Metropolis::compute_and_set_magnetization()
{
    int M = 0;
    for (int i = 0; i < m_L * m_L; i++)
    {
	M += m_spin[i];
    }
    m_M = M;
    //m_M_history.push_back(M);
}

inline void Metropolis::set_energy(int E)
{
    m_E = E;
}

void Metropolis::compute_and_set_energy()
{
    int E = 0;

    for (int i = 0; i < m_L * m_L; i++)
    {
	int x = i / m_L;
	int y = i - x * m_L;
	int right_nbr = (y < m_L - 1) ? i+1 : i+1-m_L;
	int up_nbr = (x > 0) ? i-m_L : i-m_L+m_L*m_L;

	E -= m_spin[i] * (m_spin[right_nbr] + m_spin[up_nbr]);
    }
    m_E = E;
    //m_E_history.push_back(E);
}

void Metropolis::one_step_evolution()
{
    int E = m_E;
    int M = m_M;

    for (int k = 0; k < m_L * m_L; k++)
    {
	int i = m_rnd.nextInt(0, m_L * m_L -1);

	int x = i / m_L;
	int y = i - x * m_L;
	int right_nbr = (y < m_L-1) ? i+1 : i+1-m_L;
	int up_nbr = (x > 0) ? i-m_L : i-m_L+m_L*m_L;
	int left_nbr = (y > 0) ? i-1 : i-1+m_L;
	int down_nbr = (x < m_L-1) ? i+m_L : i+m_L-m_L*m_L;

	int total = m_spin[right_nbr] + m_spin[up_nbr] + m_spin[left_nbr] + m_spin[down_nbr];

	int dE = 2 * m_spin[i] * total;

	if (dE <= 0 || exp(-dE / m_T) > m_rnd.nextDouble())
	{
	    M -= m_spin[i];
	    E += dE;
	    m_spin[i] *= -1;
	    M += m_spin[i];
	}
    }

    m_E = E;
    m_M = M;
    m_history += 1;

    //m_E_history.push_back(E);
    //m_M_history.push_back(M);
}

inline int Metropolis::get_energy()
{
    return m_E;
}

inline int Metropolis::get_magnetization()
{
    return m_M;
}

void Metropolis::save_spin_lattice(ofstream& file, bool row, bool binary)
{
    if (binary)
    {
	file.write((char*)m_spin, m_L*m_L*sizeof(int8_t));
    }

    else
    {
	if (row)
	{
	    for (int i = 0; i < m_L * m_L; i++)
	    {
		file << ((int)m_spin[i] + 1) / 2;
	    }
	    file << "\n";
	}

	else
	{
	    for (int i = 0; i < m_L * m_L; i++)
	    {
		file << ((int)m_spin[i] + 1) / 2;
		if ((i+1) % m_L == 0) file << "\n";
		if (i == m_L * m_L - 1) file << "\n";
	    }
	}
    }
}

#endif /* metropolis_class_h */
