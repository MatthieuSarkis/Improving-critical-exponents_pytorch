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

#ifndef write_plot_h
#define write_plot_h

#include "gnuplot.h"
#include <sstream>
#include <fstream>
#include <cstdint>
#include <vector>

/**
 Write a file in the format:
 0 E0
 1 E1
 2 E2
 ...
 */
string write_energy_to_file(vector<int> E_list, int L, float T, int n_steps)
{
    string fichier;
    stringstream ss;
    ss << "L_" << L << "_temp_" << T << "_nsteps_" << n_steps;
    fichier = ss.str();     
    fstream evol;
    
    evol.open(("./data/" + fichier + ".dat").c_str(), ios::out);
                                                                                        
    for (int i = 0; i < E_list.size(); ++i)
    {
            evol << i << " ";
            evol << E_list[i] << " ";
            evol << endl;
    }
    evol << endl;
    
    evol.close();
    
    return fichier;
}

void plot_energy(string fichier)
{
    gnuplot p;
    p("set term postscript eps");
    p("set output \"./data/" + fichier + ".eps\" ");
    p("plot \'./data/" + fichier + ".dat\'");
}

/**
 This function draws a given lattice configuration into a text file using 0s and 1s for readaility.
 */
void write_lattice_to_file(int8_t** lattice, int L, float T)
{
    string fichier;
    stringstream ss;
    ss << "L_" << L << "_temp_" << T;
    fichier = ss.str();
    fstream lat;
    
    lat.open(("./data/" + fichier + ".txt").c_str(), ios::out);
                                                                                        
    for (int j = 0; j < L; j++)
    {
        for (int k = 0; k < L; k++) lat << (lattice[j][k]+1)/2;
        lat << endl;
    }
    lat << endl;
    
    lat.close();
}

void write_for_python(int8_t** lattice, int L, bool endline)
{
    string fichier;
    stringstream ss;
    ss << "L_" << L;
    fichier = ss.str();
    fstream lat;
    
    lat.open(("./data/" + fichier + ".csv").c_str(), ios::app);
                                                                                        
    for (int j = 0; j < L; j++)
    {
        for (int k = 0; k < L; k++) lat << (int)lattice[j][k] << ",";
    }
    if (!endline) lat << endl;
    
    lat.close();
}

#endif /* write_plot_h */
