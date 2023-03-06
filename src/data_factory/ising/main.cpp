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

#include <cstdint>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <ctime>
#include <cmath>
#include <chrono>
#include <filesystem>

#include "metropolis_class.h"
#include "progress.h"
#include "params.h"
#include "utils.h"

using namespace std;
namespace fs = std::filesystem;

int main()
{
    string dir_name;
    stringstream ss;
    ss.setf(ios::fixed);
    ss << "L=" << L;
    dir_name = "./data/ising/" + ss.str();
    fs::create_directory("./data");
    fs::create_directory("./data/ising");
    fs::create_directory(dir_name);

    unsigned seed = static_cast<unsigned int>(chrono::steady_clock::now().time_since_epoch().count());

    vector<float> T;
    //for (int i = 0; i <= (int)((T_max - T_min) / dT); i++)
    //{
    //    T.push_back(T_min + i * dT);
    //}
    T.push_back(Tc);
    sort(T.begin(), T.end(), greater<float>());

    Metropolis model(L, 3.0);
    model.set_seed(seed);
    model.initialize_spins();
    int n_steps = 0;

    Progress progress((int)T.size());

    for (int i = 0; i < T.size(); i++)
    {
        string file_name;
        stringstream ss;
        ss.setf(ios::fixed);
        ss << setprecision(4);
        ss << "T=" << T[i];
        file_name = ss.str();

        ofstream file((dir_name + "/" + file_name + ".bin").c_str(), ios::out | ios::binary);

        n_steps = (i == 0 || isSame(fabs(T[i] - Tc), dT)) ? n_steps_initial : n_steps_thermalize;

        for (int j = 0; j < n_steps; j++)
        {
            model.one_step_evolution();
        }

        for (int j = 0; j < n_data_per_temp; j++)
        {
            for (int k = 0; k < n_steps_generation; k++)
            {
                model.one_step_evolution();
            }

            if (binary)
            {
                model.save_spin_lattice(file, true, true);
            }
            else
            {
                model.save_spin_lattice(file, false, false);
            }
        }

        file.close();

        progress.Next(i);
    }

    return 0;
}