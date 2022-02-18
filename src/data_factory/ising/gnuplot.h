#ifndef GNUPLOT_H
#define GNUPLOT_H
#include <iostream>
#include <string>
using namespace std;

class gnuplot
{
    public:
        gnuplot();
        ~gnuplot();
        void operator () (const string & command);

    protected:
        FILE *gnuplotpipe;
};

gnuplot::gnuplot()
{
    gnuplotpipe = popen("/usr/local/bin/gnuplot -persist","w");
    if (!gnuplotpipe)
    cerr<<("Gnuplot not found!");
}

gnuplot::~gnuplot()
{
    fprintf(gnuplotpipe,"exit\n");
    pclose(gnuplotpipe);
}
void gnuplot::operator() (const string & command)
{
    fprintf(gnuplotpipe,"%s\n", command.c_str());
    fflush(gnuplotpipe);
}
#endif
