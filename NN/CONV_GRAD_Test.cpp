#include <iostream>
#include <list>
#include <ctime>
#include <cstdlib>
#include "CONV_net.cpp"

using namespace std;
#define INTRIN
#ifndef INTRIN
#include "SimpleArray.cpp"
using matrix = SimpleMatrix;
using vector = SimpleVector;
void Print(const SimpleMatrix& m)
{
    int t = 0;
    for (int i = 0; i < m.r; ++i)
    {
        for (int j = 0; j < m.c; ++j)
        {
            cout << m.dat[t++] << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}
#else
#include "IntrinArray.cpp"
using matrix = IntrinMatrix;
using vector = IntrinVector;
void Print(const IntrinMatrix& m)
{
    int t = 0;
    for (int i = 0; i < m.r; ++i)
    {
        for (int j = 0; j < upper(m.c); ++j)
        { 
            cout << m.dat[t++] << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}
#endif

#define GRAD(ind) dat.front().ini();\
                ans.set(0.0);\
                ans.dat[rand() % 10] = 1.0;\
                double mem = ind;\
                ind += dif;\
                auto&& err = net.forward(dat);\
                err.map(Log);\
                auto Ep = -Inner(err, ans);\
                ind -= 2 * dif;\
                auto&& ers = net.forward(dat);\
                ers.map(Log);\
                auto En = -Inner(ers, ans);\
                ind += dif;\
                net.train(dat, ans, 1.0);\
                net.update();\
                double de = (Ep - En) / 2 / dif;\
                double da = mem - ind;\
                if (abs(de - da) > 1e-9)\
                    cout << "Expected gradient " << de << " Actual gradient " << da << endl;

int LINE(int n)
{
#ifndef INTRIN
    return n;
#else
    return upper(n);
#endif
}

double Sum(vector& v)
{
    double ret = 0.0;
    for (int i = 0; i < v.l; ++i)
    {
        ret += v.dat[i] * v.dat[i];
    }
    return ret;
}

int main()
{
    srand((unsigned)time(NULL));
    list<int> k{9, 9, 1, 2, 9, 9, 2, 3};
    list<int> f{12 * 12 * 3, 30, 10};
    CONV_net<matrix, vector> net(k, f);
    list<matrix> dat;
    vector ans(10);
    dat.emplace_back(28, 28);
    dat.front().ini();
    ans.ini();
    double dif = 1e-4;
    
    cout << "Convolution Layer kernel Test!" << endl;
    for (auto&& kn : net.K)
    {
        for (int ii = 0; ii < kn.n; ++ii)
        {
            for (auto&& m : kn.knl[ii])
            {
                for (int i = 0; i < m.r; ++i)
                {
                    for (int j = 0; j < m.c; ++j)
                    {
                        GRAD(m.dat[i * LINE(m.c) + j]);
                    }
                }
            }
        }
    }
    cout << "Full connection Layer weight Test!" << endl;
    for (auto&& m : net.w)
    {
        for (int i = 0; i < m.r; ++i)
        {
            for (int j = 0; j < m.c; ++j)
            {
                GRAD(m.dat[i * LINE(m.c) + j]);
            }
        }
    }
    cout << "Full connection Layer biase Test!" << endl;
    for (auto&& v : net.b)
    {
        for (int i = 0; i < v.l; ++i)
        {
            GRAD(v.dat[i]);
        }
    }
    return 0;
}