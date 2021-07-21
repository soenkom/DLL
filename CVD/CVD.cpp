#include <iostream>
#include <fstream>
#include <list>
#include <ctime>
#include <CONV_net.cpp>
using namespace std;

#define INTRIN
#ifndef INTRIN
#include "SimpleArray.cpp"
using matrix = SimpleMatrix;
using vector = SimpleVector;
#else
#include "IntrinArray.cpp"
using matrix = IntrinMatrix;
using vector = IntrinVector;
#endif
#define EPOCH(ite, bat, eta) cout << "Training with ETA = " << eta << ", batch size = " << bat << " for " << ite << " times!" << endl;\
    for (int k = 0; k < ite; ++k)\
    {\
        auto pct = cat;\
        auto pdg = dog;\
        for (int i = 0; i < trains * 2 / bat; ++i)\
        {\
            ans.dat[0] = 1.0;\
            ans.dat[1] = 0.0;\
            for (int j = 0; j < bat / 2; ++j)\
            {\
                for (auto&& m : dat)\
                {\
                    m.dat = pct;\
                    pct += pix;\
                }\
                net.train(dat, ans, eta / bat);\
            }\
            ans.dat[0] = 0.0;\
            ans.dat[1] = 1.0;\
            for (int j = 0; j < bat / 2; ++j)\
            {\
                for (auto&& m : dat)\
                {\
                    m.dat = pdg;\
                    pdg += pix;\
                }\
                net.train(dat, ans, eta / bat);\
            }\
            net.update();\
        }\
        pct = catt;\
        pdg = dogt;\
        int tot = 0;\
        for (int i = 0; i < tests; ++i)\
        {\
            for (auto&& m : dat)\
            {\
                m.dat = pct;\
                pct += pix;\
            }\
            auto&& s = net.forward(dat);\
            tot += (s.dat[0] > s.dat[1]);\
            cout << s.dat[0] << "  " << s.dat[1] << endl;\
        }\
        for (int i = 0; i < tests; ++i)\
        {\
            for (auto&& m : dat)\
            {\
                m.dat = pdg;\
                pdg += pix;\
            }\
            auto&& s = net.forward(dat);\
            tot += (s.dat[0] < s.dat[1]);\
            cout << s.dat[0] << "  " << s.dat[1] << endl;\
        }\
        cout << "Epoch " << k + 1 << " : " << tot << " / " << tests * 2 << " recognized!" << endl;\
    }

int row = 80;
int col = 80;
int pix;

void LoadData(const string& s, double *&p, const int& n)
{
    ifstream file;
    int clr;
    int l;
#ifndef INTRIN
    l = row;
#else
    l = upper(row);
#endif
    p = new double[3 * pix * n];
    auto ptr = p;
    for (int k = 0; k < n; ++k)
    {
        file.open(s + to_string(k) + ".txt", ios::in);
        for (int t = 0; t < 3; ++t)
        {
            for (int i = 0; i < row; ++i)
            {
                auto ptt = ptr;
                for (int j = 0; j < col; ++j)
                {
                    file >> clr;
                    *(ptt++) = static_cast<double>(clr) / 255.0;
                }
                ptr += l;
            }
        }
        file.close();
    }
}

void Free(void *p)
{
#ifndef INTRIN
    free(p);
#else
    _mm_free(p);
#endif
}
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
int main()
{
#ifndef INTRIN
    pix = row * col;
#else
    pix = row * upper(col);
#endif
    double* cat = nullptr;
    double* dog = nullptr;
    double* catt = nullptr;
    double* dogt = nullptr;
    list<matrix> dat;
    vector ans(2);
    cout << "Loading data..." << endl;
    LoadData("Train/Cat/", cat, 10000);
    LoadData("Train/Dog/", dog, 10000);
    LoadData("Test/Cat/", catt, 2500);
    LoadData("Test/Dog/", dogt, 2500);
    srand((unsigned)time(NULL));
    list<int> k{5, 5, 3, 32, 5, 5, 32, 64, 5, 5, 64, 64};
    list<int> f{(row - 12) * (col - 12) * 64, 128, 2};
    CONV_net<matrix, vector> net(k, f);
    for (int i = 0; i < 3; ++i)
    {
        dat.emplace_back(row, col);
        Free(dat.back().dat);
    }
    cout << "Start training..." << endl;
    int trains = 1000;
    int tests = 250;
    EPOCH(100, 25, 0.001);
    for (auto&& m : dat)
    {
        m.dat = nullptr;
    }
    Free(cat);
    Free(dog);
    Free(catt);
    Free(dogt);
    return 0;
}