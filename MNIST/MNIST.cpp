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

int row, col, els, pix, len;
unsigned char byte[4];
char labels[100000];

#define READ(var) file.read((char*)byte, 4 * sizeof(char)); var = EndianRev()
#define IMG_VARS READ(els); READ(els); READ(row); READ(col)
#define LAB_VARS READ(els); READ(els);
#define EPOCH(ite, bat, eta) cout << "Training with ETA = " << eta << ", batch size = " << bat << " for " << ite << " times!" << endl;\
    for (int k = 0; k < ite; ++k)\
    {\
        dat.front().dat = img;\
        ans.dat = lab;\
        for (int i = 0; i < trs / bat; ++i)\
        {\
            for (int j = 0; j < bat; ++j)\
            {\
                net.train(dat, ans, eta);\
                dat.front().dat += pix;\
                ans.dat += len;\
            }\
            net.update();\
        }\
        dat.front().dat = imgt;\
        int tot = 0;\
        for (int i = 0; i < tes; ++i)\
        {\
            auto&& s = net.forward(dat);\
            tot += Test(s.dat, i);\
            dat.front().dat += pix;\
        }\
        cout << "Epoch " << k + 1 << " : " << tot << " / " << tes << " recognized!" << endl;\
    }

inline int EndianRev()
{
    return static_cast<int>(byte[0] << 24) + (static_cast<int>(byte[1]) << 16) + (static_cast<int>(byte[2]) << 8) + static_cast<int>(byte[3]);
}

void LoadData(const string& s, double* &p, double* &q)
{
    ifstream file;
    char chr;
    file.open(s + "-images.idx3-ubyte", ios::in | ios::binary);
    IMG_VARS;
    double *t;
#ifndef INTRIN
    pix = row * col;
    p = (double*)malloc(pix * els * sizeof(double));
    t = p;
    for (int i = 0; i < pix * els; ++i)
    {
        file.read(&chr, sizeof(char));
        *t++ = static_cast<double>(chr) / 255.0;
    }
#else
    pix = row * upper(col);
    p = (double*)_mm_malloc(pix * els * sizeof(double), 32);
    t = p;
    for (int i = 0; i < els; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            for (int k = 0; k < col; ++k)
            {
                file.read(&chr, sizeof(char));
                *t++ = static_cast<double>(chr) / 255.0;
            }
            for (int k = col; k < upper(col); ++k)
            {
                *t++ = 0.0;
            }
        }
    }
#endif
    file.close();
    file.open(s + "-labels.idx1-ubyte", ios::in | ios::binary);
    LAB_VARS;
#ifndef INTRIN
    len = 10;
    q = (double*)malloc(10 * els * sizeof(double));
    t = q;
    for (int i = 0; i < els; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            *t++ = 0.0;
        }
        file.read(&chr, sizeof(char));
        labels[i] = chr;
        t[chr - 10] = 1.0;
    }
#else
    len = upper(10);
    q = (double*)_mm_malloc(upper(10) * els * sizeof(double), 32);
    t = q;
    for (int i = 0; i < els; ++i)
    {
        for (int j = 0; j < upper(10); ++j)
        {
            *t++ = 0.0;
        }
        file.read(&chr, sizeof(char));
        labels[i] = chr;
        t[chr - upper(10)] = 1.0;
    }
#endif
    file.close();
}

void Free(void *p)
{
#ifndef INTRIN
    free(p);
#else
    _mm_free(p);
#endif
}

inline int Test(double *p, int n)
{
    int ret = 1;
    double m = p[static_cast<int>(labels[n])];
    for (int i = 0; i < 10; ++i)
    {
        if (p[i] > m)
        {
            ret = 0;
            break;
        }
    }
    return ret;
}

int main()
{
    double* img = nullptr;
    double* lab = nullptr;
    double* imgt = nullptr;
    double* labt = nullptr;
    int trs;
    int tes;
    list<matrix> dat;
    vector ans(10);
    cout << "Loading data..." << endl;
    LoadData("train", img, lab);
    trs = els;
    LoadData("t10k", imgt, labt);
    tes = els;
    srand((unsigned)time(NULL));
    list<int> k{5, 5, 1, 5, 5, 5, 5, 10};
    list<int> f{(row - 8) * (col - 8) * 10, 120, 10};
    CONV_net<matrix, vector> net(k, f);
    dat.emplace_back(row, col);
    Free(dat.front().dat);
    Free(ans.dat);
    cout << "Start training..." << endl;
    EPOCH(100, 250, 0.001);
    dat.front().dat = nullptr;
    ans.dat = nullptr;
    Free(img);
    Free(lab);
    Free(imgt);
    Free(labt);
    return 0;
}