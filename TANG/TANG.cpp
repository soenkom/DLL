#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>
#include <cstring>
#include <algorithm>

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

#include <LSTM_gen.cpp>
using namespace std;

constexpr int len = 200;
int npm, nhd;
constexpr int wds = 256 * 256 * 256 + 1;
int freq[wds];
int indx[wds];
int poem[3000][100];
int head[3000][10];

#define LoadData(name, arr) file.open(name, ios::in);\
    num = 0;\
    while (getline(file, line))\
    {\
        auto ws = wcv.from_bytes(line);\
        int wrd = 0;\
        for (auto&& chr : ws)\
        {\
            auto sht = static_cast<int>(chr);\
            arr[num][wrd] = sht;\
            ++wrd;\
        }\
        ++num;\
    }\
    file.close();

void Transform()
{
    for (int i = 0; i < npm; ++i)
    {
        int j = 0;
        while (poem[i][j])
        {
            ++freq[poem[i][j]];
            ++j;
        }
    }
    for (int i = 0; i < wds; ++i)
    {
        indx[i] = i;
    }
    sort(indx + 1, indx + wds, [](int a, int b){ return freq[a] > freq[b]; });
}

int Index(int t, int n)
{
    if (t == 0) return 0;
    if (freq[t] < freq[indx[n - 2]]) return n - 1;
    int low = 1, upp = n - 2;
    while (upp > low)
    {
        int mid = low + (upp - low) / 2;
        if (t == indx[mid]) return mid;
        if (freq[t] > freq[indx[mid]])
        {
            upp = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
    return low;
}

template <typename vector>
int Max(const vector& v)
{
    int m = 0;
    double d = v.dat[0];
    for (int i = 0; i < v.l; ++i)
    {
        if (v.dat[i] > d)
        {
            m = i;
            d = v.dat[i];
        }
    }
    return m;
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
    ifstream file;
    string line;
    wstring_convert<codecvt_utf8<wchar_t> > wcv;
    wstring_convert<codecvt_utf8<wchar_t> > wcs;
    memset(freq, 0, sizeof(freq));
    memset(indx, 0, sizeof(indx));
    memset(poem, 0, sizeof(poem));
    memset(head, 0, sizeof(head));
    vector dat(len);
    vector ans(len);
    dat.set(0.0);
    ans.set(0.0);
    srand((unsigned)time(NULL));
    LSTM_gen<matrix, vector> net(len);
    cout << "Loading data..." << endl;
    int num;
    LoadData("Poem.txt", poem);
    npm = num;
    LoadData("Head.txt", head);
    nhd = num;
    Transform();
    cout << "Start training..." << endl;
    int ite = 100;
    double eta = 0.001;
    int bat = 20;
    for (int k = 0; k < ite; ++k)
    {
        for (int i = 0; i < npm / bat; ++i)
        {
            for (int j = 0; j < bat; ++j)
            {
                int s = 0;
                for (int t = 0; t < 100; ++t)
                {
                    dat.dat[s] = 0.0;
                    dat.dat[Index(poem[i][t], len)] = 1.0;
                    ans.dat[Index(poem[i][t], len)] = 0.0;
                    ans.dat[Index(poem[i][t + 1], len)] = 1.0;
                    s = Index(poem[i][t], len);
                    net.train(dat, ans, eta);
                    ++t;
                }
                net.clear();
            }
            dat.set(0.0);
            ans.set(0.0);
            net.update();
            ++i;
        }
        cout << "Epoch " << k + 1 << ":" << endl;
        dat.set(0.0);
        for (int i = 0; i < nhd; ++i)
        {
            int t = 0;
            int s = 0;
            while (head[i][t + 1])
            {
                cout << wcs.to_bytes(static_cast<wchar_t>(head[i][t]));
                dat.dat[s] = 0.0;
                dat.dat[Index(head[i][t], len)] = 1.0;
                s = Index(head[i][t], len);
                net.forward(dat);
                ++t;
            }
            cout << wcs.to_bytes(static_cast<wchar_t>(head[i][t]));
            dat.dat[s] = 0.0;
            dat.dat[Index(head[i][t], len)] = 1.0; 
            ++t;
            while (t < 100)
            {
                auto&& vt = net.forward(dat);
                int nxt = Max(vt);
                dat.get(vt);
                if (nxt == 0)
                {
                    continue;
                }
                else if (nxt == len - 1)
                {
                    cout << wcs.to_bytes(static_cast<wchar_t>(indx[static_cast<int>(static_cast<double>(rand()) / RAND_MAX * 2000 + len)]));
                }
                else
                {
                    cout << wcs.to_bytes(static_cast<wchar_t>(indx[nxt]));
                }
                ++t;
            }
            net.clear();
            cout << endl;
        }
    }
    return 0;
}