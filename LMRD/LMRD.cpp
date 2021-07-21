#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <list>
#include <map>

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
#include <LSTM_sen.cpp>
using namespace std;

#define LoadData(str, train, test) file.open(str, ios::in);\
    for (int i = 0; i < num_train; ++i)\
    {\
        Convert(file, train[i]);\
    }\
    for (int i = 0; i < num_test; ++i)\
    {\
        Convert(file, test[i]);\
    }\
    file.close();

constexpr int num_train = 1000;
constexpr int num_test = 250;
constexpr int len = 300;
constexpr int siz = 100;
int pos_test[num_test][len], neg_test[num_test][len];
int pos_train[num_train][len], neg_train[num_train][len];
map<string, int> dict;
int words[100000];
int freq[100000];
int num_words = 0;

void Convert(ifstream& s, int* a)
{
    string str;
    int ind = 0;
    while (s >> str)
    {
        if (str == "ENDOFLINE") break;
        auto it = dict.find(str);
        int num;
        if (it == dict.end())
        {
            num = dict.size();
            dict.insert(pair<string, int>(str, num));
            words[num_words] = num;
            ++num_words;
        }
        else
        {
            num = it->second;
        }
        freq[num] += 1;
        a[ind] = num;
        ++ind;
        if (ind > len) break;
    }
}

int Index(int t, int n)
{
    if (t == 0) return 0;
    if (freq[t] < freq[words[n - 2]]) return n - 1;
    int low = 0, upp = n - 2;
    while (upp > low)
    {
        int mid = low + (upp - low) / 2;
        if (t == words[mid]) return mid;
        if (freq[t] > freq[words[mid]])
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

int main()
{
    srand((unsigned)time(NULL));
    list<int> l{siz, 10, 2};
    LSTM_sen<matrix, vector> net(l);
    ifstream file;
    memset(pos_train, 0, sizeof(pos_train));
    memset(neg_train, 0, sizeof(neg_train));
    memset(pos_test, 0, sizeof(pos_test));
    memset(neg_test, 0, sizeof(neg_test));
    memset(freq, 0, sizeof(freq));
    dict.insert(pair<string, int>(" ", 0));
    cout << "Loading data..." << endl;
    LoadData("pos", pos_train, pos_test);
    LoadData("neg", neg_train, neg_test);
    sort(words, words + num_words, [](int a, int b){ return freq[a] > freq[b]; });
    cout << "Start training" << endl;
    vector dat(siz), ans(2);
    dat.set(0.0);
    ans.set(0.0);
    int ite = 100;
    int bat = 20;
    int eta = 0.001;
    for (int k = 0; k < ite; ++k)
    {
        int t = 0;
        for (int ii = 0; ii < num_train / bat; ++ii)
        {
            for (int jj = 0; jj < bat; ++jj)
            {
                ans.set(0.0);
                ans.dat[0] = 1.0;
                for (int i = 0; i < len; ++i)
                {
                    dat.set(0.0);
                    dat.dat[Index(pos_train[t][i], siz)] = 1.0;
                    net.forward(dat);
                }
                net.train(ans, eta);
                net.clear();
                ans.set(0.0);
                ans.dat[1] = 1.0;
                for (int i = 0; i < len; ++i)
                {
                    dat.set(0.0);
                    dat.dat[Index(neg_train[t][i], siz)] = 1.0;
                    net.forward(dat);
                }
                net.train(ans, eta);
                net.clear();
            }
            net.update();
        }
        int tot = 0;
        for (int i = 0; i < num_test; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                dat.set(0.0);
                dat.dat[Index(pos_test[i][j], siz)] = 1.0;
                net.forward(dat);
            }
            auto&& vec = net.predict();
            tot += (vec.dat[0] > vec.dat[1]);
            net.clear();
            for (int j = 0; j < len; ++j)
            {
                dat.set(0.0);
                dat.dat[Index(neg_test[i][j], siz)] = 1.0;
                net.forward(dat);
            }
            auto&& vee = net.predict();
            tot += (vee.dat[0] < vee.dat[1]);
            net.clear();
        }
        cout << "Epoch " << k + 1 << " : " << tot << " / " << num_test * 2 << " recognized!" << endl;\
    }
    return 0;
}