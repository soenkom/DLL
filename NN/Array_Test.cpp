#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

#include "SimpleArray.cpp"
#include "IntrinArray.cpp"

#define EPS 1e-9
#define NEQ(a, b) fabs(a - b) > EPS
#define RND(n) n = (double) rand() / RAND_MAX + 1.0

#define INI(a, b, m, n) do {\
        a.set(0.0);\
        b.set(0.0);\
        for (int i = 0; i < m; ++i)\
        {\
            for (int j = 0; j < n; ++j)\
            {\
                t = (double)rand() / RAND_MAX;\
                a.dat[i * n + j] = t;\
                b.dat[i * upper(n) + j] = t;\
            }\
        }\
    } while(0)

#define TIME(S_fun, I_fun, s, a, b) do {\
        now = clock();\
        S_fun;\
        cout << "Time used (simple " s "): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;\
        now = clock();\
        I_fun;\
        cout << "Time used (intrin " s "): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;\
        if (!Test(a.dat, b.dat, a.r, a.c))\
        {\
            cout << "FAIL " s << endl;\
            goto END;\
        }\
    } while(0)

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

bool Test(double *a, double *b, int m, int n)
{
    bool ret = true;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (NEQ(a[i * n + j], b[i * upper(n) + j]))
            {
                ret = false;
                break;
            }
        }
        if (!ret) break;
    }
    return ret;
}

int main()
{
    time_t now;
    int r, c, h, kr, kc;
    cout << "Input rows, cols, hights and kernel size: " << endl;
    cin >> r >> c >> h >> kr >> kc;
    SimpleMatrix sa(r, c), sb(r, c), sc(c, h), sk(kr, kc);
    SimpleVector su(r), sv(c), sw(r);
    IntrinMatrix ia(r, c), ib(r, c), ic(c, h), ik(kr, kc);
    IntrinVector iu(r), iv(c), iw(r);
    int cnt = 0;
    auto sigmoid = [] (const double d) { return 1.0 / (1.0 + exp(-d)); };
    do {
        srand(clock());
        double t;
        INI(sa, ia, r, c);
        INI(sb, ib, r, c);
        INI(sc, ic, c, h);
        INI(su, iu, 1, r);
        INI(sw, iw, 1, r);
        INI(sv, iv, 1, c);
        INI(sk, ik, kr, kc);

        RND(t);
        TIME(su.set(t),  iu.set(t),  "V_SET", su, iu);
        TIME(su.get(sw), iu.get(iw), "V_GET", su, iu);

        RND(t);
        TIME(su.add(t), iu.add(t), "V_ADD", su, iu);
        RND(t);
        TIME(su.sub(t), iu.sub(t), "V_SUB", su, iu);
        RND(t);
        TIME(su.mul(t), iu.mul(t), "V_MUL", su, iu);
        RND(t);
        TIME(su.div(t), iu.div(t), "V_DIV", su, iu);

        TIME(su.add(sw), iu.add(iw), "V_ADD_V", su, iu);
        TIME(su.sub(sw), iu.sub(iw), "V_SUB_V", su, iu);
        TIME(su.mul(sw), iu.mul(iw), "V_MUL_V", su, iu);
        TIME(su.div(sw), iu.div(iw), "V_DIV_V", su, iu);
        TIME(sv.map(sigmoid), iv.map(sigmoid), "V_SIGMOID", su, iu);

        RND(t);
        TIME(sa.set(t),  ia.set(t),  "M_SET", sa, ia);
        TIME(sa.get(sb), ia.get(ib), "M_GET", sa, ia);

        RND(t);
        TIME(sa.add(t), ia.add(t), "M_ADD", sa, ia);
        RND(t);
        TIME(sa.sub(t), ia.sub(t), "M_SUB", sa, ia);
        RND(t);
        TIME(sa.mul(t), ia.mul(t), "M_MUL", sa, ia);
        RND(t);
        TIME(sa.div(t), ia.div(t), "M_DIV", sa, ia);

        TIME(sa.add(sb), ia.add(ib), "M_ADD_M", sa, ia);
        TIME(sa.sub(sb), ia.sub(ib), "M_SUB_M", sa, ia);
        TIME(sa.mul(sb), ia.mul(ib), "M_MUL_M", sa, ia);
        TIME(sa.div(sb), ia.div(ib), "M_DIV_M", sa, ia);
        TIME(sb.map(sigmoid), ib.map(sigmoid), "V_SIGMOID", sb, ib);

        if (NEQ(Inner(su, sw), Inner(iu, iw)))
        {
            cout << "Fail u . v" << endl;
            break;
        }
        cout << "Inner Finished!" << endl;
        
        if (!Test(Outer(su, sv).dat, Outer(iu, iv).dat, r, c))
        {
            cout << "Fail u x v" << endl;
            break;
        }
        cout << "Outer Finished!" << endl;

        if (!Test(Times(sa, sv).dat, Times(ia, iv).dat, 1, r))
        {
            cout << "Fail Av" << endl;
            break;
        }
        cout << "Av Finished!" << endl;

        if (!Test(Times(su, sa).dat, Times(iu, ia).dat, 1, c))
        {
            cout << "Fail vA" << endl;
            break;
        }
        cout << "vA Finished!" << endl;

        now = clock();
        auto&& sm = Times(sa, sc);
        cout << "Time used (simple TIMES): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        now = clock();
        auto&& im = Times(ia, ic);
        cout << "Time used (intrin TIMES): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        if (!Test(sm.dat, im.dat, r, h))
        {
            cout << "Fail AB" << endl;
            break;
        }

        now = clock();
        auto&& so = Conv(sa, sk);
        cout << "Time used (simple CONV): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        now = clock();
        auto&& io = Conv(ia, ik);
        cout << "Time used (intrin CONV): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        if (!Test(so.dat, io.dat, so.r, so.c))
        {
            cout << "Fail CONV" << endl;
            break;
        }

        now = clock();
        ConvTo(sa, sk, so);
        cout << "Time used (simple CONV_TO): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        now = clock();
        ConvTo(ia, ik, io);
        cout << "Time used (intrin CONV_TO): " << (double)(clock() - now) / CLOCKS_PER_SEC << " s" << endl;
        if (!Test(so.dat, io.dat, so.r, so.c))
        {
            cout << "Fail CONV" << endl;
            break;
        }

        cout << "All tests passed for " << ++cnt << " times!\n" << endl;

    } while(cnt < 1024);
END:
    return 0;
}