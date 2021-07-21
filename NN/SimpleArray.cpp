#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
using std::list;

class SimpleMatrix
{
public:
    int r, c, l, s;
    double* dat;
    SimpleMatrix();
    SimpleMatrix(const int m, const int n);
    ~SimpleMatrix();
    inline void ini() const;
    inline void set(const double d) const;
    inline void get(const SimpleMatrix& a) const;
    inline void add(const double d) const;
    inline void sub(const double d) const;
    inline void mul(const double d) const;
    inline void div(const double d) const;
    inline void add(const SimpleMatrix& a) const;
    inline void sub(const SimpleMatrix& a) const;
    inline void mul(const SimpleMatrix& a) const;
    inline void div(const SimpleMatrix& a) const;
    inline void map(double (*f)(const double d)) const;
    inline void transpose() const;
    inline double sum() const;
};

class SimpleVector : public SimpleMatrix
{
public:
    SimpleVector();
    SimpleVector(const int n);
    ~SimpleVector();
};

inline double       Inner(const SimpleVector& u, const SimpleVector& v);
inline SimpleMatrix Outer(const SimpleVector& u, const SimpleVector& v);
inline SimpleVector Times(const SimpleMatrix& m, const SimpleVector& v);
inline SimpleVector Times(const SimpleVector& v, const SimpleMatrix& m);
inline SimpleMatrix Times(const SimpleMatrix& m, const SimpleMatrix& n);
inline SimpleMatrix  Conv(const SimpleMatrix& I, const SimpleMatrix& K);
inline void        ConvTo(const SimpleMatrix& I, const SimpleMatrix& K, SimpleMatrix& S);
inline double* Copy(double *p, const SimpleMatrix& m);

SimpleMatrix::SimpleMatrix()
{
    this->r = 0;
    this->c = 0;
    this->l = 0;
    this->s = 0;
    this->dat = nullptr;
}

SimpleMatrix::SimpleMatrix(const int m, const int n)
{
    this->r = m;
    this->c = n;
    this->l = m * n;
    this->s = m * n | 1;
    this->dat = (double*)malloc(this->s * sizeof(double));
}

SimpleMatrix::~SimpleMatrix()
{
    free(this->dat);
}

inline void SimpleMatrix::ini() const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }
}

inline void SimpleMatrix::set(const double d) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] = d;
    }
}

inline void SimpleMatrix::get(const SimpleMatrix& m) const
{
    memcpy(this->dat, m.dat, m.s * sizeof(double));
}

inline void SimpleMatrix::add(const double d) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] += d;
    }
}

inline void SimpleMatrix::sub(const double d) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] -= d;
    }
}

inline void SimpleMatrix::mul(const double d) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] *= d;
    }
}

inline void SimpleMatrix::div(const double d) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] /= d;
    }
}

inline void SimpleMatrix::add(const SimpleMatrix& a) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] += a.dat[i];
    }
}

inline void SimpleMatrix::sub(const SimpleMatrix& a) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] -= a.dat[i];
    }
}

inline void SimpleMatrix::mul(const SimpleMatrix& a) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] *= a.dat[i];
    }
}

inline void SimpleMatrix::div(const SimpleMatrix& a) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] /= a.dat[i];
    }
}

inline void SimpleMatrix::map(double (*f)(const double d)) const
{
    for (int i = 0; i < this->l; ++i)
    {
        this->dat[i] = f(this->dat[i]);
    }
}

inline void SimpleMatrix::transpose() const
{
    auto fst = this->dat, lst = this->dat + this->l - 1;
    for (int i = this->l / 2; i > 0; --i)
    {
        auto t = *fst;
        *fst = *lst;
        *lst = t;
        ++fst;
        --lst;
    }
}

inline double SimpleMatrix::sum() const
{
    double ans = 0.0;
    for (int i = 0; i < this->l; ++i)
    {
        ans += this->dat[i];
    }
    return ans;
}

SimpleVector::SimpleVector() : SimpleMatrix() {}

SimpleVector::SimpleVector(const int n) : SimpleMatrix(1, n) {}

SimpleVector::~SimpleVector() {}

inline double Inner(const SimpleVector& u, const SimpleVector& v)
{
    double s = 0.0;
    for (int i = 0; i < u.l; ++i)
    {
        s += u.dat[i] * v.dat[i];
    }
    return s;
}

inline SimpleMatrix Outer(const SimpleVector& u, const SimpleVector& v)
{
    SimpleMatrix s(u.l, v.l);
    int t = 0;
    for (int i = 0; i < u.l; ++i)
    {
        for (int j = 0; j < v.l; ++j)
        {
            s.dat[t++] = u.dat[i] * v.dat[j];
        }
    }
    return s;
}

SimpleVector Times(const SimpleMatrix& m, const SimpleVector& v)
{
    SimpleVector s(m.r);
    int t = 0;
    s.set(0.0);
    for (int i = 0; i < m.r; ++i)
    {
        for (int j = 0; j < m.c; ++j)
        {
            s.dat[i] += m.dat[t++] * v.dat[j];
        }
    }
    return s;
}

inline SimpleVector Times(const SimpleVector& v, const SimpleMatrix& m)
{
    SimpleVector s(m.c);
    int t = 0;
    s.set(0.0);
    for (int i = 0; i < m.r; ++i)
    {
        for (int j = 0; j < m.c; ++j)
        {
            s.dat[j] += m.dat[t++] * v.dat[i];
        }
    }
    return s;
}

inline SimpleMatrix Times(const SimpleMatrix& m, const SimpleMatrix& n)
{
    SimpleMatrix s(m.r, n.c);
    s.set(0.0);
    for (int k = 0; k < m.c; ++k)
    {
        int u = 0;
        for (int i = 0; i < m.r; ++i)
        {
            int v = k * n.c;
            double r = m.dat[i * m.c + k];
            for (int j = 0; j < n.c; ++j)
            {
                s.dat[u++] += r * n.dat[v++];
            }
        }
    }
    return s;
}

inline SimpleMatrix Conv(const SimpleMatrix& I, const SimpleMatrix& K)
{
    int rs = I.r - K.r + 1, cs = I.c - K.c + 1;
    SimpleMatrix T(rs * cs, K.l);
    double *ptr, *pts;
    for (int i = 0; i < K.r; ++i)
    {
        ptr = T.dat + i * K.c;
        for (int j = 0; j < rs; ++j)
        {
            pts = I.dat + (i + j) * I.c;
            for (int k = 0; k < cs; ++k)
            {
                for (int l = 0; l < K.c; ++l)
                {
                    ptr[l] = pts[l];
                }
                pts += 1;
                ptr += T.c;
            }
        }
    }
    SimpleVector U(K.l);
    ptr = K.dat;
    pts = U.dat;
    for (int i = 0; i < K.r; ++i)
    {
        for (int j = 0; j < K.c; ++j)
        {
            *pts++ = *ptr++;
        }
    }
    SimpleVector&& V = Times(T, U);
    SimpleMatrix S(rs, cs);
    ptr = V.dat;
    pts = S.dat;
    for (int i = 0; i < rs; ++i)
    {
        for (int j = 0; j < cs; ++j)
        {
            *pts++ = *ptr++;
        }
    }
    return S;
}

inline void ConvTo(const SimpleMatrix& I, const SimpleMatrix& K, SimpleMatrix& S)
{
    int rs = I.r - K.r + 1, cs = I.c - K.c + 1;
    SimpleMatrix T(rs * cs, K.l);
    double *ptr, *pts;
    for (int i = 0; i < K.r; ++i)
    {
        ptr = T.dat + i * K.c;
        for (int j = 0; j < rs; ++j)
        {
            pts = I.dat + (i + j) * I.c;
            for (int k = 0; k < cs; ++k)
            {
                for (int l = 0; l < K.c; ++l)
                {
                    ptr[l] = pts[l];
                }
                pts += 1;
                ptr += T.c;
            }
        }
    }
    SimpleVector U(K.l);
    ptr = K.dat;
    pts = U.dat;
    for (int i = 0; i < K.r; ++i)
    {
        for (int j = 0; j < K.c; ++j)
        {
            *pts++ = *ptr++;
        }
    }
    SimpleVector&& V = Times(T, U);
    ptr = V.dat;
    pts = S.dat;
    for (int i = 0; i < rs; ++i)
    {
        for (int j = 0; j < cs; ++j)
        {
            *pts++ = *ptr++;
        }
    }
}

inline double* Copy(double *p, const SimpleMatrix& m)
{
    memcpy(p, m.dat, m.l * sizeof(double));
    return p + m.l;
}

inline void ReCons(list<SimpleMatrix>& L, const SimpleVector& V)
{
    auto ptr = V.dat;
    for (auto&& m : L)
    {
        memcpy(m.dat, ptr, m.l * sizeof(double));
        ptr += m.l;
    }
}

inline void ZeroPad(const list<SimpleMatrix>& src, list<SimpleMatrix>& dst, const int dr, const int dc)
{
    auto is = src.begin();
    auto id = dst.begin();
    while (is != src.end())
    {
        auto ps = is->dat;
        auto pd = id->dat + dr * id->c + dc;
        for (int i = 0; i < is->r; ++i)
        {
            memcpy(pd, ps, is->c * sizeof(double));
            ps += is->c;
            pd += id->c;
        }
        ++is;
        ++id;
    }
}

inline void DTime(const SimpleMatrix& M, const SimpleVector& V)
{
    auto ptr = M.dat;
    for (int i = 0; i < M.r; ++i)
    {
        auto t = V.dat[i];
        for (int j = 0; j < M.c; ++i)
        {
            *ptr++ *= t;
        }
    }
}

inline SimpleMatrix LTime(const SimpleMatrix& M, const SimpleMatrix& V)
{
    int l = V.l, c = M.c;
    SimpleMatrix ans(l, c);
    auto ptr = ans.dat, pts = M.dat;
    for (int i = 0; i < l; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            ptr[j] = V.dat[i] * pts[j];
        }
        ptr += c;
        pts += c;
    }
    return ans;
}