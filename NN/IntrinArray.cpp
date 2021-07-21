#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <list>
using std::list;

#define WIDTH 4
#define block(n) (((n) + WIDTH - 1) / WIDTH)
#define upper(n) (block(n) * WIDTH)
#define length(n) (upper(n) + 1)
#define SUM(A) (A[0] + A[1] + A[2] + A[3])

#define ADD _mm256_add_pd
#define SUB _mm256_sub_pd
#define MUL _mm256_mul_pd
#define DIV _mm256_div_pd
#define LOAD _mm256_load_pd
#define STORE _mm256_store_pd
#define SET _mm256_set1_pd
#define SETI _mm256_set_epi32
#define SETA _mm256_set1_epi32
#define ADDI _mm256_add_epi32
#define MULI _mm256_mul_epi32
#define ANDI _mm256_and_si256
#define PERM _mm256_permutevar8x32_epi32
#define CAST _mm256_castsi256_si128
#define CONV _mm256_cvtepi32_pd

typedef __m256d TYPE;
typedef __m128i HINT;
typedef __m256i MINT;

class IntrinMatrix
{
public:
    int r, c, b, e, l, s;
    double* dat;
    IntrinMatrix();
    IntrinMatrix(const int m, const int n);
    ~IntrinMatrix();
    inline void ini() const;
    inline void set(const double d) const;
    inline void get(const IntrinMatrix& m) const;
    inline void add(const double d) const;
    inline void sub(const double d) const;
    inline void mul(const double d) const;
    inline void div(const double d) const;
    inline void add(const IntrinMatrix& m) const;
    inline void sub(const IntrinMatrix& m) const;
    inline void mul(const IntrinMatrix& m) const;
    inline void div(const IntrinMatrix& m) const;
    inline void map(double (*f)(const double d)) const;
    inline void transpose() const;
    inline double sum() const;
};

class IntrinVector : public IntrinMatrix
{
public:
    IntrinVector();
    IntrinVector(const int l);
    ~IntrinVector();
};

inline double       Inner(const IntrinVector& u, const IntrinVector& v);
inline IntrinMatrix Outer(const IntrinVector& u, const IntrinVector& v);
inline IntrinVector Times(const IntrinMatrix& m, const IntrinVector& v);
inline IntrinVector Times(const IntrinVector& v, const IntrinMatrix& m);
inline IntrinMatrix Times(const IntrinMatrix& m, const IntrinMatrix& n);
inline IntrinMatrix  Conv(const IntrinMatrix& I, const IntrinMatrix& K);
inline void        ConvTo(const IntrinMatrix& I, const IntrinMatrix& K, IntrinMatrix& S);
inline double* Copy(double *p, const IntrinMatrix& m);

IntrinMatrix::IntrinMatrix()
{
    this->r = 0;
    this->c = 0;
    this->b = 0;
    this->e = 0;
    this->l = 0;
    this->s = 0;
    this->dat = NULL;
}

IntrinMatrix::IntrinMatrix(const int m, const int n)
{
    this->r = m;
    this->c = n;
    this->b = block(n);
    this->e = n - upper(n);
    this->l = m * n;
    this->s = length(m * upper(n));
    this->dat = (double*)_mm_malloc(this->s * sizeof(double), 32);
}

IntrinMatrix::~IntrinMatrix()
{
    _mm_free(this->dat);
}

inline void IntrinMatrix::ini() const
{
    HINT r;
    MINT s;
    TYPE u;
    MINT a = SETA(513);
    MINT c = SETA(9973);
    MINT m = SETA(2147483647);
    MINT p = SETI(0, 0, 0, 0, 0, 2, 4, 6);
    MINT t = SETI(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
    TYPE q = SET(1073741823.5);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = MULI(a, t);
            t = ADDI(c, t);
            t = ANDI(m, t);
            s = PERM(t, p);
            r = CAST(s);
            u = CONV(r);
            u = SUB(u, q);
            u = DIV(u, q);
            STORE(ptr, u);
            ptr += WIDTH;
        }
        for (int j = this->e; j < 0; ++j)
        {
            ptr[j] = 0.0;
        }
    }
}

inline void IntrinMatrix::set(const double d) const
{
    TYPE u = SET(d);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            STORE(ptr, u);
            ptr += WIDTH;
        }
        for (int j = this->e; j < 0; ++j)
        {
            ptr[j] = 0.0;
        }
    }
}

inline void IntrinMatrix::get(const IntrinMatrix& m) const
{
    memcpy(this->dat, m.dat, m.s * sizeof(double));
}

inline void IntrinMatrix::add(const double d) const
{
    TYPE t;
    TYPE u = SET(d);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = LOAD(ptr);
            t = ADD(t, u);
            STORE(ptr, t);
            ptr += WIDTH;
        }
        for (int j = this->e; j < 0; ++j)
        {
            ptr[j] = 0.0;
        }
    }
}

inline void IntrinMatrix::sub(const double d) const
{
    TYPE t;
    TYPE u = SET(d);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = LOAD(ptr);
            t = SUB(t, u);
            STORE(ptr, t);
            ptr += WIDTH;
        }
        for (int j = this->e; j < 0; ++j)
        {
            ptr[j] = 0.0;
        }
    }
}

inline void IntrinMatrix::mul(const double d) const
{
    TYPE t;
    TYPE u = SET(d);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = LOAD(ptr);
            t = MUL(t, u);
            STORE(ptr, t);
            ptr += WIDTH;
        }
    }
}

inline void IntrinMatrix::div(const double d) const
{
    TYPE t;
    TYPE u = SET(d);
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = LOAD(ptr);
            t = DIV(t, u);
            STORE(ptr, t);
            ptr += WIDTH;
        }
    }
}

inline void IntrinMatrix::add(const IntrinMatrix& m) const
{
    TYPE s, t;
    double* psr = m.dat;
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            s = LOAD(psr);
            t = LOAD(ptr);
            t = ADD(t, s);
            STORE(ptr, t);
            psr += WIDTH;
            ptr += WIDTH;
        }
    }
}

inline void IntrinMatrix::sub(const IntrinMatrix& m) const
{
    TYPE s, t;
    double* psr = m.dat;
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            s = LOAD(psr);
            t = LOAD(ptr);
            t = SUB(t, s);
            STORE(ptr, t);
            psr += WIDTH;
            ptr += WIDTH;
        }
    }
}

inline void IntrinMatrix::mul(const IntrinMatrix& m) const
{
    TYPE s, t;
    double* psr = m.dat;
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            s = LOAD(psr);
            t = LOAD(ptr);
            t = MUL(t, s);
            STORE(ptr, t);
            psr += WIDTH;
            ptr += WIDTH;
        }
    }
}

inline void IntrinMatrix::div(const IntrinMatrix& m) const
{
    TYPE s, t;
    double* psr = m.dat;
    double* ptr = this->dat;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            s = LOAD(psr);
            t = LOAD(ptr);
            t = DIV(t, s);
            STORE(ptr, t);
            psr += WIDTH;
            ptr += WIDTH;
        }
        for (int j = this->e; j < 0; ++j)
        {
            ptr[j] = 0.0;
        }
    }
}

inline void IntrinMatrix::map(double (*f)(const double d)) const
{
    int k = 0;
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->c; ++j)
        {
            this->dat[k] = f(this->dat[k]);
            ++k;
        }
        k -= this->e;
    }
}

inline void IntrinMatrix::transpose() const
{
    auto fst = this->dat, lst = this->dat + this->r * this->b * WIDTH + this->e - 1;
    for (int i = (this->r * this->b * WIDTH + this->e) / 2; i > 0; --i)
    {
        auto t = *fst;
        *fst = *lst;
        *lst = t;
        ++fst;
        --lst;
    }
}

inline double IntrinMatrix::sum() const
{
    double ans[WIDTH] = { 0.0 };
    TYPE t;
    double* ptr = this->dat;
    TYPE s = SET(0.0);
    for (int i = 0; i < this->r; ++i)
    {
        for (int j = 0; j < this->b; ++j)
        {
            t = LOAD(ptr);
            s = ADD(s, t);
            ptr += WIDTH;
        }
    }
    STORE(ans, s);
    return SUM(ans);
}

IntrinVector::IntrinVector() : IntrinMatrix() {}

IntrinVector::IntrinVector(const int n) : IntrinMatrix(1, n) {}

IntrinVector::~IntrinVector() {}

inline double Inner(const IntrinVector& u, const IntrinVector& v)
{
    double ans[WIDTH];
    TYPE s, p, q;
    double* ptr = u.dat;
    double* psr = v.dat;
    s = SET(0.0);
    for (int i = 0; i < u.b; ++i)
    {
        p = LOAD(ptr);
        q = LOAD(psr);
        p = MUL(p, q);
        s = ADD(s, p);
        ptr += WIDTH;
        psr += WIDTH;
    }
    STORE(ans, s);
    return SUM(ans);
}

IntrinMatrix Outer(const IntrinVector& u, const IntrinVector& v)
{
    TYPE s, t;
    IntrinMatrix m(u.l, v.l);
    double* psr;
    double* ptr = m.dat;
    for (int i = 0; i < m.r; ++i)
    {
        s = SET(u.dat[i]);
        psr = v.dat;
        for (int j = 0; j < m.b; ++j)
        {
            t = LOAD(psr);
            t = MUL(t, s);
            STORE(ptr, t);
            psr += WIDTH;
            ptr += WIDTH;
        }
    }
    return m;
}

inline IntrinVector Times(const IntrinMatrix& m, const IntrinVector& v)
{
    IntrinVector s(m.r);
    TYPE r, t, d;
    double* prr = m.dat;
    double* psr;
    double ans[WIDTH];
    for (int i = 0; i < m.r; ++i)
    {
        d = SET(0.0);
        psr = v.dat;
        for (int j = 0; j < m.b; ++j)
        {
            r = LOAD(prr);
            t = LOAD(psr);
            t = MUL(t, r);
            d = ADD(d, t);
            prr += WIDTH;
            psr += WIDTH;
        }
        STORE(ans, d);
        s.dat[i] = SUM(ans);
    }
    for (int i = s.c; i < s.b * WIDTH; ++i)
    {
        s.dat[i] = 0.0;
    }
    return s;
}

inline IntrinVector Times(const IntrinVector& v, const IntrinMatrix& m)
{
    IntrinVector s(m.c);
    TYPE r, t, d;
    double* prr = m.dat;
    double* psr;
    s.set(0.0);
    for (int i = 0; i < m.r; ++i)
    {
        psr = s.dat;
        r = SET(v.dat[i]);
        for (int j = 0; j < m.b; ++j)
        {
            t = LOAD(prr);
            t = MUL(t, r);
            d = LOAD(psr);
            d = ADD(d, t);
            STORE(psr, d);
            prr += WIDTH;
            psr += WIDTH;
        }
    }
    return s;
}

inline IntrinMatrix Times(const IntrinMatrix& m, const IntrinMatrix& n)
{
    IntrinMatrix s(m.r, n.c);
    TYPE u, v, w;
    double* ptr;
    double* psr;
    s.set(0.0);
    for (int k = 0; k < m.c; ++k)
    {
        psr = s.dat;
        for (int i = 0; i < m.r; ++i)
        {
            ptr = n.dat + ((k * n.b) * WIDTH);
            u = SET(m.dat[((i * m.b) * WIDTH) + k]);
            for (int j = 0; j < n.b; ++j)
            {
                v = LOAD(ptr);
                v = MUL(u, v);
                w = LOAD(psr);
                w = ADD(w, v);
                STORE(psr, w);
                ptr += WIDTH;
                psr += WIDTH;
            }
        }
    }
    return s;
}

inline IntrinMatrix Conv(const IntrinMatrix& I, const IntrinMatrix& K)
{
    TYPE o = SET(0.0);
    int rs = I.r - K.r + 1, cs = I.c - K.c + 1;
    IntrinMatrix T(rs * cs, K.l);
    double *ptr, *pts;
    ptr = T.dat + (T.b - 1) * WIDTH;
    for (int i = 0; i < T.r; ++i)
    {
        STORE(ptr, o);
        ptr += upper(T.c);
    }
    for (int i = 0; i < K.r; ++i)
    {
        ptr = T.dat + i * K.c;
        for (int j = 0; j < rs; ++j)
        {
            pts = I.dat + (i + j) * upper(I.c);
            for (int k = 0; k < cs; ++k)
            {
                memcpy(ptr, pts, K.c * sizeof(double));
                pts += 1;
                ptr += upper(T.c);
            }
        }
    }
    IntrinVector U(K.l);
    ptr = K.dat;
    pts = U.dat;
    for (int i = 0; i < K.r; ++i)
    {
        for (int j = 0; j < K.c; ++j)
        {
            *pts++ = *ptr++;
        }
        ptr -= K.e;
    }
    for (int i = U.e; i < 0; ++i)
    {
        *pts++ = 0.0;
    }
    IntrinVector&& V = Times(T, U);
    IntrinMatrix S(rs, cs);
    ptr = V.dat;
    pts = S.dat;
    for (int i = 0; i < rs; ++i)
    {
        for (int j = 0; j < cs; ++j)
        {
            *pts++ = *ptr++;
        }
        for (int j = S.e; j < 0; ++j)
        {
            *pts++ = 0.0;
        }
    }
    return S;
}

inline void ConvTo(const IntrinMatrix& I, const IntrinMatrix& K, IntrinMatrix& S)
{
    TYPE o = SET(0.0);
    int rs = I.r - K.r + 1, cs = I.c - K.c + 1;
    IntrinMatrix T(rs * cs, K.l);
    double *ptr, *pts;
    ptr = T.dat + (T.b - 1) * WIDTH;
    for (int i = 0; i < T.r; ++i)
    {
        STORE(ptr, o);
        ptr += upper(T.c);
    }
    for (int i = 0; i < K.r; ++i)
    {
        ptr = T.dat + i * K.c;
        for (int j = 0; j < rs; ++j)
        {
            pts = I.dat + (i + j) * upper(I.c);
            for (int k = 0; k < cs; ++k)
            {
                memcpy(ptr, pts, K.c * sizeof(double));
                pts += 1;
                ptr += upper(T.c);
            }
        }
    }
    IntrinVector U(K.l);
    ptr = K.dat;
    pts = U.dat;
    for (int i = 0; i < K.r; ++i)
    {
        for (int j = 0; j < K.c; ++j)
        {
            *pts++ = *ptr++;
        }
        ptr -= K.e;
    }
    for (int i = U.e; i < 0; ++i)
    {
        *pts++ = 0.0;
    }
    IntrinVector&& V = Times(T, U);
    ptr = V.dat;
    pts = S.dat;
    for (int i = 0; i < rs; ++i)
    {
        for (int j = 0; j < cs; ++j)
        {
            *pts++ = *ptr++;
        }
        for (int j = S.e; j < 0; ++j)
        {
            *pts++ = 0.0;
        }
    }
}

inline double* Copy(double *p, const IntrinMatrix& m)
{
    auto q = m.dat;
    for (int i = 0; i < m.r; ++i)
    {
        memcpy(p, q, m.c * sizeof(double));
        p += m.c;
        q += m.b * WIDTH;
    }
    return p;
}

inline void ReCons(list<IntrinMatrix>& L, const IntrinVector& V)
{
    auto ptr = V.dat;
    for (auto&& m : L)
    {
        m.set(0.0);
        auto dst = m.dat;
        for (int i = 0; i < m.r; ++i)
        {
            memcpy(dst, ptr, m.c * sizeof(double));
            ptr += m.c;
            dst += m.b * WIDTH;
        }
    }
}

inline void ZeroPad(const list<IntrinMatrix>& src, list<IntrinMatrix>& dst, const int dr, const int dc)
{
    auto is = src.begin();
    auto id = dst.begin();
    while (is != src.end())
    {
        auto ps = is->dat;
        auto pd = id->dat + dr * id->b * WIDTH + dc;
        for (int i = 0; i < is->r; ++i)
        {
            memcpy(pd, ps, is->c * sizeof(double));
            ps += is->b * WIDTH;
            pd += id->b * WIDTH;
        }
        ++is;
        ++id;
    }
}

inline void DTime(const IntrinMatrix& M, const IntrinVector& V)
{
    auto ptr = M.dat;
    for (int i = 0; i < M.r; ++i)
    {
        auto p = ptr;
        auto t = V.dat[i];
        for (int j = 0; j < M.c; ++i)
        {
            *p++ *= t;
        }
        ptr += M.b * WIDTH;
    }
}

inline IntrinMatrix LTime(const IntrinMatrix& M, const IntrinMatrix& V)
{
    int l = V.c, c = M.c, s = M.b * WIDTH;
    IntrinMatrix ans(l, c);
    ans.set(0.0);
    auto ptr = ans.dat, pts = M.dat;
    for (int i = 0; i < l; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            ptr[j] = V.dat[i] * pts[j];
        }
        ptr += s;
        pts += s;
    }
    return ans;
}