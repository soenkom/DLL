#include <list>
#include <cmath>
#include <cstring>
using std::list;

template <typename matrix, typename vector>
class Kernel
{
public:
    int r, c, h, n;
    list<matrix> *knl;
    Kernel(const int r, const int c, const int h, const int n);
    ~Kernel();
    inline void ini();
    inline void set(const double d);
};

template <typename matrix, typename vector>
class CONV_net
{
public:
    double alpha, beta, epsilon, at, bt, e, count;
    list<Kernel<matrix, vector> > K;
    list<Kernel<matrix, vector> > mK;
    list<Kernel<matrix, vector> > Km;
    list<Kernel<matrix, vector> > Kv;
    list<matrix> w;
    list<matrix> mw;
    list<vector> b;
    list<vector> mb;
    list<matrix> wm;
    list<matrix> wv;
    list<vector> bm;
    list<vector> bv;
    CONV_net(const list<int>& l, const list<int>& f);
    ~CONV_net();
    inline vector forward(const list<matrix>& I);
    inline void train(const list<matrix>& I, const vector& T, const double eta);
    inline void update();
};

constexpr double eps = 1e-9;

inline double ReLU(const double d)
{
    return d > eps ? d : 0.0;
}

inline double Sigmoid(const double d)
{
    return 1.0 / (1.0 + exp(-d));
}

inline double ReLUp(const double d)
{
    return d > eps ? 1.0 : 0.0;
}

inline double Sigmoidp(const double d)
{
    double t = Sigmoid(d);
    return t * (1 - t);
}

inline double Exp(const double d)
{
    return exp(d);
}

inline double Log(const double d)
{
    return log(d);
}

inline double Square(const double d)
{
    return  d * d;
}

inline double Sqrt(const double d)
{
    return sqrt(d);
}

inline double Inv(const double d)
{
    return 1.0 / d;
}

template <typename vector>
inline void Softmax(const vector& v)
{
    v.map(Exp);
    auto s = v.sum();
    v.div(s);
}

template <typename matrix, typename vector>
Kernel<matrix, vector>::Kernel(const int r, const int c, const int h, const int n)
{
    this->r = r;
    this->c = c;
    this->h = h;
    this->n = n;
    this->knl = new list<matrix>[this->n];
    for (int i = 0; i < this->n; ++i)
    {
        for (int j = 0; j < this->h; ++j)
        {
            this->knl[i].emplace_back(this->r, this->c);
        }
    }
}

template <typename matrix, typename vector>
Kernel<matrix, vector>::~Kernel()
{
    delete []this->knl;
}

template <typename matrix, typename vector>
inline void Kernel<matrix, vector>::ini()
{
    for (int i = 0; i < this->n; ++i)
    {
        for (auto&& m : this->knl[i])
        {
            m.ini();
            m.mul(sqrt(6.0 / (this->r * this->c * (this->h + this->n))));
        }
    }
}

template <typename matrix, typename vector>
inline void Kernel<matrix, vector>::set(const double d)
{
    for (int i = 0; i < this->n; ++i)
    {
        for (auto&& m : this->knl[i])
        {
            m.set(d);
        }
    }
}
template <typename matrix, typename vector>
CONV_net<matrix, vector>::CONV_net(const list<int>& l, const list<int>& f)
{
    int r, c, h, n;
    auto t = l.begin();
    while (t != l.end())
    {
        r = *t++;
        c = *t++;
        h = *t++;
        n = *t++;
        this->K.emplace_back(r, c, h, n);
        this->K.back().ini();
        this->mK.emplace_back(r, c, h, n);
        this->mK.back().set(0.0);
        this->Km.emplace_back(r, c, h, n);
        this->Km.back().set(0.0);
        this->Kv.emplace_back(r, c, h, n);
        this->Kv.back().set(0.0);
    }
    auto p = f.begin();
    r = *p++;
    while (p != f.end())
    {
        c = r;
        r = *p++;
        this->w.emplace_back(r, c);
        this->w.back().ini();
        this->b.emplace_back(r);
        this->b.back().ini();
        this->mw.emplace_back(r, c);
        this->mw.back().set(0.0);
        this->mb.emplace_back(r);
        this->mb.back().set(0.0);
        this->wm.emplace_back(r, c);
        this->wm.back().set(0.0);
        this->wv.emplace_back(r, c);
        this->wv.back().set(0.0);
        this->bm.emplace_back(r);
        this->bm.back().set(0.0);
        this->bv.emplace_back(r);
        this->bv.back().set(0.0);
    }
    alpha = 0.9;
    beta = 0.999;
    epsilon = 1e-8;
    at = 1.0;
    bt = 1.0;
    e = 0.0;
    count = 0.0;
}

template <typename matrix, typename vector>
CONV_net<matrix, vector>::~CONV_net() {}

template <typename matrix, typename vector>
inline vector CONV_net<matrix, vector>::forward(const list<matrix>& I)
{
    auto p = this->K.begin();
    auto pt = new list<matrix>;
    auto ps = new list<matrix>;
    Conv(I, *p++, *pt);
    for (auto&& m : *pt)
    {
        m.map(ReLU);
    }
    while (p != this->K.end())
    {
        Conv(*pt, *p++, *ps);
        auto tp = pt;
        pt = ps;
        ps = tp;
        ps->clear();
        for (auto&& m : *pt)
        {
            m.map(ReLU);
        }
    }
    vector V(this->w.front().c);
    auto ptr = V.dat;
    V.set(0.0);
    for (auto&& m : *pt)
    {
        ptr = Copy(ptr, m);
    }
    auto iw = this->w.begin();
    auto ib = this->b.begin();
    while (iw != this->w.end())
    {
        auto&& U = Times(*iw++, V);
        U.add(*ib++);
        U.map(Sigmoid);
        auto tv = V;
        V = U;
        U = tv;
        tv.dat = nullptr;
    }
    Softmax(V);
    delete pt;
    delete ps;
    return V;
}

template <typename matrix, typename vector>
inline void CONV_net<matrix, vector>::train(const list<matrix>& I, const vector& T, const double eta)
{
    count += 1.0;
    e = eta;
    auto pk = this->K.begin();
    list<matrix>* ps = nullptr;
    list<matrix>* pt = nullptr;
    auto cx = new list<matrix>*[this->K.size()];
    auto cz = new list<matrix>*[this->K.size()];
    int lay = 0;
    ps = new list<matrix>(I);
    while (pk != this->K.end())
    {
        cx[lay] = ps;
        pt = new list<matrix>;
        Conv(*cx[lay], *pk++, *pt);
        cz[lay] = pt;
        ps = new list<matrix>;
        for (auto&& m : *pt)
        {
            ps->emplace_back(m.r, m.c);
            ps->back().get(m);
            ps->back().map(ReLU);
        }
        ++lay;
    }
    pt = nullptr;
    list<vector> fx;
    list<vector> fz;
    vector V(this->w.front().c);
    auto pdt = V.dat;
    for (auto&& m : *ps)
    {
        pdt = Copy(pdt, m);
    }
    delete ps;
    auto iw = this->w.begin();
    auto ib = this->b.begin();
    while (iw != this->w.end())
    {
        fx.emplace_back(V.l);
        fx.back().get(V);
        auto&& U = Times(*iw++, V);
        U.add(*ib++);
        fz.emplace_back(U.l);
        fz.back().get(U);
        U.map(Sigmoid);
        auto tv = V;
        V = U;
        U = tv;
        tv.dat = nullptr;
    }
    auto pz = fz.rbegin();
    auto px = fx.rbegin();
    auto pw = this->w.rbegin();
    auto pb = this->b.rbegin();
    auto pmw = this->mw.rbegin();
    auto pmb = this->mb.rbegin();
    Softmax(V);
    V.sub(T);
    pz->map(Sigmoidp);
    pz->mul(V);
    while (true)
    {
        auto&& dw = Outer(*pz, *px);
        pmw->add(dw);
        pmb->add(*pz);
        ++pmw;
        ++pmb;
        if (pmw == this->mw.rend()) break;
        auto&& del = Times(*pz, *pw);
        ++pz;
        pz->map(Sigmoidp);
        pz->mul(del);
        ++px;
        ++pw;
        ++pb;
    }
    list<matrix> err;
    for (auto&& m : *cz[lay - 1])
    {
        err.emplace_back(m.r, m.c);
    }
    auto nxt = Times(*pz, *pw);
    ReCons(err, nxt);
    auto ple = err.begin();
    auto plz = cz[lay - 1]->begin();
    while (ple != err.end())
    {
        plz->map(ReLUp);
        ple->mul(*plz);
        ++plz;
        ++ple;
    }
    auto ipk = this->K.rbegin();
    auto ipm = this->mK.rbegin();
    while (true)
    {
        --lay;
        auto imp = err.begin();
        for (int i = 0; i < ipm->n; ++i)
        {
            auto inp = cx[lay]->begin();
            for (auto&& m : ipm->knl[i])
            {
                auto&& tm = Conv(*inp, *imp);
                m.add(tm);
                ++inp;
            }
            ++imp;
        }
        if (lay == 0) break;
        int r = err.front().r, c = err.front().c;
        int kr = ipk->r, kc = ipk->c;
        list<matrix> lpd;
        for (uint32_t i = 0; i < err.size(); ++i)
        {
            lpd.emplace_back(r + 2 * kr - 2, c + 2 * kc - 2);
            lpd.back().set(0.0);
        }
        ZeroPad(err, lpd, kr - 1, kc - 1);
        err.clear();
        for (int i = 0; i < ipk->h; ++i)
        {
            err.emplace_back(r + kr - 1, c + kc - 1);
            err.back().set(0.0);
        }
        auto img = lpd.begin();
        for (int i = 0; i < ipk->n; ++i)
        {
            auto ph = err.begin();
            for (auto&& m : ipk->knl[i])
            {
                m.transpose();
                auto&& tm = Conv(*img, m);
                m.transpose();
                ph->add(tm);
                ++ph;
            }
            ++img;
        }
        auto pcz = cz[lay - 1]->begin();
        auto prr = err.begin();
        for (int i = 0; i < ipk->h; ++i)
        {
            pcz->map(ReLUp);
            prr->mul(*pcz);
            ++pcz;
            ++prr;
        }
        ++ipm;
        ++ipk;
    }
    for (auto&& m : *cx[0])
    {
        m.dat = nullptr;
    }
    for (uint32_t i = 0; i < this->K.size(); ++i)
    {
        delete cx[i];
        delete cz[i];
    }
    delete[] cx;
    delete[] cz;
}

template <typename matrix, typename vector>
inline void CONV_net<matrix, vector>::update()
{
    at *= alpha;
    bt *= beta;
    auto pk = this->K.begin();
    auto pmk = this->mK.begin();
    auto pkm = this->Km.begin();
    auto pkv = this->Kv.begin();
    while (pk != this->K.end())
    {
        for (int i = 0; i < pk->n; ++i)
        {
            auto ppk = pk->knl[i].begin();
            auto ppm = pmk->knl[i].begin();
            auto ppkm = pkm->knl[i].begin();
            auto ppkv = pkv->knl[i].begin();
            while (ppk != pk->knl[i].end())
            {
                ppm->mul(1.0 / count);
                ppkm->mul(alpha);
                ppm->mul(1.0 - alpha);
                ppkm->add(*ppm);
                ppkv->mul(beta);
                ppm->mul(1.0 / (1.0 - alpha));
                ppm->map(Square);
                ppm->mul(1.0 - beta);
                ppkv->add(*ppm);
                ppm->get(*ppkv);
                ppm->map(Sqrt);
                ppm->add(epsilon * (1.0 - bt));
                ppm->map(Inv);
                ppm->mul(*ppkm);
                ppm->mul(e * (1.0 - bt) / (1.0 - at));
                ppk->sub(*ppm);
                ppm->set(0.0);
                ++ppk;
                ++ppm;
                ++ppkm;
                ++ppkv;
            }
        }
        ++pk;
        ++pmk;
        ++pkm;
        ++pkv;
    }
    auto pw = this->w.begin();
    auto pb = this->b.begin();
    auto pmw = this->mw.begin();
    auto pmb = this->mb.begin();
    auto pwm = this->wm.begin();
    auto pwv = this->wv.begin();
    auto pbm = this->bm.begin();
    auto pbv = this->bv.begin();
    while (pw != this->w.end())
    {
        pmw->mul(1.0 / count);
        pwm->mul(alpha);
        pmw->mul(1.0 - alpha);
        pwm->add(*pmw);
        pwv->mul(beta);
        pmw->mul(1.0 / (1 - alpha));
        pmw->map(Square);
        pmw->mul(1.0 - beta);
        pwv->add(*pmw);
        pmw->get(*pwv);
        pmw->map(Sqrt);
        pmw->add(epsilon * (1.0 - bt));
        pmw->map(Inv);
        pmw->mul(*pwm);
        pmw->mul(e * (1.0 - bt) / (1.0 - at));
        pw->sub(*pmw);
        pmw->set(0.0);
        pmb->mul(1.0 / count);
        pbm->mul(alpha);
        pmb->mul(1.0 - alpha);
        pbm->add(*pmb);
        pbv->mul(beta);
        pmb->mul(1.0 / (1.0 - alpha));
        pmb->map(Square);
        pmb->mul(1.0 - beta);
        pbv->add(*pmb);
        pmb->get(*pbv);
        pmb->map(Sqrt);
        pmb->add(epsilon * (1.0 - bt));
        pmb->map(Inv);
        pmb->mul(*pbm);
        pmb->mul(e * (1.0 - bt) / (1.0 - at));
        pb->sub(*pmb);
        pmb->set(0.0);
        ++pw;
        ++pb;
        ++pmw;
        ++pmb;
        ++pwm;
        ++pwv;
        ++pbm;
        ++pbv;
    }
    count = 0.0;
}

template <typename matrix, typename vector>
inline void Conv(const list<matrix>& I, const Kernel<matrix, vector>& K, list<matrix>& L)
{
    int r = I.front().r - K.r + 1, c = I.front().c - K.c + 1;
    for (int i = 0; i < K.n; ++i)
    {
        L.emplace_back(r, c);
    }
    int t = 0;
    matrix C(r, c);
    for (auto&& m : L)
    {
        m.set(0.0);
        auto ii = I.begin();
        auto ik = K.knl[t].begin();
        while (ii != I.end())
        {
            ConvTo(*ii++, *ik++, C);
            m.add(C);
        }
        ++t;
    }
}
