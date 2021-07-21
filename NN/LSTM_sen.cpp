#include <list>
#include <cstring>
#include <cmath>
using std::list;

template <typename matrix, typename vector>
class LSTM_sen
{
public:
    int n;
    double alpha, beta, epsilon, at, bt, e, count;
    list<matrix> w, mw, wm, wv;
    list<vector> b, mb, bm, bv;
    list<vector> cell, hide, data;
    list<vector> before[5];
    list<vector> after[5];
    list<matrix> difw[8];
    list<vector> difb[4];
    list<matrix> wa, mwa, wam, wav;
    list<vector> ba, mba, bam, bav, z, x;
    LSTM_sen(list<int> l);
    ~LSTM_sen();
    void forward(const vector& V);
    inline vector predict();
    inline void train(const vector& T, const double eta);
    inline void update();
    inline void clear();
};

inline double Tanh(const double d)
{
    return tanh(d);
}

inline double Sigmoid(const double d)
{
    return 1.0 / (1.0 + exp(-d));
}

inline double Tanhp(const double d)
{
    auto t = Tanh(d);
    return 1 - t * t;
}

inline double Sigmoidp(const double d)
{
    auto t = Sigmoid(d);
    return t * (1 - t);
}

inline double Exp(const double d)
{
    return exp(d);
}

template <typename vector>
inline void Softmax(const vector& v)
{
    v.map(Exp);
    auto s = v.sum();
    v.div(s);
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

template <typename matrix, typename vector>
LSTM_sen<matrix, vector>::LSTM_sen(list<int> l) : n(l.front()), alpha(0.9), beta(0.999), epsilon(1e-8), at(1.0), bt(1.0), e(0.0), count(0.0)
{
    for (int i = 0; i < 8; ++i)
    {
        this->w.emplace_back(n, n);
        this->w.back().ini();
        this->mw.emplace_back(n, n);
        this->mw.back().set(0.0);
        this->wm.emplace_back(n, n);
        this->wm.back().set(0.0);
        this->wv.emplace_back(n, n);
        this->wv.back().set(0.0);
    }
    for (int i = 0; i < 4; ++i)
    {
        this->b.emplace_back(n);
        this->b.back().ini();
        this->mb.emplace_back(n);
        this->mb.back().set(0.0);
        this->bm.emplace_back(n);
        this->bm.back().set(0.0);
        this->bv.emplace_back(n);
        this->bv.back().set(0.0);
    }
    this->cell.emplace_back(n);
    this->cell.back().set(0.0);
    this->hide.emplace_back(n);
    this->hide.back().set(0.0);
    for (int i = 0; i < 8; ++i)
    {
        this->difw[i].emplace_back(n, n);
        this->difw[i].back().set(0.0);
    }
    for (int i = 0; i < 4; ++i)
    {
        this->difb[i].emplace_back(n);
        this->difb[i].back().set(0.0);
    }
    auto adv = l.begin();
    auto ptr = l.begin();
    ++adv;
    this->x.emplace_back(*ptr);
    while (adv != l.end())
    {
        this->wa.emplace_back(*adv, *ptr);
        this->wa.back().ini();
        this->mwa.emplace_back(*adv, *ptr);
        this->mwa.back().set(0.0);
        this->wam.emplace_back(*adv, *ptr);
        this->wam.back().set(0.0);
        this->wav.emplace_back(*adv, *ptr);
        this->wav.back().set(0.0);
        this->ba.emplace_back(*adv);
        this->ba.back().ini();
        this->mba.emplace_back(*adv);
        this->mba.back().set(0.0);
        this->bam.emplace_back(*adv);
        this->bam.back().set(0.0);
        this->bav.emplace_back(*adv);
        this->bav.back().set(0.0);
        this->x.emplace_back(*adv);
        this->z.emplace_back(*adv);
        ++adv;
        ++ptr;
    }
}

template <typename matrix, typename vector>
LSTM_sen<matrix, vector>::~LSTM_sen() {}

template <typename matrix, typename vector>
inline void LSTM_sen<matrix, vector>::forward(const vector& V)
{
    this->data.emplace_back(this->n);
    this->data.back().get(V);
    auto iw = this->w.begin();
    auto ib = this->b.begin();
    auto&& ic = Times(*iw++, this->hide.back());
    auto&& ix = Times(*iw++, V);
    ix.add(ic);
    ix.add(*ib++);
    before[0].emplace_back(n);
    before[0].back().get(ix);
    before[0].back().map(Sigmoidp);
    ix.map(Sigmoid);
    after[0].emplace_back(n);
    after[0].back().get(ix);
    auto&& fc = Times(*iw++, this->hide.back());
    auto&& fx = Times(*iw++, V);
    fx.add(fc);
    fx.add(*ib++);
    before[1].emplace_back(n);
    before[1].back().get(fx);
    before[1].back().map(Sigmoidp);
    fx.map(Sigmoid);
    after[1].emplace_back(n);
    after[1].back().get(fx);
    auto&& cc = Times(*iw++, this->hide.back());
    auto&& cx = Times(*iw++, V);
    cx.add(cc);
    cx.add(*ib++);
    before[2].emplace_back(n);
    before[2].back().get(cx);
    before[2].back().map(Tanhp);
    cx.map(Tanh);
    after[2].emplace_back(n);
    after[2].back().get(cx);
    ix.mul(cx);
    fx.mul(this->cell.back());
    ix.add(fx);
    before[3].emplace_back(n);
    before[3].back().get(ix);
    before[3].back().map(Tanhp);
    this->cell.emplace_back(this->n);
    this->cell.back().get(ix);
    ix.map(Tanh);
    after[3].emplace_back(n);
    after[3].back().get(ix);
    auto&& oc = Times(*iw++, this->hide.back());
    auto&& ox = Times(*iw++, V);
    ox.add(oc);
    ox.add(*ib++);
    before[4].emplace_back(n);
    before[4].back().get(ox);
    before[4].back().map(Sigmoidp);
    ox.map(Sigmoid);
    after[4].emplace_back(n);
    after[4].back().get(ox);
    ox.mul(ix);
    this->hide.emplace_back(this->n);
    this->hide.back().get(ox);
    do
    {
        decltype(this->difw[0].rbegin()) idw[8];
        decltype(this->difb[0].rbegin()) idb[4];
        for (int i = 0; i < 8; ++i)
        {
            idw[i] = this->difw[i].rbegin();
        }
        for (int i = 0; i < 4; ++i)
        {
            idb[i] = this->difb[i].rbegin();
        }
        auto ih = this->hide.rbegin();
        auto id = this->data.rbegin();
        auto ie = this->cell.rbegin();
        ++ih;
        ++ie;
        vector utp(n), vtp(n);
        utp.get(after[2].back());
        utp.mul(before[0].back());
        auto&& dwhi = Outer(utp, *ih);
        auto&& dwxi = Outer(utp, *id);
        auto&& dvhi = LTime(*idw[0], after[1].back());
        auto&& dvxi = LTime(*idw[1], after[1].back());
        dwhi.add(dvhi);
        dwxi.add(dvxi);
        vtp.get(after[1].back());
        vtp.mul(*idb[0]);
        utp.add(vtp);
        this->difw[0].emplace_back(this->n, this->n);
        this->difw[0].back().get(dwhi);
        this->difw[1].emplace_back(this->n, this->n);
        this->difw[1].back().get(dwxi);
        this->difb[0].emplace_back(this->n);
        this->difb[0].back().get(utp);
        utp.get(*ie);
        utp.mul(before[1].back());
        auto&& dwhf = Outer(utp, *ih);
        auto&& dwxf = Outer(utp, *id);
        auto&& dvhf = LTime(*idw[2], after[1].back());
        auto&& dvxf = LTime(*idw[3], after[1].back());
        dwhf.add(dvhf);
        dwxf.add(dvxf);
        vtp.get(after[1].back());
        vtp.mul(*idb[1]);
        utp.add(vtp);
        this->difw[2].emplace_back(this->n, this->n);
        this->difw[2].back().get(dwhf);
        this->difw[3].emplace_back(this->n, this->n);
        this->difw[3].back().get(dwxf);
        this->difb[1].emplace_back(this->n);
        this->difb[1].back().get(utp);
        utp.get(after[0].back());
        utp.mul(before[2].back());
        auto&& dwhc = Outer(utp, *ih);
        auto&& dwxc = Outer(utp, *id);
        auto&& dvhc = LTime(*idw[4], after[1].back());
        auto&& dvxc = LTime(*idw[5], after[1].back());
        dwhc.add(dvhc);
        dwxc.add(dvxc);
        vtp.get(after[1].back());
        vtp.mul(*idb[2]);
        utp.add(vtp);
        this->difw[4].emplace_back(this->n, this->n);
        this->difw[4].back().get(dwhc);
        this->difw[5].emplace_back(this->n, this->n);
        this->difw[5].back().get(dwxc);
        this->difb[2].emplace_back(this->n);
        this->difb[2].back().get(utp);
        utp.get(after[1].back());
        auto&& dvho = LTime(*idw[6], *ih);
        auto&& dvxo = LTime(*idw[7], *id);
        utp.mul(*idb[3]);
        this->difw[6].emplace_back(this->n, this->n);
        this->difw[6].back().get(dvho);
        this->difw[7].emplace_back(this->n, this->n);
        this->difw[7].back().get(dvxo);
        this->difb[3].emplace_back(this->n);
        this->difb[3].back().get(utp);
    } while(false);
}

template <typename matrix, typename vector>
inline vector LSTM_sen<matrix, vector>::predict()
{
    auto iw = this->wa.begin();
    auto ib = this->ba.begin();
    vector x(this->n);
    x.get(this->hide.back());
    while (iw != this->wa.end())
    {
        auto&& z = Times(*iw, x);
        z.add(*ib);
        z.map(Sigmoid);
        auto tz = z;
        z = x;
        x = tz;
        tz.dat = nullptr;
        ++iw;
        ++ib;
    }
    Softmax(x);
    return x;
}
template <typename matrix, typename vector>
inline void LSTM_sen<matrix, vector>::train(const vector& T, const double eta)
{
    e = eta;
    count += 1.0;
    auto iw = this->wa.begin();
    auto ib = this->ba.begin();
    auto ix = this->x.begin();
    auto iz = this->z.begin();
    ix->get(this->hide.back());
    while (iw != this->wa.end())
    {
        auto&& z = Times(*iw, *ix);
        z.add(*ib);
        iz->get(z);
        iz->map(Sigmoidp);
        ++ix;
        ix->get(z);
        ix->map(Sigmoid);
        ++iw;
        ++ib;
        ++iz;
    }
    auto rw = this->mwa.rbegin();
    auto rb = this->mba.rbegin();
    auto vw = this->wa.rbegin();
    auto rz = this->z.rbegin();
    auto rx = this->x.rbegin();
    auto sx = this->x.rbegin();
    ++sx;
    Softmax(*rx);
    rx->sub(T);
    while (rw != this->mwa.rend())
    {
        rx->mul(*rz);
        auto&& tm = Outer(*rx, *sx);
        rw->add(tm);
        rb->add(*rx);
        ++rz;
        auto&& err = Times(*rx, *vw);
        ++vw;
        ++rw;
        ++rb;
        ++rx;
        ++sx;
        rx->get(err);
    }
    vector err(this->n);
    err.get(this->x.front());
    decltype(before[0].rbegin()) ibf[5];
    decltype(after[0].rbegin()) ift[5];
    auto icl = this->cell.rbegin();
    auto ihd = this->hide.rbegin();
    auto idt = this->data.rbegin();
    ++icl;
    ++ihd;
    for (int i = 0; i < 5; ++i)
    {
        ibf[i] = before[i].rbegin();
        ift[i] = after[i].rbegin();
    }
    decltype(this->difw[0].rbegin()) idw[8];
    decltype(this->difb[0].rbegin()) idb[4];
    for (int i = 0; i < 8; ++i)
    {
        idw[i] = this->difw[i].rbegin();
    }
    for (int i = 0; i < 4; ++i)
    {
        idb[i] = this->difb[i].rbegin();
    }
    while (icl != this->cell.rend())
    {
        auto imw = this->mw.begin();
        auto imb = this->mb.begin();
        vector vtp(n), utp(n);;
        vtp.get(*ift[4]);
        vtp.mul(*ibf[3]);
        vtp.mul(err);
        utp.get(vtp);
        auto&& dih = LTime(*idw[0], vtp);
        auto&& dix = LTime(*idw[1], vtp);
        vtp.mul(*idb[0]);
        imw->add(dih);
        ++imw;
        imw->add(dix);
        ++imw;
        imb->add(vtp);
        ++imb;
        vtp.get(utp);
        auto&& dfh = LTime(*idw[2], vtp);
        auto&& dfx = LTime(*idw[3], vtp);
        vtp.mul(*idb[1]);
        imw->add(dfh);
        ++imw;
        imw->add(dfx);
        ++imw;
        imb->add(vtp);
        ++imb;
        vtp.get(utp);
        auto&& dch = LTime(*idw[4], vtp);
        auto&& dcx = LTime(*idw[5], vtp);
        vtp.mul(*idb[2]);
        imw->add(dch);
        ++imw;
        imw->add(dcx);
        ++imw;
        imb->add(vtp);
        ++imb;
        vtp.get(utp);
        auto&& doh = LTime(*idw[6], vtp);
        auto&& dox = LTime(*idw[7], vtp);
        vtp.mul(*idb[3]);
        utp.get(after[3].back());
        utp.mul(before[4].back());
        utp.mul(err);
        auto&& duh = Outer(utp, *ihd);
        auto&& dux = Outer(utp, *idt);
        doh.add(duh);
        dox.add(dux);
        vtp.add(utp);
        imw->add(doh);
        ++imw;
        imw->add(dox);
        ++imw;
        imb->add(vtp);
        ++imb;
        ++ihd;
        ++idt;
        if (idt == this->data.rend()) break;
        iw = this->w.begin();
        ib = this->b.begin();
        vtp.get(*ift[4]);
        vtp.mul(*ibf[3]);
        vtp.mul(*ift[2]);
        vtp.mul(*ibf[0]);
        auto&& e1 = LTime(*iw, vtp);
        ++iw; ++iw;
        vtp.get(*ift[4]);
        vtp.mul(*ibf[3]);
        vtp.mul(*icl);
        vtp.mul(*ibf[1]);
        auto&& e2 = LTime(*iw, vtp);
        ++iw; ++iw;
        vtp.get(*ift[4]);
        vtp.mul(*ibf[3]);
        vtp.mul(*ift[0]);
        vtp.mul(*ibf[2]);
        auto&& e3 = LTime(*iw, vtp);
        ++iw; ++iw;
        vtp.get(*ift[3]);
        vtp.mul(*ibf[4]);
        auto&& e4 = LTime(*iw, vtp);
        e1.add(e2);
        e1.add(e3);
        e1.add(e4);
        auto&& ers = Times(err, e1);
        auto ptt = ers.dat;
        ers.dat = err.dat;
        err.dat = ptt;
        for (int i = 0; i < 5; ++i)
        {
            ++ibf[i];
            ++ift[i];
        }
        for (int i = 0; i < 8; ++i)
        {
            ++idw[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            ++idb[i];
        }
        ++icl;
    }
}

template <typename matrix, typename vector>
inline void LSTM_sen<matrix, vector>::update()
{
    at *= alpha;
    bt *= beta;
    auto iw = this->w.begin();
    auto ib = this->b.begin();
    auto imw = this->mw.begin();
    auto imb = this->mb.begin();
    auto iwm = this->wm.begin();
    auto iwv = this->wv.begin();
    auto ibm = this->bm.begin();
    auto ibv = this->bv.begin();
    while (iw != this->w.end())
    {
        iwm->mul(alpha);
        imw->mul((1.0 - alpha) / count);
        iwm->add(*imw);
        imw->mul(1.0 / (1.0 - alpha));
        imw->map(Square);
        iwv->mul(beta);
        imw->mul(1.0 - beta);
        iwv->add(*imw);
        imw->get(*iwv);
        imw->map(Sqrt);
        imw->add(epsilon);
        imw->map(Inv);
        imw->mul(*iwm);
        imw->mul(e * (1.0 - bt) / (1.0 - at));
        iw->sub(*imw);
        imw->set(0.0);
        ++iw;
        ++imw;
        ++iwm;
        ++iwv;
    }
    while (ib != this->b.end())
    {
        ibm->mul(alpha);
        imb->mul((1.0 - alpha) / count);
        ibm->add(*imb);
        imb->mul(1.0 / (1.0 - alpha));
        imb->map(Square);
        ibv->mul(beta);
        imb->mul(1.0 - beta);
        ibv->add(*imb);
        imb->get(*ibv);
        imb->map(Sqrt);
        imb->add(epsilon);
        imb->map(Inv);
        imb->mul(*ibm);
        imb->mul(e * (1.0 - bt) / (1.0 - at));
        ib->sub(*imb);
        imb->set(0.0);
        ++ib;
        ++imb;
        ++ibm;
        ++ibv;
    }
    auto iwa = this->wa.begin();
    auto imwa = this->mwa.begin();
    auto iwam = this->wam.begin();
    auto iwav = this->wav.begin();
    while (iwa != this->wa.end())
    {
        imwa->mul(alpha);
        iwam->mul((1.0 - alpha) / count);
        iwam->add(*imwa);
        imwa->mul(1.0 / (1.0 - alpha));
        imwa->map(Square);
        iwav->mul(beta);
        imwa->mul(1.0 - beta);
        iwav->add(*imwa);
        imwa->get(*iwav);
        imwa->map(Sqrt);
        imwa->add(epsilon);
        imwa->map(Inv);
        imwa->mul(*iwam);
        imwa->mul(e * (1.0 - bt) / (1.0 - at));
        iwa->sub(*imwa);
        imwa->set(0.0);
        ++iwa;
        ++imwa;
        ++iwam;
        ++iwav;
    }
    auto iba = this->ba.begin();
    auto imba = this->mba.begin();
    auto ibam = this->bam.begin();
    auto ibav = this->bav.begin();
    while (iba != this->ba.end())
    {
        imba->mul(alpha);
        ibam->mul((1.0 - alpha) / count);
        ibam->add(*imba);
        imba->mul(1.0 / (1.0 - alpha));
        imba->map(Square);
        ibav->mul(beta);
        imba->mul(1.0 - beta);
        ibav->add(*imba);
        imba->get(*ibav);
        imba->map(Sqrt);
        imba->add(epsilon);
        imba->map(Inv);
        imba->mul(*ibam);
        imba->mul(e * (1.0 - bt) / (1.0 - at));
        iba->sub(*imba);
        imba->set(0.0);
        ++iba;
        ++imba;
        ++ibam;
        ++ibav;
    }
    for (int i = 0; i < 5; ++i)
    {
        this->before[i].clear();
        this->after[i].clear();
    }
    while (this->cell.size() > 1)
    {
        this->cell.pop_back();
        this->hide.pop_back();
    }
    for (int i = 0; i < 8; ++i)
    {
        while (this->difw[i].size() > 1)
        {
            this->difw[i].pop_back();
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        while (this->difb[i].size() > 1)
        {
            this->difb[i].pop_back();
        }
    }
    this->data.clear();
    count = 0.0;
}

template <typename matrix, typename vector>
inline void LSTM_sen<matrix, vector>::clear()
{
    for (int i = 0; i < 5; ++i)
    {
        this->before[i].clear();
        this->after[i].clear();
    }
    while (this->cell.size() > 1)
    {
        this->cell.pop_back();
        this->hide.pop_back();
    }
    for (int i = 0; i < 8; ++i)
    {
        while (this->difw[i].size() > 1)
        {
            this->difw[i].pop_back();
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        while (this->difb[i].size() > 1)
        {
            this->difb[i].pop_back();
        }
    }
    this->data.clear();
}