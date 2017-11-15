try:
    import numpy.random_intel as rnd
except:
    import numpy.random as rnd

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

SEED = 7777777
S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2
TEST_ARRAY_LENGTH = 1024

###############################################

def gen_data(nopt):
    return (
        rnd.uniform(S0L, S0H, nopt),
        rnd.uniform(XL, XH, nopt),
        rnd.uniform(TL, TH, nopt),
        )

nopt=100000
price, strike, t = gen_data(nopt)
call = [0.0 for i in range(nopt)]
put = [-1.0 for i in range(nopt)]
price=list(price)
strike=list(strike)
t=list(t)

from math import log, sqrt, exp, erf
import numpy as np
invsqrt = lambda x: 1.0/sqrt(x)

def black_scholes ( nopt, price, strike, t, rate, vol, call, put ):
    mr = -rate
    sig_sig_two = vol * vol * 2

    for i in range(nopt):
        P = float( price [i] )
        S = strike [i]
        T = t [i]

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = invsqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        call [i] = P * d1 - Se * d2
        put [i] = call [i] - P + Se


if __name__ == '__main__':
    black_scholes(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)
