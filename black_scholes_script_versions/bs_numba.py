#Boilerplate for the example
import numpy as np
import numba as nb
from math import log, sqrt, exp, erf

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
call = np.zeros(nopt, dtype=np.float64)
put  = -np.ones(nopt, dtype=np.float64)


def black_scholes_numba_opt(price, strike, t, mr, sig_sig_two):
        P = price
        S = strike
        T = t

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = 1./sqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        r  = P * d1 - Se * d2
        return complex(r, r - P + Se)

black_scholes_numba_opt_vec = nb.vectorize('c16(f8,f8,f8,f8,f8)', target="parallel")(black_scholes_numba_opt)

@nb.jit
def black_scholes(nopt, price, strike, t, rate, vol):
    sig_sig_two = vol*vol*2
    mr = -rate
    black_scholes_numba_opt_vec(price, strike, t, mr, sig_sig_two)


if __name__ == '__main__':
    black_scholes(nopt, price, strike, t, RISK_FREE, VOLATILITY)
