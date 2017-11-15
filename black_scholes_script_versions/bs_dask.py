import numpy as np
import dask
import dask.multiprocessing
import dask.array as da

from dask.array import log, sqrt, exp
from numpy import erf, invsqrt
# ------------------------------------
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
# RISK_FREE = np.float32(0.1)
# VOLATILITY = np.float32(0.2)
# C10 = np.float32(1.)
# C05 = np.float32(.5)
# C025 = np.float32(.25)
TEST_ARRAY_LENGTH = 1024

###############################################

def gen_data(nopt):
    return (
        rnd.uniform(S0L, S0H, nopt),
        rnd.uniform(XL, XH, nopt),
        rnd.uniform(TL, TH, nopt),
        )

nopt=100000
chunk=2000
price, strike, t = gen_data(nopt)
price = da.from_array(price, chunks=(chunk,), name=False)
strike = da.from_array(strike, chunks=(chunk,), name=False)
t = da.from_array(t, chunks=(chunk,), name=False)


from numpy import log, exp, erf, invsqrt

def black_scholes_numpy_mod ( nopt, price, strike, t, rate, vol ):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

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

    call = P * d1 - Se * d2
    put = call - P + Se

    return np.stack((call, put))

def black_scholes_dask ( nopt, price, strike, t, rate, vol, schd=None ):
    return da.map_blocks( black_scholes_numpy_mod, nopt, price, strike, t, rate, vol, new_axis=0 ).compute(get = schd)


if __name__ == '__main__':
    black_scholes(nopt, price, strike, t, RISK_FREE, VOLATILITY)
