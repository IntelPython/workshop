import numpy as np
import numexpr as ne
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


def black_scholes(price, strike, t, rate, vol ):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    call = ne.evaluate("P * (0.5 + 0.5 * erf((log(P / S) - T * mr + 0.25 * T * sig_sig_two) * 1/sqrt(T * sig_sig_two))) - S * exp(T * mr) * (0.5 + 0.5 * erf((log(P / S) - T * mr - 0.25 * T * sig_sig_two) * 1/sqrt(T * sig_sig_two))) ")
    put = ne.evaluate("call - P + S * exp(T * mr) ")

    return call, put

#ne.set_vml_num_threads(ne.detect_number_of_cores())
ne.set_num_threads(ne.detect_number_of_cores())
ne.set_vml_accuracy_mode('high')


if __name__ == '__main__':
    black_scholes(price, strike, t, RISK_FREE, VOLATILITY)
