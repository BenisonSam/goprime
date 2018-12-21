from goprime.common import static_class


@static_class
class Constants(object):

    def __init__(self):
        raise TypeError("Constants: Is a Static class. Object not allowed to be created!")

    PUCT_C = 0.1
    N_SIMS = 1600
    TEMPERATURE = 2
    PROPORTIONAL_STAGE = 3
    EXPAND_VISITS = 1
    RAVE_EQUIV = 100
    REPORT_PERIOD = int(N_SIMS / 4)
    PRIOR_NET = 40
    PRIOR_EVEN = 4  # should be even number; 0.5 prior
    P_ALLOW_RESIGN = 0.8
    RESIGN_THRES = 0.025
