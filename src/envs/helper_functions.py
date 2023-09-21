import random
import scipy.stats as stats


def choice_passenger(price, mtype=None):
    """Choice model for passenger. Return 1 if accept else return 0."""
    if mtype is None:
        # Use default exponential disteibution
        reject_prob = stats.expon.cdf(price, scale=1/2)
        sample = random.uniform(0,1)
        if sample <= reject_prob:
            return 0
        else:
            return 1