import random
import scipy.stats as stats


class Passenger:
    """Passenger class"""

    def __init__(self, id, origin, destination, request_time, price, assign_time=None, wait_time=0, choice=None, max_wait=5) -> None:
        """
        price: price set for the trip
        choice: choice model for passenger
        max_wait: maximum waiting time
        """
        self.id = id
        self.origin = origin
        self.destination = destination
        self.request_time = request_time
        self.price = price
        self.assign_time = assign_time
        self.wait_time = wait_time
        self.choice = choice
        self.max_wait = max_wait

    def unmatched_update(self):
        """Update state of passenger if not matched. Return True if maximum waiting time is reached otherwise False."""

        self.wait_time += 1
        if self.wait_time >= self.max_wait:
            return True
        else:
            return False

    def match(self, timestamp):
        """Update state of passenger once get matched. Return True if the passenger accept the price otherwise False."""
        accept = choice_passenger(self.price, self.choice)
        if accept:
            self.assign_time = timestamp
            return True
        else:
            return False

def choice_passenger(price, mtype=None):
    """Choice model for passenger. Return 1 if accept else return 0."""
    if mtype is None:
        # Use default exponential disteibution
        # reject_prob = stats.expon.cdf(price, scale=1/2)
        reject_prob = 0
        sample = random.uniform(0,1)
        if sample < reject_prob:
            return 0
        else:
            return 1


def generate_passenger(demand, arrivals=None):
    """
    Generate passenger according to the demand

    demand: (origin,destination,time,total demand,price)
    arrivals: number of passengers already arrive in the system

    return: list of new passengers, total number of passenger arrivals
    """
    newp = []
    ori, des, t, d, p = demand
    for i in range(d):
        if arrivals is None:
            newp.append(Passenger(i, ori, des, t, p))
        else:
            newp.append(Passenger(arrivals+1, ori, des, t, p))
            arrivals += 1

    return newp, arrivals
