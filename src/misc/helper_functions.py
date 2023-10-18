


def demand_update(d_ori, p, ph, pf):
    """ 
    Update demand by price

    d_ori: Original demand
    p: price
    ph: maximum price
    pf: reference price.
    
    """
    return round(d_ori*(ph-p)/(ph-pf))