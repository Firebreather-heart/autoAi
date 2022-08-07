def split_by_sign(iterable:list):
    """
    Takes a given list and splits it into positive an negative values
    """
    s1,s2 = [i for i in iterable if i>0 ],[j for j  in iterable if j<0]
    return s1,s2 

def fill_to_threshold(p:list, n:list, threshold:int):
    """
        return a list filled with values gotten from the highest 
        of the two input lists as dictated by the threshold
    """
    if len(p) + len(n)  < threshold:
        raise ValueError(f"The sum of the lengths of the entries must be equal to or greater than the threshold {threshold}")
    p = sorted(p, reverse=True)
    n = sorted(n, )
    whole = []
    if len(p)>= int(threshold/2) and len(n) >= int(threshold/2):
        whole.extend(p[:threshold/2])
        whole.extend(n[:threshold/2])
    else:
        if len(p) > threshold/2:
            a = 'p'
        elif len(n) >threshold/2:
            a = 'n'
        if a == 'p':
            whole.extend(n)
            whole.extend(p[:(threshold-len(n))])
        elif a =='n':
            whole.extend(p)
            whole.extend(n[:(threshold-len(p))])
        assert len(whole) == threshold
        return whole 
            
            