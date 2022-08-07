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
    threshold = int(threshold)
    if len(p) + len(n)  < threshold:
        raise ValueError(f"The sum of the lengths of the entries must be equal to or greater than the threshold {threshold}")
    p = sorted(p, reverse=True)
    n = sorted(n, )
    whole = []
    half_t = int(threshold/2)
    if len(p)>= int(threshold/2) and len(n) >= int(threshold/2):
        whole.extend(p[:half_t])
        whole.extend(n[:half_t])
        return whole
    else:
        if len(p) > half_t:
            a = 'p'
        elif len(n) >half_t:
            a = 'n'
        if a == 'p':
            tn = threshold-len(n)
            whole.extend(n)
            whole.extend(p[:tn])
        elif a =='n':
            tp = threshold-len(p)
            whole.extend(p)
            whole.extend(n[:tp])
        assert len(whole) == threshold
        return whole 
            
            