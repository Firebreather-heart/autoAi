def approximator(a)->int:
    if a >= 0.5:
        return 1
    else:
        return 0

def fastForward(num:int)->int:
    threshold = False
    req_pow = 2
    while threshold is False:
        if num > req_pow:
            req_pow *= 2
            rem = num - req_pow
        if num < req_pow and req_pow < num*2:
            threshold = True
    return req_pow
def breakDown(num:int) -> int:
    return num/256

