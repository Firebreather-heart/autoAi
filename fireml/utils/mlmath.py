def approximator(a)->int:
    if a >= 0.5:
        return 1
    else:
        return 0

def fastForward(num:int)->int:
    """
    Find the smallest power of 2 that is greater than or equal to num.
    
    Args:
        num: An integer input
        
    Returns:
        The next power of 2 that is >= num
    """
    power = 1
    while power < num:
        power *= 2
    return power
    
def breakDown(num:int) -> float:
    return num/256

