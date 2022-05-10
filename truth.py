
def truthTable(base:list,comp:list)->list:
    bucket = zip(base,comp)
    basin = []
    for i,j in bucket:
        if j == True or j == 'True':
            basin.append(i)
    return basin 

import pandas as pd
def targetTypechecker(data)->bool:
    'returns False if the data is not numerical'
    try:
        int(data[0])
    except (ValueError,TypeError):
        return False
    else:
        return True