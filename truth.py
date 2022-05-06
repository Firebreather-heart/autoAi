
def truthTable(base:list,comp:list)->list:
    bucket = zip(base,comp)
    basin = []
    for i,j in bucket:
        if j == True or j == 'True':
            basin.append(i)
    return basin 


