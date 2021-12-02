import random
from datetime import datetime
random.seed(datetime.now())

def takeSecond(elem):
    return elem[0]

# random list
random1 = [(2, 2), (3, 4), (4, 1), (1, 3)]

# sort list with key
random1.sort(key=takeSecond)

# print list
print('Sorted list:', random1)