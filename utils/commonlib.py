def streval(i_):
    if type(i_) == str:
        ri = eval(i_)
    else:
        ri = i_
    return ri

def first_caps(str1):
    """
    capitalises the first character of a string sentence.
    """
    return str1[0].upper() + str1[1:]

def flatten(t):
    """
    flattens a list of list into a list
    """
    return [item for sublist in t for item in sublist]

def unique(lst):
    """
    returns the unique elements of a list
    """
    set1 = set(lst)
    lst = list(set1)
    return lst