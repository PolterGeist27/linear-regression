

def normalize(s):
    return [((_ - min(s)) / (max(s) - min(s))) for _ in s]

def normalizeElem(list, elem):
    return ((elem - min(list)) / (max(list) - min(list)))

def denormalizeElem(list, elem):
    return ((elem * (max(list) - min(list))) + min(list))