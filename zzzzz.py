def topSort(graph):
    in_degrees = dict((u, 0) for u in graph)
    num = len(in_degrees)
