# Takes nested dictionary strings and prints a connection list for online
# graph visualizers

import ast

A = input("Input Graph String: ")
A = ast.literal_eval(A)

for a in A.keys():
    for b in A[a]:
        print(a, '->', b, ';')
