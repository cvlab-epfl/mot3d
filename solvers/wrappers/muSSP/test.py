#import wrapper
import mot3d
help(mot3d)

graph = [
    "p min 28 62",
    "c ------ source->pre-nodes ------",
    "a 1 2 0.0100000",
    "a 1 4 0.0100000",
    "a 1 6 0.0100000",
    "a 1 8 0.0100000",
    "a 1 10 0.0100000",
    "a 1 12 0.0100000",
    "a 1 14 0.0100000",
    "a 1 16 0.0100000",
    "a 1 18 0.0100000",
    "a 1 20 0.0100000",
    "a 1 22 0.0100000",
    "a 1 24 0.0100000",
    "a 1 26 0.0100000",
    "c ------ post-nodes->sink ------",
    "a 3 28 0.0000000",
    "a 5 28 0.0000000",
    "a 7 28 0.0000000",
    "a 9 28 0.0000000",
    "a 11 28 0.0000000",
    "a 13 28 0.0000000",
    "a 15 28 0.0000000",
    "a 17 28 0.0000000",
    "a 19 28 0.0000000",
    "a 21 28 0.0000000",
    "a 23 28 0.0000000",
    "a 25 28 0.0000000",
    "a 27 28 0.0000000",
    "c ------ pre-node->post-nodes ------",
    "a 2 3 -0.5000000",
    "a 4 5 -0.5000000",
    "a 6 7 -0.5000000",
    "a 8 9 -0.5000000",
    "a 10 11 -0.5000000",
    "a 12 13 -0.5000000",
    "a 14 15 -0.5000000",
    "a 16 17 -0.5000000",
    "a 18 19 -0.5000000",
    "a 20 21 -0.5000000",
    "a 22 23 -0.5000000",
    "a 24 25 -0.5000000",
    "a 26 27 -0.5000000",
    "c ------ post-node->pre-nodes ------",
    "a 3 4 -0.0000096",
    "a 3 6 -0.0000000",
    "a 5 8 -0.0000000",
    "a 5 6 -0.0000034",
    "a 7 8 -0.0002076",
    "a 7 10 -0.0000000",
    "a 9 12 -0.0000000",
    "a 9 10 -0.0017129",
    "a 11 12 -0.0000897",
    "a 11 14 -0.0000000",
    "a 13 16 -0.0000000",
    "a 13 14 -0.0000842",
    "a 15 16 -0.0000044",
    "a 15 18 -0.0000000",
    "a 17 20 -0.0000000",
    "a 17 18 -0.0007890",
    "a 19 20 -0.0000776",
    "a 19 22 -0.0000000",
    "a 21 24 -0.0000000",
    "a 21 22 -0.0000013",
    "a 23 24 -0.0000914",
    "a 23 26 -0.0000000",
    "a 25 26 -0.0004939"
]

verbose = 2
output = wrapper.solve(graph, verbose)

print("----- output ----")
print(output)