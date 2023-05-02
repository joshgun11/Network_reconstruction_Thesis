from data_generation import Generate_Data

data_generator = Generate_Data()

from parsers import KParseArgs
import sys

parser = KParseArgs()
args = parser.parse_args()
flag = len(sys.argv) == 1


args.dynamics == 'gol'
args.data_size = 10000
node_size = [10]
graphs = ['erdos']

for i in range(len(graphs)):
    args.graph = graphs[i]
    for j in range(len(node_size)):
        args.node_size = node_size[j]
        args.r = 3.8
        data_generator.gol(args)

