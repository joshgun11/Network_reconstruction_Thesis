from data_generation import Generate_Data
from parsers import KParseArgs
import sys

data_generator = Generate_Data()
parser = KParseArgs()
args = parser.parse_args()
flag = len(sys.argv) == 1

''' Simple example for generating data manually, you can generate a data with different 
node sizes on different graphs at the same time'''

args.dynamics == 'rps'
args.data_size = 500
node_size = [5]
graphs = ['erdos']

for i in range(len(graphs)):
    args.graph = graphs[i]
    for j in range(len(node_size)):
        args.node_size = node_size[j]
        args.r = 3.8
        data_generator.rps(args)

