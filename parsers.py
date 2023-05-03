import argparse

class KParseArgs():

    def __init__(self):
        self.args = parser = argparse.ArgumentParser()

        self.parser = argparse.ArgumentParser()
        

        
        # Parametrs for model training
        self.parser.add_argument("--data", help="data_path", action='store', nargs='?', default='grid_25_voter_10000.pickle',
                            type=str)

        self.parser.add_argument("--epochs", help="epochs", action='store', nargs='?', default=100,
                            type=int)    

        self.parser.add_argument("--batch_size", help="batchs size", action='store', nargs='?', default=64,
                            type=int)    
        self.parser.add_argument("--node", help="traget node", action='store', nargs='?', default=1,
                            type=int)    
        self.parser.add_argument("--model", help="model type", action='store', nargs='?', default='MLP',
                            type=str,choices=['MLP','LSTM'])    
        self.parser.add_argument("--n_steps", help="length of sequence for LSTM", action='store', nargs='?', default=10,
                            type=int)
        self.parser.add_argument("--direction", help="directed or undirected network", action='store', nargs='?', default='undirected',
                            type=str,choices=['directed','undirected'])


        #parametrs for data generation  
        self.parser.add_argument("--every_x_step", help="graph type", action='store', nargs='?', default=5,
                            type=int)
        self.parser.add_argument("--graph", help="graph type", action='store', nargs='?', default='grid',
                            type=str,choices=['grid','erdos','geom','albert'])  
        self.parser.add_argument("--node_size", help="graph type", action='store', nargs='?', default=25,
                            type=int)
        self.parser.add_argument("--data_size", help="data size", action='store', nargs='?', default=1000,
                            type=int)
        self.parser.add_argument("--beta", help="infection rate", action='store', nargs='?', default=1.0,
                            type=float)#for SIS dynamics
        self.parser.add_argument("--mu", help="recovery rate", action='store', nargs='?', default=2.0,
                            type=float)#for SIS dynamics
        self.parser.add_argument("--dynamics", help="dynamical model", action='store', nargs='?', default='voter',
                            type=str,choices=['sis','voter','gol','rps','sis','cml'])
        self.parser.add_argument("--s", help="cml", action='store', nargs='?', default=0.2,
                            type=float)#for CML dynamics
        self.parser.add_argument("--r", help="cml", action='store', nargs='?', default=3.5,
                            type=float)#for CML dynamics

        #parametrs for Sensitivity Analaysis
        self.parser.add_argument("--method", help="analysis method", action='store', nargs='?', default='input_change',
                            type=str,choices=['input_change','gradient_based','deeplift'])
        self.parser.add_argument("--num_classes", help="number of output units", action='store', nargs='?', default=2,
                            type=int)
        self.parser.add_argument("--problem_type", help="type of learning", action='store', nargs='?', default='classification',
                            type=str,choices=['classification','regression'])   
        


        self.parser.add_argument("--plot", help="plot or not", action='store', nargs='?', default='no',
                            type=str, choices=['yes', 'no'])  
        self.parser.add_argument("--experiment_name", help="experiment ID", action='store', nargs='?', default=True,
                            type=str)        

        self.parser.add_argument("--file_name", help="file to log results", action='store', nargs='?', default='example',
                            type=str)            
         
    
    
    def parse_args(self):
        return self.parser.parse_args()

    def parse_args_list(self, args_list):
        return self.parser.parse_args(args_list)