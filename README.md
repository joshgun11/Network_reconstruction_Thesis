# Network Reconstruction Using Deep Learning and Sensitivty Analysis - (NSNR - Neural Sensitivity based Network Reconstruction)
There are many natural processes that can be represented as dynamic systems, and the components of these systems are connected to each other with some un- derlying network structure. Understanding the behavior of such systems requires knowledge of the underlying interactions and network structure, which is often difficult to observe directly. In this thesis, we use deep learning to address the network reconstruction problem. Without relying on many parametric assumptions about the network structure and dynamics. We train node-specific MLP (multi-layer perceptron) models to learn the behavior of each node. Assuming that the direct neighbors of each node have the most impact on its future behavior. Using sensitivity analysis, we identify the most important nodes for each node and reconstruct the network without any ground truth information. Unlike the many traditional (statisti- cal) network reconstruction methods, our approach can reconstruct networks with non-linear dynamics, find edges between heterogenous nodes (nodes do not behave similarly), can distinguish direct and indirect effects in sufficient amounts.
You can find the pdf version of the thesis in `thesis_main.pdf`
<div style="display:flex">
    <img src="https://user-images.githubusercontent.com/77841418/236284144-5cc15690-3bc0-4a45-a134-1df3d7b443b7.png" width="500" height="300">
    <img src="https://user-images.githubusercontent.com/77841418/236284430-a96e25eb-2517-4578-a29f-46f1be36ae52.png" width="500" height="300">
</div>



## Usage:
-  Clone the repository `git clone https://github.com/joshgun11/Network_reconstruction_Thesis.git`
-  Navigate to the repository directory in your terminal `cd Network_reconstruction_Thesis`
-  Create a virtual environment on your machine `python -m venv example_env`
-  Activate the environment `source env/bin/activate`
-  Install the required packages `pip install -r requirements.txt`
-  If the example datasets are zipped, you should unzip the files to be accessible by NSNR. Example data should be in a path `datasets\dynamics\example_data`

Then:

## RECONSTRUCTION:
You can run the following command in the terminal:

 `python reconstruct_graph.py --problem_type classification --num_classes 2 --node_size 9 --method input_change --model MLP --epochs 100 --dynamics voter --graph grid --data_size 10000 --experiment_name example --data grid_10_voter_10000_5.pickle --file_name example`

to run the experiment for reconstructing the 10 (9) nodes grid graph with voter dynamics by using MLP models and input permutation SA. You can look for the other options as `LSTM` for the model , or other two possible SA methods from the `parsers.py` file. If you are using `LSTM` as the predictor, only input permutation SA is available, which is named as `input_change`. You need to specify `--problem_type` argument as `classification` for the binary and discrete dynamical models (Voter, SIS, Game of life, RPS, Forest fire) and `regression` for CML. Also you should define `--num_classes` argument 2 for the SIS, Voter, and Game of life, 3 for RPS and Forest fire, 1 for the CML. 

Note that NSNR is not using ground truth graph and dynamics. They are arguments just to reach to the data as data is located in path `datasets\args.dynamics\args.data`, and return plots for comparison at the end. By defining `--file_name` argument we write the name of `csv` file that we store our results (metrics). 

Example dataset is named as `grid_10_voter_10000_5.pickle` where `grid` shows the graph type, `10` is number of nodes, `voter` is dynamics, `10000` is the size of data, and `5` is sampling rate. You can re-run the experiments based on this formulation with the desired dataset. 

`trained_models` folder are containing the trained models, where if you want to use a specific data for experiment and if there are trained models for this data set, process will start from the SA (sensitivty analysis) part. You can delete the trained model of the specific data set if you want to train new models. In both cases, NSNR will return a weighted adjacency matrix (`scores.npy`), ground truth adjacency matrix (`original_adj_matrix.npy`) and predicted adjacency matrix (`predicted_adj_matrix.npy`) as numpy array, and it will be stored in a `results` folder together with the average `loss` and `accuracy` plots, and `ground truth` and `predicted` network plots. If you wish to get feature importance plots, clustering plots, heatmaps and distance plots, you have to add argument `--plot yes` to the command above. 

If you want to run multiple experiments, you can use the `experiment.py` file. You need to define a name of csv file where the metrics (name, Acc, Tp, Fp, Loss, Run time) of the experiment will be stored. And other details should be entered to the lists.  
You can find the results of experiments mentioned in thesis from the `Experiment_results` folder. Baselines methods have been applied in a `Baselines.ipynb` notebook. 

## DATA GENERATION:

You can generate a new data by running `python data_generation.py --dynamics voter --graph erdos --node_size 10 --data_size 5000 --every_x_step 5 ` as an example which will create a data with voter dynamics on 10 node erdos graph with the size of 5000 and sampling rate 5, in a path `datasets\voter\erdos_10_voter_5000_5.pickle`. You can look at the `parser.py` file to see the all arguments and their default values. 

You should un-zip the datasets to use example ready datasets which are used in thesis (Not all are there because of storage, but comparison with baselines dataset mainly)




