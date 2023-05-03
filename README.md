# NSNR - Neural Sensitivty based Network Reconstruction
There are many natural processes that can be represented as dynamic systems
operating on networks of individual components, such as the spread of viruses in
social networks or the propagation of electrical signals in the brain. Understanding
the behavior of such systems requires knowledge of the underlying interaction
structure, which is often difficult to observe directly. In this thesis, we use deep
learning to address the network reconstruction problem. Without relying on many
parametric assumptions about the network structure and dynamics. We train node-
specific MLP (multi-layer perceptron) models to learn the behavior of each node.
Assuming that the direct neighbors of each node have the most impact on its future
behavior. Using sensitivity analysis, we identify the most important nodes for each
node and reconstruct the network without any ground truth information. Unlike
the many traditional (statistical) network reconstruction methods, our approach
can reconstruct networks with highly non-linear dynamics, find edges between
heterogenous nodes (nodes do not behave similarly), can distinguish direct and
indirect effects in sufficient amounts. We tested our method on various dynamical
models and networks, with high accuracy. Despite the fact that our approach
successfully reconstructs networks with high accuracy, scalability is a major challenge
due to the time complexity and complexity of interactions on larger networks.
Overall, this work contributes to the establishment of a prediction-based machine
learning approach as a valid alternative to traditional statistical methods for network
reconstruction. See the example reconstructed network:
![input_change_MLP_erdos_50_50000_reconstructed](https://user-images.githubusercontent.com/77841418/235725943-a859971b-d42d-4e55-ad2b-8f6343383e49.png)

You can run the following command in terminal:
 `python reconstruct_graph.py --problem_type classification --num_classes 2 --node_size 10 --method input_change --model MLP --epochs 100 --dynamics voter --graph grid --data_size 10000 --experiment_name example --data grid_10_voter_10000_5.pickle`
to run experiment for reconstructing the 10 (9) nodes grid graph with voter dynamics. Note than NSNR is not using ground truth graph and dynamics. they are arguments just to reach to the data as data is located in path `datasets\args.dynamics\args.data`. 

Sample data set is named as `grid_10_voter_10000_5.pickle` where `grid` shows the graph type, `10` is number of nodes, `voter` is dynamics, `10000` is the size of data, and `5` is sampling rate. You can re-run the experiments based on this fromulation. 

`trained_models` folder are containing the trained models, where if you want to use a specific data for experiment and if there are trained models for this data set, process will start from the SA (sensitivty analysis) part. You can delete the trained model of the specific data set if you want to train new models. In both cases, NSNR will return a weighted adjacency matrix (`scores.npy`), ground truth adjacency matrix (`original.npy`) and predicted adjacency matrix (`predicted.npy`) as numpy array, and it will be stored in a `results` folder together with the average `loss` and `accuracy` plots, and `ground truth` and `predicted` network plots. If you wish to get feture importance plots, clustering plots, heatmaps and distance plots, you have to add argument `--plot True` to the command above. If you want to run multiple experiments, you can use the `experiment.py` file. You need to define a name of csv file where the metrics (name, Acc, Tp, Fp, Loss, Run time) of the experiment will be stored. 

You can generate a new data by running `python data_generation.py --dynamics voter --graph erdos --node_size 10 --data_size 5000 --every_x_step 5 ` as an example which will create a data with voter dynamics on 10 node erdos graph with the size of 5000 and sampling rate 5, in a path `datasets\voter\erdos_10_voter_5000_5.pickle`. You can look at the `parser.py` file to see the all arguments and their default values. 


