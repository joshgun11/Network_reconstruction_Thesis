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
You can run the following command in terminal: `python reconstruct_graph.py --problem_type classification --num_classes 2 --node_size 10 --method input_change --model MLP --epochs 100 --dynamics voter --graph grid --data_size 10000 --experiment_name example --data grid_10_voter_10000_5.pickle`
to run experiment for reconstructing the 10 (9) nodes grid graph with voter dynamics. Note than NSNR is not using ground truth graph and dynamics. they are arguments just to reach to the data as data is located in path `datasets\args.dynamics\args.graph\args.data`
