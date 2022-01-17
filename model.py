import torch
import torch.nn as nn

class VRPNet(nn.Module):
    def __init__(self, device, 
                 n_iters=10, 
                 n_node_features=128, 
                 n_node_inputs=3, 
                 n_msg_features=10,
                 n_edge_features=64, 
                 n_edge_inputs=1,
                 n_edge_outputs=2, 
                 hidden_dims_msg=(96, 96),
                 hidden_dims_output=(96, 96)):
        """
        Args:
          n_iters: Number of graph iterations.
          n_node_features: Number of features in the states of each node.
          n_node_inputs: Number of input features to each graph node.
          n_msg_features: Number of features in the final layer of the message function.
          n_edge_features: Number of features in the messages sent along the edges of the graph (produced
              by the message network).
          n_edge_inputs: Number of input features to each edge.
          n_edge_outputs: Number of output features for each edge.
          hidden_dims_msg: Number of features in the first two layers of the message function.
          hidden_dims_output: Number of features in the first two layers of the edge output function.
        """
        super(VRPNet, self).__init__()
        
        self.device = device
        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_msg_features = n_msg_features
        self.n_edge_features = n_edge_features
        self.n_edge_inputs = n_edge_inputs
        self.n_edge_outputs = n_edge_outputs
        

        self.msg_net = nn.Sequential(
            nn.Linear(2*n_node_features+n_edge_inputs, hidden_dims_msg[0]),
            nn.BatchNorm1d(hidden_dims_msg[0]),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dims_msg[0], hidden_dims_msg[1]),
            nn.BatchNorm1d(hidden_dims_msg[1]),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dims_msg[1], n_msg_features)
        )
        
        self.gru_nodes = nn.GRU(input_size=n_msg_features+self.n_node_inputs, hidden_size=n_node_features)
        self.gru_edges = nn.GRU(input_size=2*n_node_features+self.n_edge_inputs, hidden_size=n_edge_features)
        
        
        self.output = nn.Sequential(
            nn.Linear(n_edge_features, hidden_dims_output[0]),
            nn.BatchNorm1d(hidden_dims_output[0]),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dims_output[0], hidden_dims_output[1]),
            nn.BatchNorm1d(hidden_dims_output[1]),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dims_output[1], n_edge_outputs)
        )


    def forward(self, node_inputs, edge_inputs, src_ids, dst_ids):
        """
        Args:
          node_inputs of shape (batch_size, n_nodes_per_batch, n_node_inputs): Tensor of inputs to every node of the graph.
          edge_inputs of shape (n_edges, 1): Tensor of weights for every edge of the graph.
          src_ids of shape (n_edges): Indices of source nodes of every edge.
          dst_ids of shape (n_edges): Indices of destination nodes of every edge.
          
        Returns:
          outputs of shape (n_iters, n_nodes,, n_nodes_per_batch, n_node_outputs): Outputs of all the edge at every iteration.
        """
        
        batch_size, n_nodes_per_batch, _ = node_inputs.shape
        n_nodes = batch_size * n_nodes_per_batch
        n_edges = edge_inputs.shape[0]
        
        node_inputs = node_inputs.flatten(start_dim=0, end_dim=1)

        node_states = torch.zeros(1, n_nodes, self.n_node_features).to(self.device)
        edge_states = torch.zeros(1, n_edges, self.n_edge_features).to(self.device)
        
        output = torch.zeros(self.n_iters,  
                             n_nodes, 
                             n_nodes_per_batch, 
                             self.n_edge_outputs).to(self.device)

        output[:, :, :, 0] = 1
        output[:, :, :, 1] = -10

        
        for i in range(self.n_iters):
            src_states = node_states[0, src_ids, :]
            dst_states = node_states[0, dst_ids, :]

            msg_inp = torch.cat([src_states, dst_states, edge_inputs], dim=1)
            
            msg = self.msg_net(msg_inp)

            agg_msg = torch.zeros(n_nodes, self.n_msg_features).to(self.device)
            agg_msg.index_add_(0, dst_ids, msg)
            
            gru_nodes_inp = torch.cat((agg_msg, node_inputs), dim=1).unsqueeze(0)
            
            gru_nodes_out, node_states = self.gru_nodes(gru_nodes_inp, node_states)
            
            src_states = node_states[0, src_ids, :]
            dst_states = node_states[0, dst_ids, :]

            gru_edges_inp = torch.cat([src_states, dst_states, edge_inputs], dim=1).unsqueeze(0)

            gru_out, edge_states = self.gru_edges(gru_edges_inp, edge_states)

            iter_out = self.output(gru_out[0])
            
            output[i, src_ids, torch.fmod(dst_ids, n_nodes_per_batch)] = iter_out

        
        return output