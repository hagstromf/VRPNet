import numpy as np
import torch


class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.
    References:
        For TSP: https://github.com/chaitjo/graph-convnet-tsp/blob/master/utils/beamsearch.py
    """

    def __init__(self, beam_size, batch_size, num_nodes, num_vehicles, device,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, random_start=False):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.device = device
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong).to(self.device)
        if random_start == True:
            # Random starting nodes
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong).to(self.device)
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes).type(self.dtypeFloat).to(self.device)

        if self.num_vehicles > 1:
            # Counts how many times the depot has been visited during the search. When it has been  
            # visited num_vehicles times the depot should also remain masked.
            self.depot_mask_counter = torch.zeros(batch_size, beam_size).type(self.dtypeFloat).to(self.device)
            self.ones = torch.ones(batch_size, beam_size).type(self.dtypeFloat).to(self.device)

        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search

        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat).to(self.device)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        """Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state

    def get_current_origin(self):
        """Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        """Advances the beam based on transition probabilities.
        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """
        # Compound the previous scores (summing logits == multiplying probabilities)
        if len(self.prev_Ks) > 0:
            beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(self.dtypeFloat).to(self.device)

        
        beam_lk += 1e-20
        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = (bestScoresId / self.num_nodes).type(torch.long)
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)
        self.mask = self.mask.gather(1, perm_mask)

        if self.num_vehicles > 1:
            # Re-index depot mask counter
            self.depot_mask_counter = self.depot_mask_counter.gather(1, perm_mask[:, :, 0])

        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        """Sets new_nodes to zero in mask.
        """
        arr = (torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(self.dtypeLong).to(self.device))

        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        
        update = torch.eq(arr, new_nodes).type(self.dtypeFloat).to(self.device)

        update_mask = 1 - update

        self.mask = self.mask * update_mask

        if self.num_vehicles > 1:
            self.depot_mask_counter += update[:, :, 0]
            self.mask[:, :, 0] = torch.min(self.ones, self.num_vehicles - self.depot_mask_counter)

    def sort_best(self):
        """Sort the beam.
        """
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score and index of the best hypothesis in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis.
        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
        assert self.num_nodes + self.num_vehicles - 1 == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes + self.num_vehicles - 1).type(self.dtypeLong).to(self.device)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp


