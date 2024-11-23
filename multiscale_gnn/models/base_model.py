from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable

import haiku as hk
import jax.numpy as jnp

# TODO: Implement base multiscale GNN class (interface)
# Typical inputs to the BaseModel specified in LagrangeBench are of type (feature_dict, particle_types)
# feature_dict may contain edge-level features too (such as relative displacement, relative distances). in particular, these are used by the EGNN baseline. We might want to discuss if it is relevant having hierarchies on edges as well, or rather edge-level features for a supergraph.

# In our case, we probably need 2 input graphs with the actual super-graph generation as a separate process. We also should have some type of mapping between these graphs as an input so we can have the bi-level aggregation step. given a super-graph with k nodes, let's just say that this is a callable function.

class MultiscaleBaseModel(hk.Module, ABC):
    """
    Base MultiScale model class.
    """
    @abstractmethod
    def __call__(
        self, 
        graph_sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray],
        supergraph_sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray],
        graph_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass.

        We specify the dimensions of the inputs and outputs using the number of nodes N,
        the number of edges E, number of historic velocities K (=input_seq_length - 1),
        and the dimensionality of the feature vectors dim.

        Args:
            graph_sample: Data for original graph. Tuple with feature dictionary and particle type. Possible features

                - "abs_pos" (N, K+1, dim), absolute positions
                - "vel_hist" (N, K*dim), historical velocity sequence
                - "vel_mag" (N,), velocity magnitudes
                - "bound" (N, 2*dim), distance to boundaries
                - "force" (N, dim), external force field
                - "rel_disp" (E, dim), relative displacement vectors
                - "rel_dist" (E, 1), relative distances, i.e. magnitude of displacements
                - "senders" (E), sender indices
                - "receivers" (E), receiver indices

            supergraph_sample: Data for supergraph. Tuple with feature dictionary and particle type, with same features as graph_sample.

            graph_map: Some function that inputs indices for the original graph and embeddings for the supergraph, and returns the expanded embeddings for original graph nodes.

        Returns:
            Dict with model output.
            The keys must be at least one of the following:

                - "acc" (N, dim), (normalized) acceleration
                - "vel" (N, dim), (normalized) velocity
                - "pos" (N, dim), (absolute) next position
        """
        raise NotImplementedError