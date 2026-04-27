"""Passthrough iterator wrapper for RINO-normalized features.

The standard IteratorWrapper applies coordinate transforms and edge
construction. For RINO data, features are already pre-normalized so
we just pass through, adding empty edges/adjmat for interface compat.
"""

import numpy as np


class RINOIteratorWrapper:
    """Wrapper that passes pre-normalized RINO features through unchanged.

    Produces the 6-tuple (edges, nodes, high, adjmat, mask, label)
    expected by Bert's _shared_step.
    """

    def __init__(
        self,
        base_iterator,
        augmentation_list=None,
        augmentation_prob=1.0,
        coordinates=None,
        del_r_edges=0,
        boost_mopt=0,
    ):
        self.base_iterator = base_iterator
        self.n_classes = getattr(base_iterator, "n_classes", 10)
        if not hasattr(self.base_iterator, "n_classes"):
            self.n_classes = base_iterator.get_nclasses()

    def __next__(self):
        result = next(self.base_iterator)
        # RINOH5Iterator yields a 5-tuple when precomputed code_labels are
        # available in a sibling *_tokens.h5 file, else a 4-tuple.
        if len(result) == 5:
            high, nodes, mask, label, code_labels = result
        else:
            high, nodes, mask, label = result
            code_labels = None
        # Bert expects (edges, nodes, high, adjmat, mask, label) normally;
        # when precomputed tokens are present, we pass a 7-tuple and
        # IterableBert.preprocess_inputs handles the extra field.
        n_csts = nodes.shape[0]
        edges = np.empty((0, 0), dtype=np.float32)
        adjmat = np.zeros((n_csts, n_csts), dtype=bool)
        if code_labels is not None:
            return edges, nodes, high, adjmat, mask, label, code_labels
        return edges, nodes, high, adjmat, mask, label
