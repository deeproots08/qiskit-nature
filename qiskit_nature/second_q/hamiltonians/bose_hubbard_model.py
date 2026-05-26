# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Bose-Hubbard model"""
import numpy as np

from qiskit_nature.second_q.operators import BosonicOp

from .lattice_model import LatticeModel
from .lattices import Lattice

class BoseHubbardModel(LatticeModel):

    r"""The Bose-Hubbard model.

    This class implements the following Hamiltonian:

    .. math::
        H = \sum_{i, j} t_{i, j} b_i^\dagger b_j
            + \frac{U}{2} \sum_i n_i (n_i - 1),

    where :math:`b_i^\dagger` and :math:`b_i` are creation and annihilation
    operators of a boson at site :math:`i`. The operator :math:`n_i` is the
    number operator, defined by :math:`n_i = b_i^\dagger b_i`. The matrix
    :math:`t_{i, j}` is a Hermitian matrix describing hopping between lattice
    sites, :math:`U` is the strength of the on-site interaction.

    This model describes interacting bosons on a lattice and reduces to a
    non-interacting system when :math:`U = 0`.

    This model is instantiated using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.Lattice`. For example, using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.LineLattice`:

    .. code-block:: python

        line_lattice = LineLattice(num_nodes=10, boundary_condition=BoundaryCondition.OPEN)

        fermi_hubbard_model = BoseHubbardModel(
            line_lattice.uniform_parameters(
                uniform_interaction=-1.0,
                uniform_onsite_potential=0.0,
            ),
            onsite_interaction=5.0,
        )
    """


    def __init__(self, lattice: Lattice, onsite_interaction: complex) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
            onsite_interaction: The strength of the on-site interaction.
        """
        super().__init__(lattice)
        self._onsite_interaction = onsite_interaction

    def hopping_matrix(self) -> np.ndarray:
        """Return the hopping matrix."""
        return self.interaction_matrix()
    
    @property
    def register_length(self) -> int:
        return self._lattice.num_nodes

    
