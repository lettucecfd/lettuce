import string

import torch
import numpy as np

from lettuce.symmetry import SymmetryGroup

__all__= ["GConv", "GConvPermutation", "EquivariantNet", "EquivariantNeuralCollision"]


class GConv(torch.nn.Module):
    """Group Convolution Layer.
    Linear layer without bias that is equivariant with respect to a given symmetry group.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of input channels
    group_action : torch.Tensor
        Tensor of shape (group_order, dim, dim) that defines the group representation in GL(n).
        For the LBM: M * P_g * M^{-1} gives the moment action for the g-th permutation of fs.
    inverse_group_action : torch.Tensor
        Tensor of shape (group_order, dim, dim) that defines the inverse group representation in GL(n).
    in_indices : np.ndarray
        Index array. Indices of the input tensors that are convolved.
    out_indices : np.ndarray
        Index array. Indices of the output tensors of the convolution.
    feature_dim : int
        Dimensions that contains the features (in the LBM, the Q-dimension).
    channel_dim : int
        Dimension that contains the channels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        group_actions,
        inverse_group_actions,
        in_indices=None,
        out_indices=None,
        feature_dim=1,
        channel_dim=0
    ):
        super().__init__()
        self.dim = group_actions.shape[1]
        self.in_indices = np.arange(self.dim) if in_indices is None else in_indices
        self.out_indices = np.arange(self.dim) if out_indices is None else out_indices
        self.actions = group_actions
        self.inverse_actions = inverse_group_actions
        self.kernels = torch.nn.Parameter(torch.randn([out_channels, in_channels, self.dim, self.dim]))
        assert feature_dim != channel_dim
        self.feature_dim = feature_dim
        self.channel_dim = channel_dim

    def forward(self, m):
        in_to_out = self._in_to_out()
        m_indices = string.ascii_lowercase[:len(m.shape)]
        m_indices = m_indices[:self.channel_dim] + "u" + m_indices[self.channel_dim+1:]
        m_indices = m_indices[:self.feature_dim] + "w" + m_indices[self.feature_dim+1:]
        out_indices = m_indices.replace("u", "v").replace("w", "x")
        return torch.einsum(f"vuxw,{m_indices}->{out_indices}", in_to_out, m)

    def _in_to_out(self):
        return torch.einsum(
            "gij,cdjk,gkl->cdil",
            self.actions[:, self.out_indices, :],
            self.kernels,
            self.inverse_actions[:, :, self.in_indices]
        )


class GConvPermutation(GConv):
    """Group Convolution Layer based on permutations as group actions

    See GConv. The only difference are in the following parameters.

    Parameters
    ----------
    group_action : np.ndarray
        Index tensor of shape (group_order, dim) that defines the permutations.
    inverse_group_action : np.ndarray
        Index tensor of shape (group_order, dim) that defines the permutations.

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        group_actions,
        inverse_group_actions=None,
        in_indices=None,
        out_indices=None,
        feature_dim=1,
        channel_dim=0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            group_actions=group_actions,
            inverse_group_actions=inverse_group_actions,
            in_indices=in_indices,
            out_indices=out_indices,
            feature_dim=feature_dim,
            channel_dim=channel_dim
        )

    def _in_to_out(self):
        return (
            self.kernels[:, :, self.actions[:, self.out_indices], :][
                ..., self.inverse_actions[:, self.in_indices]
            ].sum(4).sum(2)
        )


class EquivariantNet(torch.nn.Module):
    """Render net equivariant by summing over all group representations.

    Parameters
    ----------
    net : torch.nn.Module
        The net that is wrapped to be equivariant.
    group_actions : torch.Tensor
        see GConv
    inverse_group_actions : torch.Tensor
        see GConv
    in_indices : np.ndarray
        see GConv
    out_indices : np.ndarray
        see GConv
    """
    def __init__(
        self,
        net,
        group_actions,
        inverse_group_actions=None,
        in_indices=None,
        out_indices=None
    ):
        super().__init__()
        self.dim = group_actions.shape[1]
        self.in_indices = np.arange(self.dim) if in_indices is None else in_indices
        self.out_indices = np.arange(self.dim) if out_indices is None else out_indices
        self.actions = group_actions
        self.inverse_actions = inverse_group_actions
        self.net = net

    def forward(self, x):
        x_in_group = torch.einsum(
            "gij,...j->g...i",
            self.inverse_actions[:, self.in_indices, :][:, :, self.in_indices],
            x
        )
        out_group = self.net(x_in_group)
        out = torch.einsum(
            "gij,g...j->...i",
            self.actions[:, self.out_indices, :][:, :, self.out_indices],
            out_group
        )
        return out


class EquivariantNeuralCollision(torch.nn.Module):
    """An MRT model that is equivariant under the lattice symmetries by relaxing all moments of the same
    order with the same rate.

    Parameters
    ----------
    lower_tau : float
        The default relaxation parameter operating on lower-order moments.
    tau_net : torch.nn.Module
        ...
    moment_transform : Transform
        The moment transformation.
    in_indices : np.ndarray
        Indices of the moments that the learned relaxation rates are conditined on.
        If None, use all moments with order <= 2.
    out_indices : np.ndarray
        Indices of the moments that relaxation rates are learned for.
        If None, use all moments with order > 2.
    """
    def __init__(self, lower_tau, tau_net, moment_transform, in_indices=None, out_indices=None):
        super().__init__()
        self.trafo = moment_transform
        self.lattice = moment_transform.lattice
        self.tau = lower_tau
        # infer moment order from moment name
        self.moment_order = np.array([sum(name.count(x) for x in "xyz") for name in moment_transform.names])
        self.last_taus = None
        # symmetries; wrap tau net equivariant
        symmetry_group = SymmetryGroup(moment_transform.lattice.stencil)
        self.in_indices = np.where(self.moment_order <= 2)[0] if in_indices is None else in_indices
        self.out_indices = np.where(self.moment_order > 2)[0] if out_indices is None else out_indices
        self.net = EquivariantNet(
            tau_net,
            symmetry_group.moment_action(moment_transform),
            symmetry_group.inverse_moment_action(moment_transform),
            in_indices=self.in_indices,
            out_indices=self.out_indices
        )
        self.net.to(dtype=self.lattice.dtype, device=self.lattice.device)

    @staticmethod
    def gt_half(a):
        """transform into a value > 0.5"""
        result = 1.5 + torch.nn.ELU()(a)
        assert (result >= 0.5).all()
        return result

    def _compute_relaxation_parameters(self, m):
        # move Q-axis to the back
        q_dim = len(m.shape) - 1 - self.lattice.D
        m = m.moveaxis(q_dim, len(m.shape)-1)
        # default taus
        taus = self.tau * torch.ones_like(m)
        # compute higher-order taus from lower-order ones through neural network
        tau = self.net(m[..., self.in_indices])
        # move Q-axis in front of grid axes
        # render tau > 0.5
        tau = self.gt_half(tau)
        taus[..., self.out_indices] = tau
        taus = taus.moveaxis(len(tau.shape) - 1, q_dim)
        return taus

    def forward(self, f):
        m = self.trafo.transform(f)
        taus = self._compute_relaxation_parameters(m)
        self.last_taus = taus
        meq = self.trafo.equilibrium(m)
        m_postcollision = m - 1. / taus * (m - meq)
        return self.trafo.inverse_transform(m_postcollision)


