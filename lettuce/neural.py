import string

import torch
import numpy as np

from lettuce.util import LettuceInvalidNetworkOutput, LettuceException
from lettuce.symmetry import SymmetryGroup

__all__= ["GConv", "GConvPermutation", "EquivariantNet", "EquivariantNeuralCollision"]


class GConv(torch.nn.Module):
    """Group Convolution Layer
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

    """
    def __init__(self, lower_tau, tau_net, moment_transform):
        super().__init__()
        self.trafo = moment_transform
        self.lattice = moment_transform.lattice
        self.tau = lower_tau
        # infer moment order from moment name
        self.moment_order = np.array([sum(name.count(x) for x in "xyz") for name in moment_transform.names])
        self.last_taus = None
        # symmetries; wrap tau net equivariant
        symmetry_group = SymmetryGroup(moment_transform.lattice.stencil)
        self.in_indices = np.where(self.moment_order <= 2)[0]
        self.out_indices = np.where(self.moment_order > 2)[0]
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


