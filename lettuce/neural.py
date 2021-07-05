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
        x_in_group = torch.einsum("gij,...j->g...i", self.inverse_actions[:, :, self.in_indices], x)
        out_group = self.net(x_in_group)
        out = torch.einsum("gij,g...j->...i", self.actions[:, self.out_indices, :], out_group)
        return out


class EquivariantNeuralCollision(torch.nn.Module):
    """An MRT model that is equivariant under the lattice symmetries by relaxing all moments of the same
    order with the same rate.

    Parameters
    ----------
    lower_tau : float
        The default relaxation parameter operating on lower-order moments.
        Lower-order moments are defined in the sense that `tau_net`
        does not produce output for those orders. See documentation there.
    tau_net : torch.nn.Module
        A network that receives moments and returns unconstrained relaxation parameters for the highest-order moments.
        The input shape to the network is (..., Q), where "..." is any number of batch and grid dimensions
        and Q is the number of discrete distributions at each node.
        The output shape is (..., N), where N is the number of moment ORDERS, whose relaxation is prescribed
        by the network. Only the N highest moment orders will be relaxed.
        Note that the output of the network should be unconstrained and will be rendered > 0.5 by this class.
    moment_transform : Transform
        The moment transformation.

    """
    def __init__(self, lower_tau, tau_net, moment_transform):
        super().__init__()
        self.trafo = moment_transform
        self.lattice = moment_transform.lattice
        self.tau = lower_tau
        self.net = tau_net.to(dtype=self.lattice.dtype, device=self.lattice.device)
        # symmetries
        symmetry_group = SymmetryGroup(moment_transform.lattice.stencil)
        self.rep = symmetry_group.moment_action(moment_transform)
        # infer moment order from moment name
        self.moment_order = np.array([sum(name.count(x) for x in "xyz") for name in moment_transform.names])
        self.last_taus = None

    @staticmethod
    def gt_half(a):
        """transform into a value > 0.5"""
        return 0.5 + torch.exp(a)

    def _compute_relaxation_parameters(self, m):
        # default taus
        taus = self.tau * torch.ones_like(m)
        # compute m under all symmetry group representations
        y = torch.einsum(
            f"npq, ...q{'xyz'[:self.lattice.D]} -> n...{'xyz'[:self.lattice.D]}p",
            self.rep, m
        )
        # compute higher-order taus from neural network
        y = self.net(y).sum(0)
        # move Q-axis in front of grid axes
        q_dim = len(y.shape) - 1 - self.lattice.D
        tau = y.moveaxis(len(y.shape) - 1, q_dim)
        # render tau > 0.5
        tau = self.gt_half(tau)
        # apply learned taus to highest order moments
        moment_orders = np.sort(np.unique(self.moment_order))
        if not len(moment_orders) >= tau.shape[q_dim]:
            raise LettuceInvalidNetworkOutput(
                f"Network produced {tau.shape[q_dim]} taus but only {len(moment_orders)} "
                f"are available. Moments of each order are relaxed with the same tau."
            )
        learned_tau_moment_orders = moment_orders[-tau.shape[q_dim]:]
        for i, order in enumerate(learned_tau_moment_orders):
            taus[self.moment_order == order] = tau[i]
        return taus

    def forward(self, f):
        m = self.trafo.transform(f)
        taus = self._compute_relaxation_parameters(m)
        self.last_taus = taus
        meq = self.trafo.equilibrium(m)
        m_postcollision = m - 1. / taus * (m - meq)
        return self.trafo.inverse_transform(m_postcollision)


