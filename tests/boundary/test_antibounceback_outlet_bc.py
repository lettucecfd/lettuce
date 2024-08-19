from copy import copy

from tests.common import *


def test_anti_bounce_back_outlet(fix_configuration, fix_stencil):
    """Compares the result of the application of the boundary to f to the
    result using the formula taken from page 195
    of "The lattice Boltzmann method" (2016 by Krüger et al.) if both are
    similar it is assumed to be working fine."""
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    f = flow.f

    # generates reference value of f using non-dynamic formula
    f_ref = copy(f)
    u = flow.u()
    D = flow.stencil.d
    Q = flow.stencil.q

    if D == 3:
        direction = [1, 0, 0]

        if Q == 27:
            u_w = u[:, -1, :, :] + 0.5 * (u[:, -1, :, :] - u[:, -2, :, :])
            u_w_norm = torch.norm(u_w, dim=0)

            for i in [1, 11, 13, 15, 17, 19, 21, 23, 25]:
                stencil_e_tensor = torch.tensor(flow.stencil.e[i],
                                                device=f.device, dtype=f.dtype)

                f_ref[flow.stencil.opposite[i], -1, :, :] = (
                        - f_ref[i, -1, :, :]
                        + (flow.stencil.w[i] * flow.rho()[0, -1, :, :]
                           * (2 + torch.einsum(
                            'c, cyz -> yz', stencil_e_tensor, u_w
                           ) ** 2
                           / flow.stencil.cs ** 4
                           - (u_w_norm / flow.stencil.cs) ** 2))
                )

        if Q == 19:
            u_w = u[:, -1, :, :] + 0.5 * (u[:, -1, :, :] - u[:, -2, :, :])
            u_w_norm = torch.norm(u_w, dim=0)

            for i in [1, 11, 13, 15, 17]:
                stencil_e_tensor = torch.tensor(flow.stencil.e[i],
                                                device=f.device, dtype=f.dtype)

                f_ref[flow.stencil.opposite[i], -1, :, :] = (
                        - f_ref[i, -1, :, :]
                        + (flow.stencil.w[i]
                           * flow.rho()[0, -1, :, :]
                           * (2 + torch.einsum('c, cyz -> yz',
                                               stencil_e_tensor,
                                               u_w) ** 2
                              / flow.stencil.cs ** 4
                              - (u_w_norm / flow.stencil.cs) ** 2)))

    if D == 2 and Q == 9:
        direction = [1, 0]
        u_w = u[:, -1, :] + 0.5 * (u[:, -1, :] - u[:, -2, :])
        u_w_norm = torch.norm(u_w, dim=0)

        for i in [1, 5, 8]:
            stencil_e_tensor = torch.tensor(flow.stencil.e[i],
                                            device=f.device, dtype=f.dtype)

            f_ref[flow.stencil.opposite[i], -1, :] = - f_ref[i, -1, :] + (
                    flow.stencil.w[i] * flow.rho()[0, -1, :]
                    * (2 + torch.einsum('c, cy -> y', stencil_e_tensor,
                                        u_w) ** 2
                       / flow.stencil.cs ** 4 - (
                               u_w_norm / flow.stencil.cs) ** 2))

    if D == 1 and Q == 3:
        direction = [1]
        u_w = u[:, -1] + 0.5 * (u[:, -1] - u[:, -2])
        u_w_norm = torch.norm(u_w, dim=0)

        for i in [1]:
            stencil_e_tensor = torch.tensor(flow.stencil.e[i],
                                            device=f.device, dtype=f.dtype)

            f_ref[flow.stencil.opposite[i], -1] = (
                    - f_ref[i, -1]
                    + (flow.stencil.w[i]
                       * flow.rho()[0, -1]
                       * (2 + torch.einsum('c, x -> x', stencil_e_tensor,
                                           u_w) ** 2
                       / flow.stencil.cs ** 4
                          - (u_w_norm / flow.stencil.cs) ** 2)
                       )
            )

    # generates value from actual boundary implementation
    abb_outlet = AntiBounceBackOutlet(direction=direction, flow=flow)
    f = abb_outlet(flow)
    assert f.cpu().numpy() == pytest.approx(f_ref.cpu().numpy())
