
from lettuce import native


__all__ = ["ClassWithNativeImplementation"]


class ClassWithNativeImplementation:
    def __init__(
            self,
            lattice,
            collision=None,
            streaming=None,
            use_no_streaming_mask=False,
            use_no_collision_mask=False,
    ):
        self.native_call = None
        from ..collision import NoCollision
        from ..streaming import NoStreaming
        collision = NoCollision if collision is None else collision
        streaming = NoStreaming if streaming is None else streaming

        if lattice.use_native:
            if not hasattr(lattice.stencil, 'native_class'):
                print('stencil not natively implemented')
                return
            if not hasattr(lattice.equilibrium, 'native_class'):
                print('equilibrium not natively implemented')
                return
            if not hasattr(collision, 'native_class'):
                print('collision not natively implemented')
                return
            if not hasattr(streaming, 'native_class'):
                print('stream not natively implemented')
                return

            self.native_call = native.resolve(
                 lattice.stencil.native_class.name,
                 lattice.equilibrium.native_class.name,
                 collision.native_class.name,
                 streaming.native_class.name,
                 use_no_streaming_mask,
                 use_no_collision_mask
            )

            if self.native_call is None:
                print('combination not natively generated')


    def call(self, *args, **kwargs):
        # virtual method
        return NotImplemented

    def __call__(self, *args, **kwargs):
        if self.native_call is not None: 
            return self.native_call(*args, **kwargs)
        else:
            return self.call(*args, **kwargs)

