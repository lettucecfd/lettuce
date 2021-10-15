"""
datautils for writing/reading hdf5 files.
"""

import h5py
from torch.utils import data
from lettuce.util import *

class hdf5writer:
    """ HDF5 writer for distribution function f in lettuce containing metadata of the simulation.

        Parameters
        ----------
            filebase : string
                Path to the hdf5 file with annotations.
            metadata : dictionary
                Optional metadata can be saved. The passed values must be of type string.
                >>> metadata = {"attribut_1": "str_value_1", "attribut_2": "str_value_2"}
            interval : integer
                Define the step interval after the writer is applied.

        Examples
        --------
        Create a HDF5 reporter.
        >>> import lettuce as lt
        >>> lattice = lt.Lattice(lt.D3Q27, device="cpu")
        >>> flow = lt.TaylorGreenVortex3D(50, 300, 0.1, lattice)
        >>> collision = ...
        >>> simulation = ...
        >>> hdf5_reporter = lt.hdf5writer(
        >>>     flow=flow,
        >>>     lattice=lattice,
        >>>     collision=collision,
        >>>     interval= 100,
        >>>     filebase="./hdf5_output")
        >>> simulation.reporters.append(hdf5_reporter)
        """

    def __init__(self, flow, lattice, collision, interval, filebase='./output', metadata=None):
        self.lattice = lattice
        self.interval = interval
        self.filebase = filebase
        fs = h5py.File(self.filebase + '.hdf5', 'w')
        fs.attrs['device'] = self.lattice.device
        fs.attrs['dtype'] = str(self.lattice.dtype)
        fs.attrs['stencil'] = self.lattice.stencil.__name__
        fs.attrs["flow"] = flow.__class__.__name__
        fs.attrs["relaxation_parameter"] = flow.units.relaxation_parameter_lu
        fs.attrs['resolution'] = flow.resolution
        fs.attrs['reynolds_number'] = flow.units.reynolds_number
        fs.attrs['mach_number'] = flow.units.mach_number
        fs.attrs['collision'] = collision.__class__.__name__
        if metadata:
            for attr in metadata:
                fs.attrs[attr] = metadata[attr]
        fs.close()
        self.shape =tuple(
            j for i in (flow.units.lattice.Q, flow.grid[0].shape) for j in (i if isinstance(i, tuple) else (i,)))

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            with h5py.File(self.filebase + '.hdf5', 'r+') as fs:
                dset = fs.create_dataset(f"{i:06d}", self.shape)
                dset[:] = self.lattice.convert_to_numpy(f)
                fs.attrs['data'] = str(len(fs))
                fs.attrs['steps'] = str(i)

class LettuceDataset(data.Dataset):
    """ Custom dataset for HDF5 files in lettuce that can be used by torch's dataloader.

    Parameters
    ----------
        filebase : string
            Path to the hdf5 file with annotations.
        transform : class object
            Optional transform to be applied on a f loaded from HDF5 file.
        target : logical operation (True, False)
            Returns also the next dataset[idx + skip_idx_to_target] - default=False
        skip_idx_to_target : integer
            Define which next target dataset is returned if target is True - default=1

    Examples
        --------
        Create a data loader.
        >>> import lettuce as lt
        >>> import torch
        >>> lattice = lt.Lattice(lt.D3Q27, device="cpu")
        >>> dataset_train = lt.LettuceDataset(lattice=lattice,
        >>>              filebase= "./hdf5_output.hdf5",
        >>>              target=True)
        >>> train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True)
        >>> for (f, target, idx) in train_loader:
        >>>     ...
        """

    def __init__(self, lattice, filebase, transform=None, target=False, skip_idx_to_target=1):
        super().__init__()
        self.filebase = filebase
        self.lattice = lattice
        self.transform = transform
        self.target = target
        self.skip_idx_to_target = skip_idx_to_target
        with h5py.File(self.filebase, "r") as fs:
            self.keys = list(fs.keys())

    def __str__(self):
        print("Metadata:")
        with h5py.File(self.filebase, "r") as fs:
            for attr, value in fs.attrs.items():
                print("    " + attr + ": " + str(value))
        print("Object:")
        print(f"    target: {self.target}")
        if self.target:
            print(f"    skip_idx_to_target: {self.skip_idx_to_target}")

        return ""

    def __len__(self):
        with h5py.File(self.filebase, "r") as fs:
            return len(fs) - self.skip_idx_to_target if self.target else len(fs)

    def __getitem__(self, idx):
        f = self.get_data(idx)
        target = []
        if self.target:
            target = self.get_data(idx + self.skip_idx_to_target)
        if self.transform:
            f = self.transform(f)
            if self.target:
                target = self.transform(target)
        return (f, target, idx) if self.target else (f, idx)

    def get_data(self, idx):
        with h5py.File(self.filebase, "r") as fs:
            f = self.lattice.convert_to_tensor(fs[self.keys[idx]][:])
        return f

    def get_attr(self, attr):
        with h5py.File(self.filebase, "r") as fs:
            attr = fs.attrs[attr]
        return attr
