"""
datautils for writing/reading hdf5 files.
"""

import h5py
from torch.utils import data
from lettuce._version import get_versions
import pickle
import io
import numpy as np

__all__ = ["HDF5Reporter", "LettuceDataset"]


class HDF5Reporter:
    """ HDF5 reporter for distribution function f in lettuce containing
        metadata of the simulation.

        Parameters
        ----------
            filebase : string
                Path to the hdf5 file with annotations.
            metadata : dictionary
                Optional metadata can be saved. The passed values must be of type string.
                >>> metadata = {"attr_1": "str_value_1", "attr_2": "str_value_2"}
            interval : integer
                Define the step interval after the reporter is applied.
                The reporter will save f every "interval" step.

        Examples
        --------
        Create a HDF5 reporter.
        >>> import lettuce as lt
        >>> lattice = lt.Lattice(lt.D3Q27, device="cpu")
        >>> flow = lt.TaylorGreenVortex3D(50, 300, 0.1, lattice)
        >>> collision = ...
        >>> simulation = ...
        >>> hdf5_reporter = lt.HDF5Reporter(
        >>>     flow=flow,
        >>>     lattice=lattice,
        >>>     collision=collision,
        >>>     interval= 100,
        >>>     filebase="./h5_output")
        >>> simulation.reporters.append(hdf5_reporter)
        """

    def __init__(self, flow, collision, interval, filebase='./output', metadata=None):
        self.lattice = flow.units.lattice
        self.interval = interval
        self.filebase = filebase
        fs = h5py.File(self.filebase + '.h5', 'w')
        fs.attrs['lettuce_version'] = get_versions()['version']
        fs.attrs["flow"] = self._pickle_to_h5(flow)
        fs.attrs['collision'] = self._pickle_to_h5(collision)
        if metadata:
            for attr in metadata:
                fs.attrs[attr] = metadata[attr]
        self.shape = (self.lattice.Q, *flow.grid[0].shape)
        fs.create_dataset(name="f",
                          shape=(0, *self.shape),
                          maxshape=(None, *self.shape))
        fs.close()

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            with h5py.File(self.filebase + '.h5', 'r+') as fs:
                fs["f"].resize(fs["f"].shape[0]+1, axis=0)
                fs["f"][-1, ...] = self.lattice.convert_to_numpy(f)
                fs.attrs['data'] = str(fs["f"].shape[0])
                fs.attrs['steps'] = str(i)

    @staticmethod
    def _pickle_to_h5(instance):
        bytes_io = io.BytesIO()
        pickle.dump(instance, bytes_io)
        bytes_io.seek(0)
        return np.void(bytes_io.getvalue())


class LettuceDataset(data.Dataset):
    """ Custom dataset for HDF5 files in lettuce that can be used by torch's
        dataloader.

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
        >>>              filebase= "./hdf5_output.h5",
        >>>              target=True)
        >>> train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True)
        >>> for (f, target, idx) in train_loader:
        >>>     ...
        """

    def __init__(self, filebase, transform=None, target=False, skip_idx_to_target=1):
        super().__init__()
        self.filebase = filebase
        self.transform = transform
        self.target = target
        self.skip_idx_to_target = skip_idx_to_target
        self.fs = h5py.File(self.filebase, "r")
        self.shape = self.fs["f"].shape
        self.keys = list(self.fs.keys())
        self.lattice = self._unpickle_from_h5(self.fs.attrs["flow"]).units.lattice

    def __str__(self):
        for attr, value in self.fs.attrs.items():
            if attr in ('flow', 'collision'):
                print(attr + ": " + str(self._unpickle_from_h5(self.fs.attrs[attr])))
            else:
                print(attr + ": " + str(value))
        return ""

    def __len__(self):
        return self.shape[0] - self.skip_idx_to_target if self.target else self.shape[0]

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

    def __del__(self):
        self.fs.close()

    def get_data(self, idx):
        return self.lattice.convert_to_tensor(self.fs["f"][idx])

    def get_attr(self, attr):
        return self.fs.attrs[attr]

    @staticmethod
    def _unpickle_from_h5(byte_str):
        return pickle.load(io.BytesIO(byte_str))
