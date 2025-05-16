import h5py
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet


class LaserFrameIO(LaserFrame):
    def save_snapshot(self, path, results_r=None, pars=None):
        """
        Save this LaserFrame and optional extras to an HDF5 snapshot file.

        Parameters:
            path: Destination file path
            results_r: Optional 2D numpy array of recovered counts
            pars: Optional PropertySet or dict of parameters
        """
        with h5py.File(path, "w") as f:
            self._save(f, "people")

            if results_r is not None:
                f.create_dataset("recovered", data=results_r)

            if pars is not None:
                data = pars.to_dict() if isinstance(pars, PropertySet) else pars
                self._save_dict(data, f.create_group("pars"))

    def _save(self, parent_group, name):
        """
        Internal method to save this LaserFrame under the given group name.
        """
        group = parent_group.create_group(name)
        group.attrs["count"] = self._count
        group.attrs["capacity"] = self._capacity

        for key in dir(self):
            if not key.startswith("_"):
                value = getattr(self, key)
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value[: self._count])

    def _save_dict(self, data, group):
        """
        Internal method to save a dict as datasets and attributes in a group.
        """
        for key, value in data.items():
            try:
                group.create_dataset(key, data=value)
            except TypeError:
                group.attrs[key] = str(value)

    @staticmethod
    def __load(f, name):
        """
        Load a known object from an HDF5 file by name.

        Parameters:
            f: h5py.File or h5py.Group (open in read mode)
            name: "people", "recovered", or "pars"

        Returns:
            LaserFrame, ndarray, or dict depending on name
        """
        if name == "people":
            group = f[name]
            count = int(group.attrs["count"])
            capacity = int(group.attrs["capacity"])
            from laser_core import LaserFrame  # or however LaserFrame is imported

            frame = LaserFrame(capacity=capacity, initial_count=count)

            for key in group:
                data = group[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                setattr(frame, key, np.zeros(capacity, dtype=dtype))
                getattr(frame, key)[:count] = data

            return frame

        elif name == "recovered":
            return f[name][()]  # returns full ndarray

        elif name == "pars":
            group = f[name]
            d = {key: group[key][()] for key in group}
            d.update(dict(group.attrs))
            return d

        else:
            raise ValueError(f"Unsupported group name: {name}")

    @classmethod
    def load_snapshot(cls, path):
        """
        Load a LaserFrameIO and optional extras from an HDF5 snapshot file.

        Returns:
            frame (LaserFrameIO)
            results_r (np.ndarray or None)
            pars (dict or None)
        """
        with h5py.File(path, "r") as f:
            group = f["people"]
            count = int(group.attrs["count"])
            capacity = int(group.attrs["capacity"])

            frame = cls(capacity=capacity, initial_count=count)
            for key in group:
                data = group[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                setattr(frame, key, np.zeros(capacity, dtype=dtype))
                getattr(frame, key)[:count] = data

            results_r = f["recovered"][()] if "recovered" in f else None
            pars = {key: f["pars"][key][()] for key in f["pars"]} if "pars" in f else None
            if "pars" in f:
                pars.update(dict(f["pars"].attrs))

        return frame, results_r, pars
