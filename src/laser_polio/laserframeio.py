import h5py
import numpy as np
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet


class LaserFrameIO(LaserFrame):
    @staticmethod
    def save(obj, parent_group, name):
        """
        Save a LaserFrame, ndarray, scalar, or dict into an HDF5 group.

        Parameters:
            obj: The object to save.
            parent_group: An h5py.File or h5py.Group.
            name: The name of the dataset or subgroup to create.
        """
        #if hasattr(obj, "_count") and hasattr(obj, "_capacity"):
        if isinstance(obj, LaserFrame):
            group = parent_group.create_group(name)
            group.attrs["count"] = obj._count
            group.attrs["capacity"] = obj._capacity

            for key in dir(obj):
                if not key.startswith("_"):
                    value = getattr(obj, key)
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value[:obj._count])

        elif isinstance(obj, PropertySet):
            group = parent_group.create_group(name)
            for key, value in obj.to_dict().items():
                try:
                    group.create_dataset(key, data=value)
                except TypeError:
                    group.attrs[key] = str(value)

        elif isinstance(obj, (np.ndarray, int, float)):
            parent_group.create_dataset(name, data=obj)

        else:
            raise TypeError(f"Unsupported object type for saving: {type(obj)}")

    @staticmethod
    def load(f, name):
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
