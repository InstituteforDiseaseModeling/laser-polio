import h5py
import numpy as np
from laser_core.laserframe import LaserFrame


class LaserFrameIO(LaserFrame):
    @staticmethod
    def save_to_group(frame, group, initial_populations=None, age_distribution=None, cumulative_deaths=None, eula_age=None):
        """
        Save LaserFrameIO properties to an existing HDF5 group.

        Parameters:
            frame: The LaserFrame object to save
            group: An h5py.Group object
            initial_populations: Optional array to save as attribute
            age_distribution: Optional array to save as attribute
            cumulative_deaths: Optional array to save as attribute
            eula_age: Optional value to save as attribute
        """
        # Core frame attributes
        group.attrs["count"] = frame._count
        group.attrs["capacity"] = frame._capacity

        # Optional metadata
        if initial_populations is not None:
            group.attrs["init_pops"] = initial_populations
        if age_distribution is not None:
            group.attrs["age_dist"] = age_distribution
        if cumulative_deaths is not None:
            group.attrs["cumulative_deaths"] = cumulative_deaths
        if eula_age is not None:
            group.attrs["eula_age"] = eula_age

        # Save all np.ndarray properties
        for key in dir(frame):
            if not key.startswith("_"):
                value = getattr(frame, key)
                if isinstance(value, np.ndarray):
                    data = value[: frame._count]
                    group.create_dataset(key, data=data)

    @classmethod
    def load(cls, filename: str, capacity=None):
        """Load a LaserFrameIO object from the 'people' group inside an HDF5 file."""
        with h5py.File(filename, "r") as hdf:
            if "people" not in hdf:
                raise ValueError(f"No 'people' group found in {filename}")
            group = hdf["people"]

            saved_count = int(group.attrs["count"])
            saved_capacity = int(group.attrs["capacity"])
            saved_capacity = int(1.5 * saved_count)  # hack

            # Allow user override of capacity
            final_capacity = capacity if capacity is not None else saved_capacity

            # Initialize the LaserFrame
            frame = cls(capacity=final_capacity, initial_count=saved_count)

            # Recover properties
            for key in group.keys():
                data = group[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                setattr(frame, key, np.zeros(frame._capacity, dtype=dtype))  # Preallocate
                getattr(frame, key)[:saved_count] = data  # Fill values up to saved count

            return frame
