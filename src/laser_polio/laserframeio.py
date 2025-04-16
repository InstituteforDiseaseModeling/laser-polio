import numpy as np
import h5py
from laser_core.laserframe import LaserFrame

class LaserFrameIO(LaserFrame):
    def save(self, filename: str, initial_populations=None, age_distribution=None, cumulative_deaths=None, eula_age=None) -> None:
        """Save LaserFrame properties to an HDF5 file."""
        with h5py.File(filename, 'w') as hdf:
            hdf.attrs['count'] = self._count
            hdf.attrs['capacity'] = self._capacity
            
            if initial_populations is not None:
                hdf.attrs['init_pops'] = initial_populations
            if age_distribution is not None:
                hdf.attrs['age_dist'] = age_distribution
            if cumulative_deaths is not None:
                hdf.attrs['cumulative_deaths'] = cumulative_deaths
            if eula_age is not None:
                hdf.attrs['eula_age'] = eula_age

            for key in dir(self):
                if not key.startswith("_"):
                    value = getattr(self, key)
                    if isinstance(value, np.ndarray):
                        data = value[:self._count]
                        hdf.create_dataset(key, data=data)

    @classmethod
    def load(cls, filename: str, capacity=None):
        """Load a LaserFrameIO object from an HDF5 file."""
        with h5py.File(filename, 'r') as hdf:
            saved_count = int(hdf.attrs['count'])
            saved_capacity = int(hdf.attrs['capacity'])
            saved_capacity = int(1.1 * saved_count)# hack
            
            # Allow user override of capacity
            final_capacity = capacity if capacity is not None else saved_capacity
            
            # Initialize the LaserFrame
            frame = cls(capacity=final_capacity, initial_count=saved_count)
            
            # Recover properties
            for key in hdf.keys():
                data = hdf[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                setattr(frame, key, np.zeros(frame._capacity, dtype=dtype))  # Preallocate
                getattr(frame, key)[:saved_count] = data  # Fill values up to saved count
                
            return frame
