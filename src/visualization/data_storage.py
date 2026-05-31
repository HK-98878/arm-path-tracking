"""Episode data storage utilities using HDF5."""

import h5py
import numpy as np
from pathlib import Path


class EpisodeDataStorage:
    """Save/load episode data to/from HDF5 files."""

    def save(self, episode_data, filepath, metadata=None):
        """Save episode data to HDF5.

        Args:
            episode_data: Dict from EpisodeRecorder.get_episode_data()
            filepath: Output path (e.g., 'episode_000.h5')
            metadata: Optional dict of metadata (checkpoint path, etc.)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Metadata
            if metadata:
                meta_grp = f.create_group('metadata')
                for key, val in metadata.items():
                    if isinstance(val, str):
                        meta_grp.attrs[key] = val
                    else:
                        meta_grp.attrs[key] = val

            # Arrays
            for key, val in episode_data.items():
                if isinstance(val, np.ndarray):
                    f.create_dataset(key, data=val, compression='gzip')
                elif isinstance(val, list) and len(val) > 0:
                    # Handle list of dicts (reward_components)
                    if isinstance(val[0], dict):
                        grp = f.create_group(key)
                        # Extract keys from first element
                        for subkey in val[0].keys():
                            data = [v.get(subkey, 0.0) for v in val]
                            grp.create_dataset(subkey, data=np.array(data),
                                             compression='gzip')
                    else:
                        f.create_dataset(key, data=np.array(val),
                                       compression='gzip')

        print(f"Episode data saved: {filepath}")

    def load(self, filepath):
        """Load episode data from HDF5.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dict with all datasets
        """
        with h5py.File(filepath, 'r') as f:
            data = {}

            # Recursively load all datasets
            def load_group(group, prefix=''):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        data[prefix + key] = item[:]
                    elif isinstance(item, h5py.Group):
                        load_group(item, prefix + key + '/')

            load_group(f)

            # Load metadata
            if 'metadata' in f:
                data['metadata'] = dict(f['metadata'].attrs)

        return data
