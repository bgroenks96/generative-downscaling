import os.path
import gcsfs

class EraiRasDataLoader:
    def __init__(self, gcs_bucket, gcs_project, auth='browser'):
        self.gcs = gcsfs.GCSFileSystem(project=gcs_project, token=auth)
        self.bucket = gcs_bucket

    def rasmussen(self, name='daily-deg1'):
        """
        Returns key/value mapper for rasmussen ZARR
        """
        if os.path.exists('./data/regular-ras'):
            return os.path.join('./data/regular-ras/', f'{name}.zarr')
        return self.gcs.get_mapper(os.path.join(self.bucket, 'regular-ras', f'{name}.zarr'))

    def erai(self, name='daily-deg1'):
        """
        Returns key/value mapper for ERA-interim ZARR
        """
        if os.path.exists('./data/erai-raw'):
            return os.path.join('./data/erai-raw/', f'{name}.zarr')
        return self.gcs.get_mapper(os.path.join(self.bucket, 'erai-raw', f'{name}.zarr'))
