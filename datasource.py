import os.path
from gcsfs_map import GCSFSMap

source_id = None
with open('datasource.cfg', 'r') as f:
    source_id = f.read().strip()

curc_basename = '/projects/brgr6137/data/erai-rasmussen'
gcp_bucket = 'erai-rasmussen'
gcp_project = 'thesis-research'
service_account_filename = 'gcp-access.secret.json'

def set_gcp_bucket(bucket):
    gcp_bucket = bucket
    
def set_gcp_project(project):
    gcp_project = project
    
def set_service_account_cred_file(filename):
    service_account_filename = filename
    
def gcp_fs():
    from fs_gcsfs import GCSFS
    from google.cloud.storage.client import Client
    import google.oauth2.service_account as service_account
    credentials = service_account.Credentials.from_service_account_file(filename=service_account_filename)
    return GCSFS(bucket_name=gcp_bucket, client=Client(project=gcp_project, credentials=credentials))

def gcp_fix_storage():
    gcp_fs().fix_storage()

def rasmussen(name='daily-deg1'):
    if os.path.exists('./data/regular-ras'):
        return os.path.join('./data/regular-ras/', f'{name}.zarr')
    elif source_id == 'gcp':
        return GCSFSMap(os.path.join('regular-ras', f'{name}.zarr'), gcp_fs())
    elif source_id == 'curc':
        return os.path.join(curc_basename, 'regular-ras', f'{name}.zarr')
    else:
        raise Exception(source_id)
        
def erai(name='daily-deg1'):
    if os.path.exists('./data/erai-raw'):
        return os.path.join('./data/erai-raw/', f'{name}.zarr')
    elif source_id == 'gcp':
        return GCSFSMap(os.path.join('erai-raw', f'{name}.zarr'), gcp_fs())
    elif source_id == 'curc':
        return os.path.join(curc_basename, 'erai-raw', f'{name}.zarr')
    else:
        raise Exception(source_id)
