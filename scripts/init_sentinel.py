from sentinelsat import SentinelAPI
import os

def create_sentinel_client():
    user = os.getenv('SENTINELAPI_USERNAME')
    password = os.getenv('SENTINELAPI_PASSWORD')
    api_url = os.getenv('SENTINELAPI_URL', 'https://scihub.copernicus.eu/dhus')

    if not user or not password:
        raise RuntimeError("Missing SentinelAPI credentials in environment")

    api = SentinelAPI(user, password, api_url)
    return api

if __name__ == '__main__':
    client = create_sentinel_client()
    print('Sentinel client created:', client)
