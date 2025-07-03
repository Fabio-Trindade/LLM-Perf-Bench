def get_url(host, port, endpoint):
    return f"http://{host}:{port}/v1/{endpoint}"

def get_url_from_config(config):
    return get_url(config.host, config.port, config.endpoint)