import os

from server.app import APIApp
from server.api import api_bp


def get_app(config_path=None):
    config_path = config_path or os.getenv('CONFIG')

    api_app = APIApp(__name__)
    api_app.init(config_path)
    api_app.setup_error_handlers()
    api_app.register_blueprint(api_bp)
    return api_app


if __name__ == "__main__":
    app = get_app(os.getenv('CONFIG', 'local'))
    app.debug = (os.getenv('DEBUG') in ['true', '1', 't', 'on', 'y'])
    app.run(host='0.0.0.0', port=8085)
