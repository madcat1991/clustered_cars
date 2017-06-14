import json
import logging
import logging.config

from flask import Flask, jsonify

from server.exceptions import BaseApiException
from server.functions import get_abs_path

logger = logging.getLogger(__name__)


def handle_exception_with_as_dict_method(error):
    logger.error(error)
    return jsonify(error.as_dict())


class ApiAppJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class APIApp(Flask):
    def __init__(self, *args, **kwargs):
        super(APIApp, self).__init__(*args, **kwargs)
        self.response_class.default_mimetype = 'application/json'

    def _load_config(self, config_path):
        if not config_path.startswith('/'):
            config_path = get_abs_path('etc', config_path, 'config.py')
        self.config.from_pyfile(config_path)

    def _setup_logger(self):
        logging_options = self.config.get('LOGGING', {})
        self.logger.handlers = []
        logging.config.dictConfig(logging_options)
        logger.info(u"Logger has been setuped")

    def init(self, config_path):
        self._load_config(config_path)
        self._setup_logger()
        logger.info(u"API has been initialized")

    def setup_error_handlers(self):
        self.register_error_handler(BaseApiException, handle_exception_with_as_dict_method)

    def make_response(self, rv):
        if isinstance(rv, self.response_class):
            return rv
        if isinstance(rv, (dict, list)):
            rv = json.dumps(rv, cls=ApiAppJsonEncoder)
        elif isinstance(rv, bool):
            rv = "true" if rv else "false"
        return super(APIApp, self).make_response(rv)
