import json
import logging
import logging.config

import numpy as np
from flask import Flask, jsonify

from server.data_provider import UserDataProvider, BookingDataProvider, ItemDataProvider
from server.exceptions import BaseApiException
from server.functions import get_abs_path, clean_json_dict_keys
from server.recommender import ClusterRecommender, PopItemRecommender, CBItemRecommender

logger = logging.getLogger(__name__)


def handle_exception_with_as_dict_method(error):
    logger.error(error)
    return jsonify(error.as_dict())


class ApiAppJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
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
        self._load_data_providers()
        self._load_recommenders()
        logger.info(u"API has been initialized")

    def _load_data_providers(self):
        self.user_dp = UserDataProvider.load(self.config)
        logger.info(u"User data provider has been initialized")

        self.booking_dp = BookingDataProvider.load(self.config)
        logger.info(u"Booking data provider has been initialized")

        self.item_dp = ItemDataProvider.load(self.config)
        logger.info(u"Item data provider has been initialized")

    def _load_recommenders(self):
        self.item_pop_recommender = PopItemRecommender.load(
            self.booking_dp, self.item_dp
        )
        logger.info(u"Item popularity based recommender has been initialized")

        self.item_cb_recommender = CBItemRecommender.load(
            self.booking_dp, self.item_dp
        )
        logger.info(u"Item content-based recommender has been initialized")

        self.bg_recommender = ClusterRecommender.load(
            self.config, self.user_dp, self.booking_dp, self.item_dp
        )
        logger.info(u"Personalized booking clusters recommender has been initialized")

    def setup_error_handlers(self):
        self.register_error_handler(BaseApiException, handle_exception_with_as_dict_method)

    def make_response(self, rv):
        if isinstance(rv, self.response_class):
            return rv
        if isinstance(rv, (dict, list)):
            rv = clean_json_dict_keys(rv)
            rv = json.dumps(rv, cls=ApiAppJsonEncoder)
        elif isinstance(rv, bool):
            rv = "true" if rv else "false"
        return super(APIApp, self).make_response(rv)
