import logging

from flask import Blueprint


api_bp = Blueprint('api_bp', __name__)
logger = logging.getLogger(__name__)


@api_bp.route('/ping/')
def ping_handler():
    """Ping-pong command
    """
    return {"result": "pong"}
