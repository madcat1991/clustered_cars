import logging

from flask import Blueprint, request

from server.api.views import get_recs
from server.exceptions import ArgErrorException

api_bp = Blueprint('api_bp', __name__)
logger = logging.getLogger(__name__)


DEFAULT_TOP_CLUSTERS = 3
DEFAULT_TOP_ITEMS = 5


@api_bp.route('/ping/')
def ping_handler():
    """Ping-pong command
    """
    return {"result": "pong"}


@api_bp.route('/cluster/recs/')
def cluster_recs_handler():
    uid = request.args.get("uid")
    if uid is None:
        raise ArgErrorException("uid", "argument has to be specified")

    top_clusters = request.args.get("top", type=int, default=DEFAULT_TOP_CLUSTERS)
    top_items = request.args.get("top_items", type=int, default=DEFAULT_TOP_ITEMS)
    return get_recs(uid, top_clusters, top_items)
