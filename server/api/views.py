from flask import current_app as app


def get_cluster_based_recs(uid, top_clusters, top_items):
    res = {}
    ug_id = app.user_dp.get_cluster_id(uid)

    if ug_id is not None:
        iid_recs = app.item_pop_recommender.get_recs(uid)
        bg_recs = app.bg_recommender.get_recs(ug_id, iid_recs, top_clusters, top_items)
        recs = app.item_dp.prepare_bg_recs(bg_recs, iid_recs, top_items=top_items)
        for bg_rec in recs:
            bg_rec["features"] = app.booking_dp.get_cluster_features(bg_rec["bg_id"])

        res = {
            "user": app.user_dp.get_uid_features(uid),
            "user_cluster": {ug_id: app.user_dp.get_cluster_features(ug_id)},
            "recs": recs,
            "prev_bookings_summary": app.booking_dp.get_uid_booking_summary(uid),
        }
    return {"result": res}


def get_content_based_recs(uid, top_items):
    res = {}
    if app.item_feature_dp.has_uid_features(uid):
        iid_recs = app.item_cb_recommender.get_recs(uid, top_items)
        recs = app.item_dp.prepare_iid_recs(iid_recs)

        res = {
            "user": app.user_dp.get_uid_features(uid),
            "recs": recs,
            "prev_bookings_summary": app.booking_dp.get_uid_booking_summary(uid),
        }
    return {"result": res}
