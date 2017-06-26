from flask import current_app as app


def get_recs(uid, top_clusters, top_items):
    res = {}
    ug_id = app.user_dp.get_cluster_id(uid)

    if ug_id is not None:
        iid_recs = app.item_pop_recommender.get_recs(uid)
        bg_recs = app.bg_recommender.get_recs(ug_id, iid_recs, top_clusters, top_items)
        recs = app.item_dp.prepare_bg_recs(bg_recs, iid_recs, top_items=top_items)

        res = {
            "user": app.user_dp.get_uid_features(uid),
            "user_cluster": {ug_id: app.user_dp.get_cluster_features(ug_id)},
            "booking_clusters": {bg_id: app.booking_dp.get_cluster_features(bg_id) for bg_id in recs},
            "recs": recs,
            "prev_bookings_summary": app.booking_dp.get_uid_booking_summary(uid),
        }
    return {"result": res}
