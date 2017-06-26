def get_ug_data(ug_file_path):
    """ The function creates the index uid -> ug_id.
    One user is assigned to only cluster

    :param ug_file_path: a path to the file containing information about user clusters
    :return: uid -> ug_id index
    """
    uid_to_ug = {}

    with open(ug_file_path) as f:
        # skipping
        while not next(f).startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Users:"):
                for uid in line.lstrip("Users:").split(","):
                    uid_to_ug[uid.strip()] = cl_id
            elif line.startswith("Cluster"):
                cl_id += 1
    return uid_to_ug


def get_bg_data(bg_file_path):
    """ The function creates two indices:
        * bid -> {bg_id1, bg_id2, ...}
        * bg_id -> {iid1, iid2, ...}

    One booking is assigned to only one cluster

    :param bg_file_path: a path to the file containing information about booking clusters
    :return: bid -> {bg_id1, bg_id2, ...} and bg_id -> {iid1, iid2, ...} indices
    """
    bid_to_bgs = {}
    bg_iids = {}

    with open(bg_file_path) as f:
        # skipping
        while not next(f).startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Bookings:"):
                for bid in line.lstrip("Bookings:").split(","):
                    bid_to_bgs[bid.strip()] = cl_id
            elif line.startswith("Items:"):
                bg_iids[cl_id] = {iid.strip() for iid in line.lstrip("Items:").split(",")}
            elif line.startswith("Cluster"):
                cl_id += 1
    return bid_to_bgs, bg_iids


def get_group_features(file_path):
    """ The function creates a group-feature dictionary

    :param file_path: a path to the file containing information about clusters
    :return: dict {ug_id_1: {feature_id_1: score_1, ...}, ...}
    """
    group_features = {}

    with open(file_path) as f:
        # skipping
        while not next(f).startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("->"):
                feature, score = list(map(lambda x: x.strip(), line.lstrip("->").split(": ")))
                score = float(score)
                group_features[cl_id].setdefault(feature, score)
            elif line.startswith("Cluster"):
                cl_id += 1
    return group_features
