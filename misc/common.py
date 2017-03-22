def get_ug_data(ug_file_path):
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
    bid_to_bg = {}
    bg_iids = {}
    with open(bg_file_path) as f:
        # skipping
        while not next(f).startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Bookings:"):
                for bid in line.lstrip("Bookings:").split(","):
                    bid_to_bg[bid.strip()] = cl_id
            elif line.startswith("Items:"):
                bg_iids[cl_id] = {iid.strip() for iid in line.lstrip("Items:").split(",")}
            elif line.startswith("Cluster"):
                cl_id += 1
    return bid_to_bg, bg_iids
