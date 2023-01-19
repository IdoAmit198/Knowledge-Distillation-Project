import wandb

api = wandb.Api()
project="our-awesome-project"
entity = "ido-shani-proj"
ids_2 = ["xgzvyk73", "7xac7fab", "wey2an4x"]
ids_3 = ["hdseg1mj", "vyu0e60u", "wefca963"]
ids_5 = ["w9s9rr30", "pnt1jlow", "963hodfu"]
ids_10 = ["b5c89c6b", "2wm66jkp"]
num_models_runs = {2: ids_2, 3: ids_3, 5: ids_5, 10: ids_10}

ids_sep_5 = ["1azukj4y", "ivavwx1l", "2d515yj7"]
ids_sep_10 = ["326nvjq7", "1zjcau7v", "1xepkvfs", "26x906hr", "fu9yvu6u"]
ids_sep_20 = ["23oth3ld", "35dt1m79", "4lnvwqhb", "a8k4c9zc", "2z4fjwlp"]
ids_sep_50 = ["w83iomlp", "gbhy83xi", "23uz555d", "3oen9qio", "11ex89i6"]
sep_runs = {5: ids_sep_5, 10: ids_sep_10, 20: ids_sep_20, 50: ids_sep_50}

for sep, runs in sep_runs.items():
    for run_id in runs:
        run = api.run(f"{entity}/{project}/{run_id}")
        run.config["mutual models num"] = 2
        run.config["separate epochs"] = sep
        run.update()

