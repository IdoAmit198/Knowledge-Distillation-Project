import wandb

api = wandb.Api()
project="our-awesome-project"
entity = "ido-shani-proj"
ids_2 = ["xgzvyk73", "7xac7fab", "wey2an4x"]
ids_3 = ["hdseg1mj", "vyu0e60u", "wefca963"]
ids_5 = ["w9s9rr30", "pnt1jlow", "963hodfu"]
ids_10 = ["b5c89c6b", "2wm66jkp"]
runs = {2: ids_2, 3: ids_3, 5: ids_5, 10: ids_10}

for num, runs in runs.items():
    for run_id in runs:
        run = api.run(f"{entity}/{project}/{run_id}")
        run.config["mutual models num"] = num
        run.config["separate epochs"] = 0
        run.update()

