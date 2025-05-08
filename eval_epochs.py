import os



ck_dir_path = "experiments/DanceTrack/5/exp01_SpaceAtt"
config_path = "Eval_epochs/DanceTrack/5exp01_SpaceAtt/dance_val.yaml"
device = 0

log_path = "Eval_epochs/DanceTrack/5exp01_SpaceAtt/eval_log.txt"

for epoch in range(60, 301, 20):
    print(f"Eval : {epoch}")
    tracker_name = f'epoch{epoch}'
    weight_path = os.path.join(ck_dir_path, f'dancetrack_epoch{epoch}.pt')
    os.system(f"python main.py \
                --config {config_path} \
                --device {device} \
                --name {tracker_name} \
                --weight {weight_path} \
                >> {log_path}")














