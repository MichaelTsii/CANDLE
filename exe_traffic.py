import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import setproctitle

from main_model import CSDI_Value
from dataset_traffic import get_dataloader
from utils import train, evaluate

setproctitle.setproctitle("main-市区@qxq")

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:1', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
foldername = "./save/" + "shiqu_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
)

test_dataset = test_loader.dataset
test_data = np.array([test_dataset[i] for i in range(len(test_dataset))])
np.save("test_dataset.npy", test_data)

model = CSDI_Value(config, args.device).to(args.device)

args.modelfolder = "shiqu_0523_001050-E600N500-C"

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
