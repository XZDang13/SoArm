import json
import glob

from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class PairDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.files = glob.glob(f"{data_path}/json/*.json")

        #self.to_over_lay = Image.open("real.png").convert("RGBA")

        self.transform = v2.Compose([
            v2.ToImage(),
            #v2.ColorJitter(brightness=0.2),
            #v2.CenterCrop((480, 480)),
            v2.Resize((112, 112)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.files)
    
    def process_img(self, img_file):
        img = Image.open(img_file)

        #alpha = 0.5
        #img = Image.blend(img, self.to_over_lay, alpha).convert("RGB")

        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file = self.files[index]

        with open(file, "r") as f:
            data = json.load(f)

        state = torch.as_tensor(data["state"])
        
        #frame = [self.process_img(img_file) for img_file in data["frames"]]
        #frame = torch.concat(frame, dim=0)
        frame = self.process_img(data["frame"])

        return state, frame

def get_dataloader():
    dataset = PairDataset("replays")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)

    return dataloader