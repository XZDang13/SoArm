import os

from uuid import uuid4
from PIL import Image
import json

class Writer:
    replay_path = "replays"
    @staticmethod
    def save_obs(state_obs, frame_obs):
        file_id = str(uuid4())
        json_path = f"{Writer.replay_path}/json/{file_id}.json"
        frame_0_path = f"{Writer.replay_path}/img/{file_id}_0.png"
        frame_1_path = f"{Writer.replay_path}/img/{file_id}_1.png"
        state_obs = state_obs.tolist()
        item = {
            "state": state_obs,
            "frame": [frame_0_path, frame_1_path]
        }

        with open(json_path, "w") as f:
            json.dump(item, f)

        frame_obs[0].save(frame_0_path)
        frame_obs[1].save(frame_1_path)
