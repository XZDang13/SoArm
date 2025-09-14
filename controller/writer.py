import json
from .controller import untile_image

class Writer:
    @staticmethod
    def save_data(trajectory_id: str, step: int, state_obs, frame_obs, tile_rows, tile_cols):
        num_envs = tile_rows * tile_cols
        pre_step = max(0, step-1)
        state_obs = state_obs.cpu().tolist()
        frames = untile_image(frame_obs, tile_rows, tile_cols)

        for env_id in range(num_envs):
            data = {
                "states": state_obs[env_id],
                "frames": [f"replays/img/{trajectory_id}_{env_id}_{step}.jpg",
                           f"replays/img/{trajectory_id}_{env_id}_{pre_step}.jpg"]
            }

            frame = frames[env_id]
            frame.save(f"replays/img/{trajectory_id}_{env_id}_{step}.jpg")

            with open(f"replays/json/{trajectory_id}_{env_id}_{step}.json", "w") as f:
                json.dump(data, f)