from openpi_client import websocket_client_policy as _websocket_client_policy
import argparse, os 
import numpy as np
import matplotlib.pyplot as plt

def main(args,policy):

    # Load data
    for j in range(args.num_episodes):
        data = np.load(args.npy_path+f"/episode_{j}.npy", allow_pickle=True)
        print(f"infering for {data[0]['language_instruction']}")
        print(data.shape)
        i=0
        for step in data:
            if i % 30 == 0: 
                obs = {
                    "observation/state": step['state'],
                    "observation/image": step['image'],
                    "observation/wrist_image": step['wrist_image'],
                    "prompt": step['language_instruction'],
                }
                print(f"Infering step {i}")
                policy.infer(obs)
            i+=1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True, help="Path to step_XX.npy file")
    parser.add_argument("--num_episodes", type=int, required=True, help="number of episodes")

    args = parser.parse_args()

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
    )
    metadata = ws_client_policy.get_server_metadata()


    main(args,ws_client_policy)