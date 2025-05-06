import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re

base_dir = input("Enter the base directory containing episode folders: ").strip()
episode_folders = sorted([
    os.path.join(base_dir, d)
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])

def extract_step_number(filename):
    match = re.search(r"step_(\d+)\.npy", filename)
    return int(match.group(1)) if match else -1

class InteractiveLabeler:
    def __init__(self, episode_path):
        self.episode_path = episode_path
        self.step_files = sorted(
            [f for f in os.listdir(episode_path) if f.endswith('.npy')],
            key=extract_step_number
        )
        self.steps = []
        for f in self.step_files:
            data = np.load(os.path.join(episode_path, f), allow_pickle=True).item()
            self.steps.append(data)

        self.labels = [None] * len(self.steps)
        self.index = 0
        self.fig, (self.ax_img, self.ax_wrist) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()
        plt.show()

    def update_plot(self):
        data = self.steps[self.index]
        img = data["inputs/observation/image"]
        wrist_img = data["inputs/observation/wrist_image"]

        self.ax_img.clear()
        self.ax_wrist.clear()

        self.ax_img.imshow(img)
        self.ax_img.set_title(f"Main Image (Step {self.index+1}/{len(self.steps)})")
        self.ax_img.axis('off')

        self.ax_wrist.imshow(wrist_img)
        self.ax_wrist.set_title("Wrist Image")
        self.ax_wrist.axis('off')

        label = self.labels[self.index]
        if label == 1:
            self.fig.suptitle("Label: HELP", color='red')
        elif label == 0:
            self.fig.suptitle("Label: NO HELP", color='green')
        elif label == "cut":
            self.fig.suptitle("Label: CUT (excluded)", color='orange')
        else:
            self.fig.suptitle("Label: [Unlabeled]", color='black')

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.index = min(self.index + 1, len(self.steps) - 1)
        elif event.key == 'left':
            self.index = max(self.index - 1, 0)
        elif event.key == 'h':
            self.labels[self.index] = 1
            print(f"Step {self.index+1} labeled as HELP")
        elif event.key == 'n':
            self.labels[self.index] = 0
            print(f"Step {self.index+1} labeled as NO HELP")
        elif event.key == 'c':
            self.labels[self.index] = "cut"
            print(f"Step {self.index+1} marked as CUT (will not be saved)")
        elif event.key == 'q':
            self.save_labels()
            plt.close(self.fig)
            return
        self.update_plot()

    def save_labels(self):
        filtered_labels = {
            "episode": os.path.basename(self.episode_path),
            "labels": []
        }
        for i, label in enumerate(self.labels):
            if label in [0, 1]:  # only keep valid labels
                filtered_labels["labels"].append({"step": self.step_files[i], "label": label})

        output_path = os.path.join(self.episode_path, "labels.json")
        with open(output_path, 'w') as f:
            json.dump(filtered_labels, f, indent=2)
        print(f"✅ Labels saved to {output_path} ({len(filtered_labels['labels'])} steps kept)")


# Run labeling for each episode interactively
for episode in episode_folders:
    print(f"Labeling episode: {os.path.basename(episode)}")
    InteractiveLabeler(episode)
