import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_heatmap(filepath="./data/mars.png", size=16, export_ratio=(1000, 500)):
    img = Image.open(filepath)

    variances = []

    for y in range(0, img.height - size, size):
        temp = []
        for x in range(0, img.width - size, size):
            test_image = img.crop((x, y, x + size, y + size))
            temp.append(np.var(test_image))
        variances.append(temp)

    variances = np.asarray(variances)

    cm = plt.get_cmap('magma')
    colored_image = cm(variances/variances.max())

    new_img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    return new_img.resize(export_ratio, Image.ANTIALIAS)


def smooth(data, number=20):
    return np.convolve(data, np.ones(number)/number)[number-2:]


def normalize(data):
    sums = np.sum(data)
    return data/sums


def plot_novelty(img, input, color, title, dirname=None):
    x_lim = [0, img.size[1]]
    y_lim = [0, img.size[0]]

    fig, ax = plt.subplots(facecolor='w', figsize=(8, 3.5), dpi=130)
    ax.imshow(img, extent=y_lim+x_lim, cmap='Greys_r')
    ax.invert_yaxis()

    plt.scatter(input["x"], input["y"], c=color, cmap="magma", s=1, alpha=0.3)
    plt.title(title)

    if(dirname is None):
        plt.show()
    else:
        plt.savefig(os.path.join(
            dirname, "{}_heatmap.svg".format(title.lower())))
        plt.close()


def plot_all_images(source_dir="./results2", img_name="./data/mars.png", grain_novelty_name="grain_novelty.csv", path_record_name="path_record.csv"):
    img = Image.open(img_name)

    position_dirs = [x for x in os.listdir(source_dir) if "pos_" in x]

    for dir in position_dirs:
        agent_dirs = [x for x in os.listdir(
            os.path.join(source_dir, dir)) if "Pos_" in x]
        for agent in agent_dirs:
            if "curiosity" in agent.lower():
                dir_content = os.listdir(os.path.join(source_dir, dir, agent))
                print("+1")

                if (path_record_name in dir_content) and (grain_novelty_name in dir_content):
                    grain_info = pd.read_csv(os.path.join(
                        source_dir, dir, agent, grain_novelty_name), header=None)
                    grain_pos = pd.read_csv(os.path.join(
                        source_dir, dir, agent, path_record_name), names=["x", "y"])

                    to_plot = {
                        "Mean": normalize([np.mean(list(row)) for index, row in grain_info.iterrows()]),
                        "Var": normalize([np.var(list(row)) for index, row in grain_info.iterrows()]),
                        "Median": normalize([np.median(list(row)) for index, row in grain_info.iterrows()]),
                        "Max": normalize([np.max(list(row)) for index, row in grain_info.iterrows()]),
                        "Min": normalize([np.min(list(row)) for index, row in grain_info.iterrows()])
                    }

                    for k in to_plot:
                        plot_novelty(img, grain_pos, smooth(
                            to_plot[k]), k, dirname=os.path.join(source_dir, dir, agent))
                else:
                    print("There was an error with agent",
                          os.path.join(source_dir, dir, agent))

                


if __name__ == "__main__":
    plot_all_images()
