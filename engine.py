list_map = map

from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import time
import os
import random
import tensorflow as tf

from map import Map
from agent import Agent, Curiosity, Linear, Random, Motivation
from memory import PriorityBasedMemory, ListBasedMemory
from brain import Brain


# TODO: Might want to make more efficient so that graphing doesn't take so long


def plot_paths(map: Map, agent_list: List[Agent], show: bool, save: bool, dirname: str):
    """Plots out the paths of the agents on the map
    
    Params
    ------
    map: Map
        A map that will be used to get the bounds and background for plotting
    
    agent_list: List[Agent]
        A list of agents whose paths need to be plotted
    
    show: bool
        Whether the plots should be displayed or not
    
    save: bool=False
        Whether the graphs should be saved

    dirname: str
        The directory where the images will be stored

    Returns
    -------
    None
    """

    print("Saving plots...")
    # Doesn't spend the time making the charts unless the user wants it
    if(show or save):
        # Create one large image with all agents at once

        # Gets the x,y limits of the image so the graph can be sized accordingly
        x_lim = [0, map.img.size[1]]
        y_lim = [0, map.img.size[0]]

        # Set up the plot
        fig, ax = plt.subplots(facecolor='w', figsize=(8, 3.5), dpi=130)
        # Chow the image in greyscale
        ax.imshow(map.img, extent=y_lim+x_lim, cmap='Greys_r')
        # Since the origin is at the top left, we're going to invert the y axis
        ax.invert_yaxis()

        # Cycle through every agent, put their x,y coordinates into separate lists,
        # and then plot them
        for agent in agent_list:
            # Splits x and y into separate lists (technically tuples)
            x, y = zip(*agent.history)
            # print(x, "\n\n\n")
            # Plot those points on the graph. The lines are transparent because
            # sometimes the agents cross over each other and you still want to
            # see the paths of those underneath
            ax.plot(x, y, label=agent, alpha=0.5)

        # Annotate the chart
        ax.set_title("Combined Agent Paths")
        # Place the legend outside of the chart since we never know where the
        # agents are going to go
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Legend')
        # Fix the alignment of the image on export
        ax.set_position([0.1, 0, 0.5, 1.0])

        # Save the image if they want it
        # NOTE: I chose svg because it retains its high wuality while being a relatively small file
        if(save):
            os.makedirs(dirname, exist_ok=True)
            plt.savefig("{}/all.svg".format(dirname))

        # Show it if they want it
        # NOTE: this has to be done in this order or it will save blank images
        if(show):
            plt.show()

        # Create multiple plots with one agent each
        for agent in agent_list:
            # Create a plot, same as before but shaped a little differently because
            # we don't have the legend
            fig, ax = plt.subplots(facecolor='w', figsize=(8, 4.5), dpi=80)
            # plot the image
            ax.imshow(map.img, extent=y_lim+x_lim, cmap='Greys_r')
            ax.invert_yaxis()

            # Put the x and y values in two sepearate lists
            x, y = zip(*agent.history)
            # Plot the agents. The color is arbitrary but I thought it stood out well.
            # These plots have transparencies as well to stay consistent with the others
            ax.plot(x, y, alpha=0.5, c="orange")

            for x,y in agent.history:
                rect = patches.Rectangle((x-32, y-32), 64, 64, linewidth=1, edgecolor='r', facecolor='none', alpha=0.1)
                ax.add_patch(rect)

            # Name the plot after the agent
            ax.set_title(agent)

            # Save the image if desired
            if(save):
                # get rid of all of the characters we don't want in a file path
                # filename = str(agent).replace(" ", "_").replace(
                #     "(", "").replace(")", "").replace(",", "_")
                # Save the file
                plt.savefig("{}/{}.svg".format(dirname, str(agent)))

            # Show the image if desired
            if(show):
                plt.show()


def run_agents(agent_list: List[Agent], map: Map, iterations: int, show: bool = True, save_graph: bool = True, novelty_filename: str = "results/novelty.txt", dirname: str = "results"):
    """Runs an experiment on the motication given, then handles plotting and saving data.
    
    Params
    ------
    agent_list: List[Agent]
        A list of Agent instances to be ran

    map: Map
        An instance of the Map class that will be used by the agent to handle directions, and for the plotting
    
    iterations: int
        The number of steps that each agent should take.

    show: bool=True
        Whether the graphs should be displayed

    save_graph: bool
        Whether the plots should be saved to the disk or not

    dirname: str=None
        The directory in which the graphs will be stored

    
    Returns
    -------
    None

    """

    # Make sure that the parameters are valid
    assert agent_list is not None and len(agent_list) > 0
    assert save_graph == False or dirname is not None

    progress_bar_width = 50
    num_agents = len(agent_list)
    current_agent_id = 1
    
    for agent in agent_list:
        # print(F"Running agent: {str(agent)}")
        t_start = time.time()
        for i in range(iterations):
            # Error handling in case something goes wrong
            try:
                agent.step()
            except Exception as e:
                # TODO: Should probably replace this with a stack trace
                print('Problem at step ', i, " with agent:", agent)
                print(e)
                return
            t_elapsed = time.time() - t_start
            agent_eta = (t_elapsed / (i+1)) * (iterations - i)
            agent_eta_str = get_time_str(agent_eta)
        
            # Update progress bar for agent
            frac = (i+1) / float(iterations)
            left = int(progress_bar_width * frac)
            right = progress_bar_width - left
            print(f'\rAgent {current_agent_id}/{num_agents} Experiment Progress [', '#' * left, ' ' * right, ']', f' {frac*100:.0f}% ETA: {agent_eta_str}', sep='', end='', flush=True)
        print("") # Moves carriage to next line
        current_agent_id += 1

    # Graph the agent's paths
    plot_paths(map, agent_list, show, save_graph, dirname)

    os.makedirs(dirname, exist_ok=True)

    save_agent_data(agent_list, novelty_filename, dirname)

def run_experiments(map: Map, num_starting_positions):
    """Run a series of experiments. Generate Random, Linear, and Curiosity agents for each starting position.
    Test a series of brain configurations for the Curiosity agent so we can see if there is an optimal configuration.
    """

    # Some high-level parameters
    random.seed(12345)
    base_results_dir = "results"
    path_length = 1000
    fov = 64
    grain_size = (fov, fov, 1)
    move_rate = 8 # Larger than 1 increases possible coverage of the map by the agent

    # Defines the different possible parameters used when creating the various brains
    brain_config = {}
    brain_config['memory_type'] = [PriorityBasedMemory, ListBasedMemory]
    brain_config['memory_length'] = [32, 64, 128, 256]
    brain_config['novelty_loss_type'] = ['MSE', 'MAE']
    brain_config['train_epochs_per_iter'] = [1, 2, 3, 4]
    brain_config['learning_rate'] = [0.001, 0.0005]

    # Get range of possible (x,y) pairs. Subtract 2 since I don't quite know the whole usable range given the agent's size.
    x_range = (map.fov + 2, map.img.size[0] - fov - 2)
    y_range = (map.fov + 2, map.img.size[1] - fov - 2)
    x_vals = []
    y_vals = []
    for _ in range(num_starting_positions):
        x = random.randint(x_range[0], x_range[1])
        if x not in x_vals:
            x_vals.append(x)

        y = random.randint(y_range[0], y_range[1])
        if y not in y_vals:
            y_vals.append(y)
    position_list = list(zip(x_vals, y_vals))

    # Create results directories
    print("Creating directories and novelty files...")
    results_dirs = []
    novelty_filenames = []
    for pos in position_list:
        dir = "pos_" + str(pos[0]) + "_" + str(pos[1])
        dir = os.path.join(base_results_dir, dir)
        results_dirs.append(dir)

        if not os.path.isdir(dir):
            os.makedirs(dir)

        # Create novelty files, one for each position
        nov_file = "novelty_" + str(pos[0]) + "_" + str(pos[1]) + ".txt"
        nov_file = os.path.join(dir, nov_file)
        novelty_filenames.append(nov_file)

    # Create agents
    print("Creating Linear/Random agents...")
    linear_agents = []
    random_agents = []
    for pos in position_list:
        # Lienar Agents
        linear_motiv = Linear(map, rate=move_rate)
        linear_agents.append(Agent(linear_motiv, pos))
        
        # Random Agents
        rand_motiv = Random(map, rate=move_rate)
        random_agents.append(Agent(rand_motiv, pos))

    print("Running Linear/Random agents...")

    # Curiosity Agents
    print("Creating/running Curiosity agents...")
    curiosity_agents = []
    for pos in position_list:
        print(pos)
        for mem in brain_config['memory_type']:
            for mem_len in brain_config['memory_length']:
                for nov_type in brain_config['novelty_loss_type']:
                    for train_epochs in brain_config['train_epochs_per_iter']:
                        for lr in brain_config['learning_rate']:
                            # Must call clear_session to reset the global state and avoid memory clutter for the GPU
                            tf.keras.backend.clear_session()

                            brain = Brain(mem(mem_len), grain_size, novelty_loss_type=nov_type,
                                            train_epochs_per_iter=train_epochs, learning_rate=lr)
                            cur_motiv = Curiosity(map, brain, rate=move_rate)
                            cur_agent = Agent(cur_motiv, pos)

def save_agent_data(agent_list: List[Agent], novelty_filename: str, dirname: str):
    """
    Save the path record of each agent as a csv file
    
    Params:
    ------
    agent_list: List[Agent]
        A list of agent whose path coordinates to be saved

    agent_list: List[Agent]
        A list of agent whose path coordinates to be saved
    
    dirname: str
        The directory name where the csv file will be saved

    Returns:
    ------
    None
    """

    # Iterates over all of the agents
    # Save the agent's path
    for agent in agent_list:
        fields = ['x', 'y']
        # Change the Agent's name into a valid filename
        # filename = str(agent).replace(" ", "_").replace(
        #     "(", "").replace(")", "").replace(",", "_") #+ '_{}_path_record'.format(time.time())

        os.makedirs(dirname, exist_ok=True)
        # Save the coordinates to a file
        with open(dirname + '/' + str(agent) + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(agent.history)

        # Save the agent's novelty
        with open(novelty_filename, "a") as f:
            f.write(str(agent) + ": " + str(agent.get_path_novelty()) + "\n")
        
def load_agent_data(path: str):
    """
    Loads information from a given file. This will not be used as part of the engine.
    However, I thought it would be useful to include here so that if we make changes to
    the serialization, we have the data loding close by and can edit it easily

    Params
    ------

    path: str
        The path to the file

    Returns
    -------

    List[Tuple[int]]
        Returns a list of x and y coordinates

    """

    # Makes sure that the path is given and that the file is a csv file
    assert path != None and path.split(".")[-1] == "csv"

    # Create a list to hold the coordinates as they're read from the file
    lst_content = []

    # Open the csv file
    with open(path, 'r', newline='') as agent_p_file:
        # Create a csv reader to read the information out of it
        csv_reader = csv.reader(agent_p_file, delimiter=',')
        # Iterate through all rows
        for row in csv_reader:
            # Add each row of data to our list of points
            lst_content.append(row)

    # Changing the rows from lists of strings to tuples of ints
    # (and also removing the headers)
    return list(list_map(lambda x: tuple(list_map(int, x)), lst_content[1:]))

def get_time_str(time_in): # time_in in seconds
    """Convert time_in to a human-readible string, e.g. '1 days 01h:45m:32s' 
    
    Args:
        time_in: float
            Time in seconds

    Returns:
        A string
    """
    day = round(time_in) // (24*3600)
    time2 = time_in % (24*3600)
    hour, min = divmod(time2, 3600)
    min, sec = divmod(min, 60)
    if day > 0:
        return "%d days %02dh:%02dm:%02ds" % (day, hour, min, sec)
    else:
        return "%02dh:%02dm:%02ds" % (hour, min, sec)

if __name__ == "__main__":
    # Sample code
    fov = 64 # Allowed FOVs = {32, 64, 128}
    map = Map('data/mars.png', fov, 2)

    # pos = (2000, 1000)
    # brain = Brain(PriorityBasedMemory(64), (fov,fov,1), novelty_loss_type='MSE', train_epochs_per_iter=1)
    # agent = Agent(Curiosity(map=map, brain=brain, rate=8), pos)
    # novelty_filename = "output_dir/novelty.txt"
    # run_agents([agent], map, 100, save_graph=True, show=False, novelty_filename=novelty_filename, dirname="output_dir")

    number_starting_positions = 10
    run_experiments(map, number_starting_positions)
