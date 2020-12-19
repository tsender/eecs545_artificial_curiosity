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
from memory import PriorityMemory, CircularMemory
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
                plt.savefig("{}/{}.svg".format(dirname, str(agent)))

            # Show the image if desired
            if(show):
                plt.show()
        plt.close('all')


def run_agents(agent_list: List[Agent], time_steps: int):
    """Runs an experiment on the provided agents.
    
    Params
    ------
    agent_list: List[Agent]
        A list of Agent instances to be ran
    
    time_steps: int
        The number of time steps to simulate the agent for

    Returns
    -------
    None

    """

    # Make sure that the parameters are valid
    assert agent_list is not None and len(agent_list) > 0

    progress_bar_width = 50
    num_agents = len(agent_list)
    current_agent_id = 1
    
    for agent in agent_list:
        # print(F"Running agent: {str(agent)}")
        t_start = time.time()
        for i in range(time_steps):
            p = i+1
            # Error handling in case something goes wrong
            try:
                agent.step()
            except Exception as e:
                # TODO: Should probably replace this with a stack trace
                print('Problem at step ', i, " with agent:", agent)
                print(e)
                return
            t_elapsed = time.time() - t_start
            agent_eta = (t_elapsed / p) * (time_steps - p)
            agent_eta_str = get_time_str(agent_eta)
        
            # Update progress bar for agent
            frac = p / float(time_steps)
            left = int(progress_bar_width * frac)
            right = progress_bar_width - left
            print(f'\rAgent {current_agent_id}/{num_agents} Experiment Progress [', '#' * left, ' ' * right, ']', f' {frac*100:.0f}% ETR: {agent_eta_str}', sep='', end='', flush=True)
        print("") # Moves carriage to next line
        current_agent_id += 1

def run_agent(agent: Agent, time_steps: int):
    """Runs an experiment on the provided agents.
    
    Params
    ------
    agent: Agent
        A Agen to run
    
    time_steps: int
        The number of time steps to simulate the agent for

    Returns
    -------
    None

    """

    progress_bar_width = 50
    
    # print(F"Running agent: {str(agent)}")
    t_start = time.time()
    for i in range(time_steps):
        p = i+1
        # Error handling in case something goes wrong
        try:
            agent.step()
        except Exception as e:
            # TODO: Should probably replace this with a stack trace
            print('Problem at step ', i, " with agent:", agent)
            print(e)
            return False
        t_elapsed = time.time() - t_start
        agent_eta = (t_elapsed / p) * (time_steps - p)
        agent_eta_str = get_time_str(agent_eta)
    
        # Update progress bar for agent
        frac = p / float(time_steps)
        left = int(progress_bar_width * frac)
        right = progress_bar_width - left
        print(f'\rAgent Experiment Progress [', '#' * left, ' ' * right, ']', f' {frac*100:.0f}% ETR: {agent_eta_str}', sep='', end='', flush=True)
    print("") # Moves carriage to next line
    
    return True

def run_experiments(map: Map):
    """Run a series of experiments. Generate Random, Linear, and Curiosity agents for each starting position.
    Test a series of brain configurations for the Curiosity agent so we can see if there is an optimal configuration.
    """

    # Some high-level parameters
    num_starting_positions = 10
    random.seed(12345)
    base_results_dir = "results_param"
    max_time_steps = 1000
    fov = 64
    grain_size = (fov, fov, 1)
    move_rate = 8 # Larger than 1 increases possible coverage of the map by the agent
    log_file = os.path.join(base_results_dir, "log.txt")
    position_order_file = os.path.join(base_results_dir, "position_order.txt")

    # Defines the different possible parameters used when creating the various brains
    brain_config = {}
    brain_config['memory_type'] = [PriorityMemory, CircularMemory]
    brain_config['memory_length'] = [40] # TODO: [40, 60, 80] DONE [20]
    brain_config['novelty_loss_type'] = ['MSE', 'MAE']
    brain_config['train_epochs_per_iter'] = [1, 2, 3]
    brain_config['learning_rate'] = [0.0001, 0.0002, 0.0003, 0.0004]

    # Calculate number of different curious agents per position
    num_curious_agents_per_pos = 1
    for _,v in brain_config.items():
        num_curious_agents_per_pos *= len(v)

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

    print(F"Writing position order to file: {position_order_file}")
    with open(position_order_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(position_list) # x,y format

    # Create results directories
    print("Creating directories...")
    result_dirs = []
    for pos in position_list:
        dir = "pos_" + str(pos[0]) + "_" + str(pos[1])
        dir = os.path.join(base_results_dir, dir)
        result_dirs.append(dir)

        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    with open(log_file, "a") as f:
        f.write("STARTING NEW EXPERIMENT: " + base_results_dir + "\n")

    # Create agents
    print("Creating Linear/Random agents...")
    linear_agents = []
    random_agents = []
    for i in range(num_starting_positions):
        pos = position_list[i]

        # Linear Agents
        linear_motiv = Linear(map, rate=move_rate)
        lin_agent = Agent(linear_motiv, pos)
        data_dir = os.path.join(result_dirs[i], str(lin_agent))
        lin_agent.set_data_dir(data_dir)
        linear_agents.append(lin_agent)
        
        # Random Agents
        rand_motiv = Random(map, rate=move_rate)
        rand_agent = Agent(rand_motiv, pos)
        data_dir = os.path.join(result_dirs[i], str(rand_agent))
        rand_agent.set_data_dir(data_dir)
        random_agents.append(rand_agent)

    # Run Linear agents
    print("Running Linear agents...")
    for i in range(num_starting_positions):
        print(F"\nLinear Agent {i+1}/{num_starting_positions}:")
        success = run_agent(linear_agents[i], max_time_steps)
        linear_agents[i].save_reconstruction_snapshot()
        linear_agents[i].save_data()

        with open(log_file, "a") as f:
            f.write(str(linear_agents[i]) + ": " + str(success) + "\n")

    # Run Random agents
    print("Running Random agents...")
    for i in range(num_starting_positions):
        print(F"\nRandom Agent {i+1}/{num_starting_positions}:")
        success = run_agent(random_agents[i], max_time_steps)
        random_agents[i].save_reconstruction_snapshot()
        random_agents[i].save_data()

        with open(log_file, "a") as f:
            f.write(str(random_agents[i]) + ": " + str(success) + "\n")

    # Curiosity Agents
    print("Creating/running Curiosity agents...")
    start_time = time.time()
    for i in range(num_starting_positions):
        p = i+1
        pos = position_list[i]
        pos_start_time = time.time()
        cur_agent_num = 1

        # if i != num_starting_positions - 1:
        #     print(F"Skipping position {i}")
        #     continue

        for mem in brain_config['memory_type']:
            for mem_len in brain_config['memory_length']:
                for nov_type in brain_config['novelty_loss_type']:
                    for train_epochs in brain_config['train_epochs_per_iter']:
                        for lr in brain_config['learning_rate']:
                            # Must call clear_session to reset the global state and avoid memory clutter for the GPU
                            # Allows us to create more models without worrying about memory
                            tf.keras.backend.clear_session()

                            print(F"\nCurious Agent {cur_agent_num}/{num_curious_agents_per_pos} at Pos {p}/{num_starting_positions} {pos}:")
                            
                            brain = Brain(mem(mem_len), grain_size, novelty_loss_type=nov_type,
                                            train_epochs_per_iter=train_epochs, learning_rate=lr)
                            curious_motiv = Curiosity(map, brain, rate=move_rate)
                            curious_agent = Agent(curious_motiv, pos)
                            data_dir = os.path.join(result_dirs[i], str(curious_agent))
                            curious_agent.set_data_dir(data_dir)
                            print(str(curious_agent))

                            success = run_agent(curious_agent, max_time_steps)
                            curious_agent.save_reconstruction_snapshot()
                            curious_agent.save_data()

                            with open(log_file, "a") as f:
                                f.write(str(curious_agent) + ": " + str(success) + "\n")

                            # Print estimated time remaining
                            wall_time = time.time() - start_time
                            pos_wall_time = time.time() - pos_start_time
                            pos_eta = (pos_wall_time / cur_agent_num) * (num_curious_agents_per_pos - cur_agent_num)
                            print(F"Position Wall Time: {get_time_str(pos_wall_time)}, Position ETR: {get_time_str(pos_eta)}")

                            num_agents_tested = cur_agent_num + i*num_curious_agents_per_pos
                            num_agents_remaining = num_starting_positions*num_curious_agents_per_pos - num_agents_tested
                            wall_time_eta = (wall_time / num_agents_tested) * num_agents_remaining
                            print(F"Wall Time: {get_time_str(wall_time)}, ETR: {get_time_str(wall_time_eta)}")

                            cur_agent_num += 1

def run_best_experiments(map: Map):
    """Run a series of experiments among Random, Linear, and the Best Curiosity agents for various starting position."""

    # Some high-level parameters
    num_starting_positions = 10
    random.seed(12345)
    base_results_dir = "results_best"
    max_time_steps = 1000
    fov = 64
    grain_size = (fov, fov, 1)
    move_rate = 8 # Larger than 1 increases possible coverage of the map by the agent

    # Defines the best brain config
    best_brains = [] # In the whole universe
    best_brains.append({'memory': CircularMemory(64), 'img_size': grain_size, 'novelty_loss_type': "MSE", 
                        'train_epochs_per_iter': 3, 'learning_rate': 0.0002})
    best_brains.append({'memory': CircularMemory(100), 'img_size': grain_size, 'novelty_loss_type': "MSE", 
                        'train_epochs_per_iter': 3, 'learning_rate': 0.0005})
    prob_list = [(1.0, 0.0), (0.95, 0.05), (0.9, 0.1), (0.85, 0.15), (0.8, 0.2)] # Probabilty of choosing 1st and 2nd best movements

    for p in prob_list:
        assert p[0] + p[1] == 1.0

    num_curious_agents_per_pos = len(prob_list) * len(best_brains)

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
    result_dirs = []
    for pos in position_list:
        dir = "pos_" + str(pos[0]) + "_" + str(pos[1])
        dir = os.path.join(base_results_dir, dir)
        result_dirs.append(dir)

        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    # Curiosity Agents
    print("Creating/running the best Curiosity agents...")
    start_time = time.time()
    for i in range(num_starting_positions):
        p = i+1
        pos = position_list[i]
        pos_start_time = time.time()
        cur_agent_num = 1
            
        for brain_cfg in best_brains:
            for prob in prob_list:
                tf.keras.backend.clear_session()
                print(F"\nCurious Agent {cur_agent_num}/{num_curious_agents_per_pos} at Pos {p}/{num_starting_positions} {pos}:")
                                
                brain = Brain(**brain_cfg)
                curious_motiv = Curiosity(map, brain, rate=move_rate, prob=prob)
                curious_agent = Agent(curious_motiv, pos)
                data_dir = os.path.join(result_dirs[i], str(curious_agent))
                curious_agent.set_data_dir(data_dir)
                print(str(curious_agent))

                run_agents([curious_agent], max_time_steps)
                curious_agent.save_reconstruction_snapshot()
                curious_agent.save_data()

                # Print estimated time remaining
                wall_time = time.time() - start_time
                pos_wall_time = time.time() - pos_start_time
                pos_eta = (pos_wall_time / cur_agent_num) * (num_curious_agents_per_pos - cur_agent_num)
                print(F"Position Wall Time: {get_time_str(pos_wall_time)}, Position ETR: {get_time_str(pos_eta)}")

                num_agents_tested = cur_agent_num + i*num_curious_agents_per_pos
                num_agents_remaining = num_starting_positions*num_curious_agents_per_pos - num_agents_tested
                wall_time_eta = (wall_time / num_agents_tested) * num_agents_remaining
                print(F"Wall Time: {get_time_str(wall_time)}, ETR: {get_time_str(wall_time_eta)}")

                cur_agent_num += 1
        
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

    # NOTE: Ted suggests you add a method inside the Agent class to reload the agent history since several functions
    # in engine.py reference agent.history

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
    # pos = (2000, 1000)
    # brain = Brain(PriorityMemory(64), (fov,fov,1), novelty_loss_type='MSE', train_epochs_per_iter=1)
    # agent = Agent(Curiosity(map=map, brain=brain, rate=8), pos)
    # novelty_filename = "output_dir/novelty.txt"
    # run_agents([agent], 100)
    # save_and_plot([agent], "output_dir/novelty.txt", "output_dir")

    fov = 64 # Allowed FOVs = {32, 64, 128}
    map = Map('data/mars.png', fov, 2)
    run_experiments(map)
    # run_best_experiments(map)
