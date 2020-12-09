
list_map = map

import matplotlib.pyplot as plt 
import numpy as np
import csv
import os

from map import Map
from engine import load_agent_data


## TODO novelty_analysis needs to be improved 
## TODO compare between different starting point 

class PlotResults():
    """
    class implementation for the convenience of reading the data 
    at each starting position
    """
    def __init__(self, dir_path: str, results_folder: str, pos_folder: str):
        """
        Initilize the Plot Results class by path and starting position

        Parameters:
        ------
        dir_path: the current working directory path
        results: the "results2" folder 
        pos: the "pos_xxx_xxx" folder name
        """
        self.dir_path = dir_path
        self.results_folder = results_folder
        self.pos_folder = pos_folder
    

    def agent_data_dict(self, results_folder: str, pos_folder: str):
        """
        use load_agent_data function in engine module to load results csv/txt

        Parameters:
        ------
        results_folder: the "results" folder
        pos: the "pos_xxx_xxx" folder name

        Return: six dictionaries ref_agents and cur_agents stored separately 
        ------
        {agent_name: values}
        Curiosity agents: cur_grain_novelty, cur_path_record, 
                          cur_perceived_path_novelty, cur_ave_path_novelty 
        Reference agents: ref_path_record, ref_ave_path_novelty
        """
        # access the ../results/pos_xxx_xxx folder
        results_path = os.path.join(self.dir_path, self.results_folder, self.pos_folder) 
        # access the name of files in ../results/pos_xxx_xxx
        agent_names = os.listdir(results_path) 

        # store curiosity agents data
        cur_grain_novelty = {}
        cur_path_record = {}
        cur_perceived_path_novelty = {}
        cur_avg_path_variance = {}
        # store reference agents data
        ref_path_record = {}
        ref_avg_path_variance = {}

        for agent_name in agent_names:
            # loop over each agent and record all info into dict
            # for cur_agents: [avg_path_variance.csv, grain_novelty.csv, path_record.csv, perceived_path_novelty.csv, snapshots]
            # for ref_agents: [avg_path_variance.csv, path_record.csv]
            if agent_name.startswith("Curiosity"):
                cur_grain_novelty[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'grain_novelty.csv'))
                cur_path_record[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'path_record.csv'))
                cur_perceived_path_novelty[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'perceived_path_novelty.csv'))
                # avg_path_variance is a float, pop out from a list[list[avg_path_variance]]
                cur_avg_path_variance[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'avg_path_variance.csv'))[0][0]

            else:
                ref_path_record[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'path_record.csv'))
                # avg_path_variance is a float, pop out from a list[list[avg_path_variance]]
                ref_avg_path_variance[agent_name] = self.read_csv(os.path.join(results_path, agent_name, 'avg_path_variance.csv'))[0][0]

        return cur_grain_novelty, cur_path_record, cur_perceived_path_novelty, cur_avg_path_variance, ref_path_record, ref_avg_path_variance
    


    def read_csv(self, filepath: str):
        """ 
        Help to read csv file

        Parameters:
        ------
        filepath: file path of the csv
        
        Return:
        ------
        list_content: a list of whatever data recorded (convert to float)
        """
        if (filepath != None) and (filepath.split(".")[-1] == "csv"):
            # create a list to hold data
            list_content = []
            # open the csv file
            with open(filepath, 'r', newline='') as f:
                csv_reader = csv.reader(f, delimiter=',')
                for row in csv_reader:
                    list_content.append(row)

        else: # in case there is data not been recorded
            unrecorded_data = filepath.split("/")[-2:]
            print(f"{unrecorded_data} is not recorded")

        return list(list_map(lambda x: list(list_map(float, x)), list_content))
    
    

    ## TODO novelty_analysis needs to be improved  
    def novelty_analysis(self, cur_avg_path_variance: dict, ref_avg_path_variance: dict):
        """
        analyze the novalty of each agent
        Needs more improvements
        """
        # sort novelty of curiosity agents 
        cur_avg_path_variance_sort = dict(sorted(cur_avg_path_variance.items(), key=lambda item: item[1], reverse=True))

        return cur_avg_path_variance_sort
    


    def plot_path_variance(self, cur_avg_path_variance: dict, ref_avg_path_variance: dict, show: bool=False, save: bool=False):
        """histogram of novelty of each agent
        """
        # read all novelty data
        agent_path_variance = list(cur_avg_path_variance.values()) + list(ref_avg_path_variance.values())

        # find novelty for reference agents
        ref_agent_keys = list(ref_avg_path_variance.keys())
        for ref_agent_key in ref_agent_keys:
            if ref_agent_key.startswith('Linear'):
                lin_path_variance = ref_avg_path_variance[ref_agent_key]
            else:
                rand_path_variance = ref_avg_path_variance[ref_agent_key]

        # make histogram
        if (show or save):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

            # hist plot
            ax.hist(agent_path_variance, bins=20, alpha=0.5)
            # annotate
            ax.set_title("Histogram of novelty of agents")
            ax.set_xlabel("Variance of novalty")
            ax.set_ylabel("Count agents")

            # mark linear and random agents
            ymin, ymax = plt.ylim()
            ax.vlines(lin_path_variance , ymin=ymin, ymax=ymax, linestyles='dashed', color='r')
            ax.vlines(rand_path_variance, ymin=ymin, ymax=ymax, linestyles='dashed', color='k')
            ax.text(lin_path_variance-0.001, ymax*0.6, "Linear agent", color ='r', rotation=90)
            ax.text(rand_path_variance-0.001, ymax*0.6, "Random agent", color ='k', rotation=90)

            if (save):
                # like results folder, create a plot folder for each starting pos
                dirname = 'plots' + '/' + self.pos_folder
                plotname = self.pos_folder + '_novelty_hist' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(self.dir_path, dirname, plotname))
                
            plt.show()



    def plot_paths_new(self, cur_path_record: dict, ref_path_record: dict, show: bool=False, save: bool=False):
        """
        plot selected agent path after loading the agent path data

        Parameters: 
        ------
        agent_path: dict --> the dictionary of agent path data
        num_agents: int --> number of agents to plot
        show: bool --> show plot
        save: bool --> save plot
        """
        # # Only plot path with highest [num_gents] of agent paths
        # cur_agent_keys = list(cur_agent_nov_sort.keys())
        # ref_agent_keys = list(ref_agent_nov.keys())
        # selected_agents = ref_agent_keys + cur_agent_keys[:num_agents]

        # merge ref_agents and cur_agents
        all_path_record = {**ref_path_record, **cur_path_record}
        selected_agents = list(all_path_record.keys())

        # plot path map
        if (show or save):
            # Gets the x,y limits of the image so the graph can be sized accordingly 
            x_lim = [0, map.img.size[1]]
            y_lim = [0, map.img.size[0]]

            # Set up the plot
            fig, ax = plt.subplots(facecolor='w', figsize=(8, 3.5), dpi=130)
            ax.imshow(map.img, extent=y_lim+x_lim, cmap='Greys_r') # greyscale
            ax.invert_yaxis() # origin at top left, invert y axis

            # TODO: Only select top 
            for agent in selected_agents:
                # Splits x and y into separate lists (technically tuples)
                x, y = zip(*all_path_record[agent])
                ax.plot(x, y, label=agent, alpha=0.5)

            # Annotate the chart
            ax.set_title("Combined Agent Paths")
            # Place the legend outside of the chart since we never know where the
            # agents are going to go
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Legend', fontsize=6)
            # Fix the alignment of the image on export
            ax.set_position([0.1, 0, 0.5, 1.0])

            if (save):
                # like results folder, create a plot folder for each starting pos
                dirname = 'plots' + '/' + self.pos_folder
                plotname = self.pos_folder + '_path_map_' + 'cur_agents' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(self.dir_path, dirname, plotname))
                pass
            
            plt.show()


if __name__ == "__main__":
    fov = 64
    image_path = os.path.join(os.path.dirname(__file__), 'data', 'mars.png') 
    map = Map(image_path, fov, 2)

    ########## Example results of 'pos_1772_86'
    ## initialize PlotResults class
    dir_path = os.path.dirname(__file__) # ../eecs545_artificial_curiosity
    results_folder = 'results2' # results folder ../eecs545_artificial_curiosity/results2
    # pos_folder = os.listdir(results_folder) # agent position folder
    pos_folder = ['pos_1772_86']

    for i in range(len(pos_folder)):
        print(f"Processing agents at staring position {pos_folder[i]}...")
        # initiliza class
        plot_results = PlotResults(dir_path, results_folder, pos_folder[i])
        # load data 
        # all stored five csv files are loaded
        cur_grain_novelty, cur_path_record, cur_perceived_path_novelty, cur_avg_path_variance, \
        ref_path_record, ref_avg_path_variance = plot_results.agent_data_dict(results_folder, pos_folder[i])
        
        # # analyze novelty
        # ref_agent_nov, cur_agent_nov, cur_agent_nov_sort = plot_results.novelty_analysis(agent_nov)

        # # plot novelty
        plot_results.plot_path_variance(cur_avg_path_variance, ref_avg_path_variance, show=True, save=False)
        # # plot path mat
        plot_results.plot_paths_new(cur_path_record, ref_path_record, show=True, save=False)
        