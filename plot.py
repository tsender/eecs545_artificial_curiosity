
import matplotlib.pyplot as plt 
import numpy as np
import math
import os

from map import Map
from engine import load_agent_data


## TODO novelty_analysis needs to be improved 

class PlotResults():
    """
    class implementation for the convenience of reading the data 
    at each starting position
    """
    def __init__(self, path: str, pos: str):
        """
        Initilize the Plot Results class by path and starting position

        Parameters:
        ------
        path: the "results" folder path
        pos: the "pos_xxx_xxx" folder name
        """
        self.path = path
        self.pos = pos
    

    def agent_data_dict(self, path: str, pos: str):
        """
        use load_agent_data function in engine module to load results csv/txt

        Parameters:
        ------
        path: the "results" folder path
        pos: the "pos_xxx_xxx" folder name

        Return:
        ------
        cur_agent_path: dict --> {curiosity agent_name: historical path}
        cur_agent_nov: dict --> {curiosity agent_name: average variance}
        agent_names: list --> the name of each agent; load from novelty.txt file
        """

        foldername = os.path.join(self.path, self.pos) # ../results/pos_xxx_xxx
        filenames = os.listdir(foldername) # access the name of each csv/txt file

        # access to path file and novelty file
        agent_files = [filename for filename in filenames if filename.endswith(".csv")]
        # lin_agent_file = [filename for filename in filenames if filename.startswith("Linear")].pop()
        # rand_agent_file = [filename for filename in filenames if filename.startswith("Rand")].pop()
        novelty_file = [filename for filename in filenames if filename.startswith("novelty")].pop()

        # load path data into dictionary
        agent_path = {}
        for agent_file in agent_files:
            # remove _path_record.csv from agent name
            # so that novelty dict and path dict will have the same key
            agent_name = agent_file[:-16] 
            agent_path[agent_name] = load_agent_data(os.path.join(foldername, agent_file))
        

        # load novelty data into dictionary
        agent_nov = {}
        with open(os.path.join(foldername, novelty_file)) as f:
            for line in f:
                (key, value) = line.split(":")
                agent_nov[key] = float(value)
        
        # record agent_names; maybe used for further purpose
        agent_names = list(agent_nov.keys())

        return agent_path, agent_nov, agent_names
    

    ## TODO novelty_analysis needs to be improved  
    def novelty_analysis(self, agent_nov):
        """
        analyze the novalty of each agent
        Needs more improvements
        """
        agent_nov_keys = list(agent_nov.keys())

        # separate curiosity agents and reference agents data
        cur_agent_nov = {}
        ref_agent_nov = {}
        for agent_key in agent_nov_keys:
            if agent_key.startswith('Cur'): # the name of Curiosity agent starts with "Curiosity"
                cur_agent_nov[agent_key] = agent_nov[agent_key]
            else:
                ref_agent_nov[agent_key] = agent_nov[agent_key]
        
        # sort novelty of curiosity agents 
        cur_agent_nov_sort = dict(sorted(cur_agent_nov.items(), key=lambda item: item[1], reverse=True))

        return ref_agent_nov, cur_agent_nov, cur_agent_nov_sort
    


    def plot_novelty(self, agent_nov: dict, ref_agent_nov: dict, show: bool=False, save: bool=False):
        """histogram of novelty of each agent
        """
        # read all novelty data
        agent_nov_values = list(agent_nov.values())

        # find novelty for reference agents
        ref_agent_keys = list(ref_agent_nov.keys())
        for ref_agent_key in ref_agent_keys:
            if ref_agent_key.startswith('Linear'):
                lin_nov = ref_agent_nov[ref_agent_key]
            else:
                rand_nov = ref_agent_nov[ref_agent_key]

        # make histogram
        if (show or save):
            fig, ax = plt.subplots()

            # fix the number of bins
            bins = np.linspace(math.ceil(min(agent_nov_values)), math.floor(max(agent_nov_values)),20)
            # hist plot
            ax.hist(agent_nov_values, bins=bins, alpha=0.5)
            # annotate
            ax.set_title("Histogram of novelty of agents")
            ax.set_xlabel("Variance of novalty")
            ax.set_ylabel("Count agents")

            # mark linear and random agents
            ymin, ymax = plt.ylim()
            ax.vlines(lin_nov, ymin=ymin, ymax=ymax, linestyles='dashed', color='r')
            ax.vlines(rand_nov, ymin=ymin, ymax=ymax, linestyles='dashed', color='k')
            ax.text(lin_nov-50, ymax*0.5, "Linear agent", color ='r', rotation=90)
            ax.text(rand_nov-50, ymax*0.5, "Random agent", color ='k', rotation=90)

            if (save):
                # like results folder, create a plot folder for each starting pos
                dirname = 'plots' + '/' + self.pos
                plotname = self.pos + '_novelty_hist' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(os.path.dirname(__file__), dirname, plotname))
                
            plt.show()
           

    def plot_paths_new(self, agent_path: dict, cur_agent_nov_sort:dict, ref_agent_nov:dict, num_agents: int, show: bool=False, save: bool=False):
        """
        plot selected agent path after loading the agent path data

        Parameters: 
        ------
        agent_path: dict --> the dictionary of agent path data
        num_agents: int --> number of agents to plot
        show: bool --> show plot
        save: bool --> save plot
        """
        # Only plot path with highest [num_gents] of agent paths
        cur_agent_keys = list(cur_agent_nov_sort.keys())
        ref_agent_keys = list(ref_agent_nov.keys())
        selected_agents = ref_agent_keys + cur_agent_keys[:num_agents]


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
                x, y = zip(*agent_path[agent])
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
                dirname = 'plots' + '/' + self.pos
                plotname = self.pos + '_path_map_' + str(num_agents) + 'cur_agents' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(os.path.dirname(__file__), dirname, plotname))
            
            plt.show()


    

if __name__ == "__main__":
    fov = 64
    image_path = os.path.join(os.path.dirname(__file__), 'data', 'mars.png') 
    map = Map(image_path, fov, 2)

    ## initialize PlotResults class
    results_path = os.path.join(os.path.dirname(__file__), 'results') # results folder
    agent_pos = os.listdir(results_path) # agent position folder

    for i in range(10):
        print(f"Processing agents at staring position {agent_pos[i]}...")
        # initiliza class
        plot_results = PlotResults(results_path, agent_pos[i])
        # load data
        agent_path, agent_nov, _ = plot_results.agent_data_dict(results_path, agent_pos[i])
        # analyze novelty
        ref_agent_nov, cur_agent_nov, cur_agent_nov_sort = plot_results.novelty_analysis(agent_nov)
        # plot novelty
        plot_results.plot_novelty(agent_nov, ref_agent_nov, show=True, save=False)
        # plot path mat
        plot_results.plot_paths_new(agent_path, cur_agent_nov_sort, ref_agent_nov, num_agents=70, show=True, save=False)

    
