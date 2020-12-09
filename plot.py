list_map = map

from typing import List
import matplotlib.pyplot as plt 
import numpy as np
import csv
import os

from map import Map


## TODO change the index in load_file_names function to get all agent name
## at moment it does not get the full name
## TODO novelty_analysis needs to be improved 


class PlotResults():
    """
    class implementation for the convenience of reading the data 
    at each starting position

    distribute: 
    ------
        dir_path: the current working directory path (./eecs545_artificial_curiosity/)
        results_folder: the "results2" folder 
        results_path: (./eecs545_artificial_curiosity/results2)
    
    Methods:
    ------
        load_file_names: load the name of starting positions and agent names
        resd_csv: load csv data into list
        load_agent_data: load all stored agent csv file
        novelty_analysis: TODO analyze results

        ***The following methods are for a single starting point
        plot_path_variance: plot the path variance 
        plot_paths_new: plot path map
    """
    def __init__(self, dir_path: str, results_folder: str):
        """
        Initilize the Plot Results class only by directory path and results folder
        """
        # ./eecs545_artificial_curiosity/
        self.dir_path = dir_path
        # 'results' folder
        self.results_folder = results_folder
        #./eecs545_artificial_curiosity/results/
        self.results_path = os.path.join(self.dir_path, self.results_folder) 
    

    def load_file_names(self, results_folder: str):
        """ load_data_folder for further convenient use
            results data are organized as: 
                Each strating position: ./results/
                Each agent of a starting position: ./results/pos_XXX_XXX/
                Stored data of each agent: ./results2/pos_XXX_XXX/<agent name>/

        Parameters:
        ------
            results_folder: str --> indicate which results folder to use

        Return:
        ------
            pos_list: list --> all starting positions
            agent_names: list --> all agent names without the '_pos_XXX_XXX'
                        e.g. "Curiosity_Brain_ListMem32_ImgSize64_NovMAE_Train1_Lrate0.0002_Agent_Pos_1772_86"
        """
        # results_path = os.path.join(self.dir_path, self.results_folder) 
        pos_list = os.listdir(self.results_path) # a list of "pos_XXX_XXX" in the results folder

        # read agent names in any pos_XXX_XXX folder
        # agent names are identical except the last "_Pos_XXX_XXX"
        # return agent names without "_Pos_XXX_XXX"
        agent_names = os.listdir(os.path.join(self.results_path, pos_list[2]))
        agent_names_update = []
        for agent_name in agent_names:
            agent_name = agent_name.split("_")[:-3]
            agent_names_update.append("_".join(agent_name))
        
        return pos_list, agent_names_update


    def read_csv(self, filepath: str):
        """ Help to read csv file

        Parameters:
        ------
        filepath: file path of the csv
                  e.g. ./results2/pos_XXX_XXX/<agent name>/path_record.csv
        Return:
        ------
        list_content: a list of whatever data recorded (convert to float)
        """
        list_content = []
        # try:
        #     with open(filepath, 'r', newline='') as f:
        #         csv_reader = csv.reader(f, delimiter=',')
        #         for row in csv_reader:
        #             list_content.append(row)
        # except:
        #     print("Data has not been recorded")

        try:
            f = open(filepath, 'r', newline='')
        except:
            print("...Data has not been recorded")
        else:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                 list_content.append(row)

        return list(list_map(lambda x: list(list_map(float, x)), list_content))



    def load_agent_data(self, pos_list: List[str], agent_names: List[str]):
        """
        use load_agent_data function in engine module to load results csv

        Parameters:
        ------
        pos_list: List[str] --> specify a list of Pos_XXX_XXX folder to load data
        agent_names: List[str] --> specify a list of agent_names of load data
                     agent_names is returned by load_file_names function

        Return: six dictionaries ref_agents and cur_agents stored separately 
        ------
        {agent_name: values}
        Curiosity agents: cur_grain_novelty, cur_path_record, 
                        cur_perceived_path_novelty, cur_ave_path_novelty 
        Reference agents: ref_path_record, ref_ave_path_novelty
        """
        # store curiosity agents data
        cur_grain_novelty = {}
        cur_path_record = {}
        cur_perceived_path_novelty = {}
        cur_avg_path_variance = {}
        # store reference agents data
        ref_path_record = {}
        ref_avg_path_variance = {}

        for pos in pos_list: 
            for agent_name in agent_names:
                # rename agent with Pos_XXX_XXX
                # folder name is pos_XXX_XXX, agent name is Pos_XXX_XXX
                agent_name_update = agent_name + '_' + pos.title()
                # access the agent path
                agent_path = os.path.join(self.results_path, pos, agent_name_update)

                # for cur_agents: [avg_path_variance.csv, grain_novelty.csv, path_record.csv, perceived_path_novelty.csv, snapshots]
                if agent_name.startswith("Curiosity"):
                    cur_grain_novelty[agent_name_update] = self.read_csv(os.path.join(agent_path, 'grain_novelty.csv'))
                    cur_path_record[agent_name_update] = self.read_csv(os.path.join(agent_path, 'path_record.csv'))
                    cur_perceived_path_novelty[agent_name_update] = self.read_csv(os.path.join(agent_path, 'perceived_path_novelty.csv'))
                    # avg_path_variance is a float, pop out from a list[list[avg_path_variance]]
                    cur_avg_path_variance[agent_name_update] = self.read_csv(os.path.join(agent_path, 'avg_path_variance.csv'))[0][0]

                # for ref_agents: [avg_path_variance.csv, path_record.csv]
                else:
                    ref_path_record[agent_name_update] = self.read_csv(os.path.join(agent_path,  'path_record.csv'))
                    # avg_path_variance is a float, pop out from a list[list[avg_path_variance]]
                    ref_avg_path_variance[agent_name_update] = self.read_csv(os.path.join(agent_path, 'avg_path_variance.csv'))[0][0]

        return cur_grain_novelty, cur_path_record, cur_perceived_path_novelty, cur_avg_path_variance, ref_path_record, ref_avg_path_variance
    

    ## TODO novelty_analysis needs to be improved  
    def novelty_analysis(self, cur_avg_path_variance: dict, ref_avg_path_variance: dict):
        """
        analyze the novalty of each agent
        Needs more improvements
        """
        # sort novelty of curiosity agents 
        cur_avg_path_variance_sort = dict(sorted(cur_avg_path_variance.items(), key=lambda item: item[1], reverse=True))

        return cur_avg_path_variance_sort


    def plot_path_variance(self, cur_avg_path_variance: dict, ref_avg_path_variance: dict, pos: str, show: bool=False, save: bool=False):
        """
        histogram of novelty of each agent at the same starting position

        Parameters:
        ------
            cur_avg_path_variance: the dictionary of avg curiosity agents path variance
            ref_avg_path_variance: the dictionary of ref curiosity agents path variance
            pos: str --> the name of starting position folder
            show, save: bool
        """
        # make sure only plot for the same starting position
        # only two reference variances are recored per starting position 
        assert len(ref_avg_path_variance) == 2

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
            ax.text(lin_path_variance-0.002, ymax*0.6, "Linear agent", color ='r', rotation=90)
            ax.text(rand_path_variance-0.002, ymax*0.6, "Random agent", color ='k', rotation=90)

            if (save):
                # like results folder, create a plot folder for each starting pos
                dirname = 'plots' + '/' + pos
                plotname = pos + '_novelty_hist' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(self.dir_path, dirname, plotname))
                
            plt.show() 


    def plot_paths_new(self, cur_path_record: dict, ref_path_record: dict, pos: str, show: bool=False, save: bool=False):
        """
        plot selected agent path after loading the agent path data

        Parameters: 
        ------
        cur_path_record: dict --> Curiosity agents path
        ref_path_record: dict --> reference agents path
        pos: str --> the name of starting position folder
        show/save: bool
        """
        # make sure only plot for the same starting position
        # only two reference variances are recored per starting position 
        assert len(ref_avg_path_variance) == 2

        # TODO: Only select top 
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
                dirname = 'plots' + '/' + pos
                plotname = pos + '_path_map' + '.svg'
                os.makedirs(dirname, exist_ok=True)
                plt.savefig(os.path.join(self.dir_path, dirname, plotname))
                pass

            plt.show()



if __name__ == '__main__':
    fov = 64
    image_path = os.path.join(os.path.dirname(__file__), 'data', 'mars.png') 
    map = Map(image_path, fov, 2)

    ## initialize PlotResults class
    dir_path = os.path.dirname(__file__) # ../eecs545_artificial_curiosity/
    results_folder = 'results2' 
    plot_result = PlotResults(dir_path, results_folder)
    ## read the starting position names and agent_names
    pos_list, agent_names = plot_result.load_file_names(results_folder)


    ###### Example of loading interesting starting positions and agents
    ###### Processing can be based on multiple starting positions
    pos_list_test = ['pos_1772_86', 'pos_2384_959']
    agent_names_test = agent_names[:2]
    # load all six csv
    cur_grain_novelty, cur_path_record, cur_perceived_path_novelty, cur_avg_path_variance, \
    ref_path_record, ref_avg_path_variance = plot_result.load_agent_data(pos_list_test, agent_names_test)


    ####### Examples of ploting avg_path_variance
    ####### Processing based on per starting position
    # plot avg_variance per starting position
    starting_positions = ['pos_1772_86', 'pos_2384_959']
    # starting_positions = pos_list
    for starting_position in starting_positions:
        print(f"processing the starting positions {starting_position}...")
        # the load_agent_data function needs list input
        starting_position = [starting_position]

        # load all six csv
        cur_grain_novelty, cur_path_record, cur_perceived_path_novelty, cur_avg_path_variance, \
        ref_path_record, ref_avg_path_variance = plot_result.load_agent_data(starting_position, agent_names)

        # plot variance 
        plot_result.plot_path_variance(cur_avg_path_variance, ref_avg_path_variance, starting_position[0], show=True, save=True)
        plot_result.plot_paths_new(cur_path_record, ref_path_record, starting_position[0], show=True, save=True)