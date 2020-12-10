import os
import pandas as pd
import numpy as np
import re

class AgentInfo():
    options = [
        "motivation",
        "mem_len",
        "mem_type",
        "img_size",
        "metric",
        "choochoo",
        "rate",
    ]

    def __init__(self, agent_str):
        info = agent_str.split("_")
        self.agent = info[0]

        if(self.agent.lower() == "curiosity"):
            self.motivation = info[1]
            self.memory = info[2]
            self.img_size = info[3]
            self.metric = info[4]
            self.choochoo = info[5]
            self.rate = info[6]
        else:
            self.motivation = None
            self.memory = None
            self.img_size = None
            self.metric = None
            self.choochoo = None
            self.rate = None

        self.position = "_".join(info[-4:])

    def __str__(self):
        if(self.motivation is not None):
            return "_".join([
                self.agent,
                self.motivation,
                self.memory,
                self.img_size,
                self.metric,
                self.choochoo,
                self.rate,
                self.position
            ])
        else:
            return "_".join([
                self.agent,
                self.position
            ])

    def __repr__(self):
        return str(self)

    def get_dir(self):
        return self.position.lower().replace("agent_", "")

    def load_data(self, source_dir="./"):
        self.data = pd.read_csv(os.path.join(source_dir, self.get_dir(), str(
            self), "avg_path_variance.csv"), header=None)[0][0]

    def __lt__(self, other):
        return self.data < other.data

    def __gt__(self, other):
        return self.data > other.data

    def __le__(self, other):
        return self.data <= other.data

    def __ge__(self, other):
        return self.data >= other.data

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return self.data != other.data

    @staticmethod
    def generate_agents(source_dir="./results2"):
        agents = []
        position_dirs = [x for x in os.listdir(source_dir) if "pos_" in x]

        for dir in position_dirs:
            agent_dirs = [x for x in os.listdir(
                os.path.join(source_dir, dir)) if "Pos_" in x]
            for agent in agent_dirs:
                temp_agent = AgentInfo(agent)
                temp_agent.load_data(source_dir)
                agents.append(temp_agent)

        return agents

    @staticmethod
    def sort_type(t, dir_input):
        result = {}

        if(t.lower() == "motivation"):
            for a in agents:
                if(a.motivation in result):
                    result[a.motivation].append(a)
                else:
                    result[a.motivation] = [a]
        elif(t.lower() == "agent"):
            for a in agents:
                if(a.agent in result):
                    result[a.agent].append(a)
                else:
                    result[a.agent] = [a]
        elif(t.lower() == "mem_len"):
            for a in agents:
                mem = re.sub(r'([a-zA-Z]+)(\d+)', r'Mem\2', str(a.memory))
                if(mem in result):
                    result[mem].append(a)
                else:
                    result[mem] = [a]
        elif(t.lower() == "mem_type"):
            for a in agents:
                mem = re.sub(r'([a-zA-Z]+)(\d+)', r'\1', str(a.memory))
                if(mem in result):
                    result[mem].append(a)
                else:
                    result[mem] = [a]
        elif(t.lower() == "img_size"):
            for a in agents:
                if(a.img_size in result):
                    result[a.img_size].append(a)
                else:
                    result[a.img_size] = [a]
        elif(t.lower() == "metric"):
            for a in agents:
                if(a.metric in result):
                    result[a.metric].append(a)
                else:
                    result[a.metric] = [a]
        elif(t.lower() == "choochoo" or t.lower() == "train"):
            for a in agents:
                if(a.choochoo in result):
                    result[a.choochoo].append(a)
                else:
                    result[a.choochoo] = [a]
        elif(t.lower() == "rate"):
            for a in agents:
                if(a.rate in result):
                    result[a.rate].append(a)
                else:
                    result[a.rate] = [a]

        result.pop(None, None)
        for k in result:
            result[k].sort(reverse=True)

        return result

    @staticmethod
    def average(lst_agents):
        return np.sum([x.data for x in lst_agents])/len(agents)

    @staticmethod
    def find_best_overall(dict_agents):
        bests = []
        for i in AgentInfo.options:
            averages = []
            temp = AgentInfo.sort_type(i, dict_agents)

            for k in temp:
                averages.append((k, AgentInfo.average(temp[k])))
            averages.sort(key=lambda x: x[1], reverse=True)

            bests.append(averages[0])

        return bests


if __name__ == "__main__":
    agents = AgentInfo.generate_agents()
    temp = AgentInfo.sort_type("motivation", agents)
    print(AgentInfo.find_best_overall(temp))
