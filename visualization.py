"""
This file help us for visualization. 
"""
import numpy as np 
import matplotlib.pyplot as plt 
import simulaiton_params as s

class Visual(object): 
    def __init__(self, sim_params): 
        self.number_of_games = sim_params["number_of_games"]
        self.sumo_ctrl = sim_params['sumo_ctrl']
    
    
    def general_plot (self, accidents, out_of_roads, same_lane_emg, emg_reach_end, steps, speed_violation): 
        x = []
        for i in range (self.number_of_games): 
            x.append(i)

        z = accidents
        t = out_of_roads
        q = same_lane_emg 
        p = emg_reach_end
        r = steps
        k = speed_violation 

        plt.subplot(2, 3, 1)
        plt.xlabel("Accidents")
        plt.ylim(0,  2)
        plt.bar(x, z)
        
        plt.subplot(2, 3, 4)
        plt.xlabel("Out of road")
        plt.ylim(0,  2)
        plt.bar(x, t)

        plt.subplot(2, 3, 2)
        plt.bar(x, q)
        plt.xlabel("Same lane with emg")

        plt.subplot(2, 3, 5)
        plt.bar(x, p)
        plt.xlabel("emg reaches to the end")

        plt.subplot(2, 3, 3)
        plt.bar(x, r)
        plt.xlabel("Steps/episode")        

        plt.subplot(2, 3, 6)
        plt.bar(x, k)
        plt.xlabel("speed_violation")
        plt.tight_layout()
        plt.savefig(f"Additional_result_with_SUMO_{self.sumo_ctrl}.jpg", dpi=100)

        plt.show()

    def reward_plot (self, reward): 
        x = []
        for i in range (self.number_of_games): 
            x.append(i)
        y = reward 
        t = [0, 200, 400, 500, 700, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        labele = ["0","200","400","500","700","1000","1300","1600","2000", "2500", '3000', '3500', '4000', '4500', '5000']
        plt.xticks(t, labele)
        plt.xlabel("Episode")
        plt.ylabel("Penalty")
        plt.grid(True)
        plt.scatter(x, y, s=10, c='darkred')
        plt.tight_layout()
        plt.savefig(f"Reward_result_with_SUMO_{self.sumo_ctrl}.jpg", dpi=100)
        plt.show()

    def reward_avg_plot(self, avg_reward):
        x = np.arange(0, self.number_of_games, step=5)
        y = avg_reward
        t = [0, 200, 500, 700, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        labele = ["0","200","500","700","1000","1300","1600","2000", "2500", '3000', '3500', '4000','4500', '5000']
        plt.xticks(t, labele)
        plt.xlabel("Episodes")
        plt.ylabel("Penalty")
        plt.grid(True)
        plt.scatter(x, y, s=10, c='darkred' )
        plt.tight_layout()
        plt.savefig(f"AVG_Reward_result_with_SUMO_{self.sumo_ctrl}.jpg", dpi=100)
        plt.show()
    


    def reward_plot_plot (self, reward): 
        x = []
        for i in range (self.number_of_games): 
            x.append(i)
        y = reward 
        t = [0, 200, 500, 700, 1000, 1300, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        labele = ["0","200","500","700","1000","1300","1600","2000", "2500", '3000', '3500', '4000', '4500', '5000']
        plt.xticks(t, labele)
        plt.xlabel("Episode")
        plt.ylabel("Penalty")
        plt.grid(True)
        plt.plot(x, y, linewidth=0.5, color='darkred')
        plt.tight_layout()
        plt.savefig(f"Reward_result_with_SUMO_{self.sumo_ctrl}.jpg", dpi=100)
        plt.show()

    def totext(self, rewards, avg_reward, accidents, out_of_roads, same_lane_emg, emg_reach_end, steps, speed_violation): 
        with open ("reward.txt", 'w') as file: 
            file.write(str(rewards))
        with open("avg reward.txt", 'w') as file: 
            file.write(str(avg_reward))
        with open("accidents.txt", 'w') as file: 
            file.write(str(accidents))
        with open("outroads.txt", 'w') as file: 
            file.write(str(out_of_roads))
        with open ("same_lane_emg.txt", 'w') as file: 
            file.write(str(same_lane_emg))
        with open("emg_reach_end.txt", 'w') as file: 
            file.write(str(emg_reach_end))
        with open("steps.txt", 'w') as file: 
            file.write(str(steps))
        with open("speedviolation.txt", 'w') as file: 
            file.write(str(speed_violation))

    
    