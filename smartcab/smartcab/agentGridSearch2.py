import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.success_rate = 0
        self.red_light_violations = []
        self.planner_noncompliance = []
        
        self.qLearnTable = {}
        self.prevActionIndex = None
        self.prevReward = 0
        self.prevState = None
        
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 1
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
       
        self.red_light_violations.append(0)
        self.planner_noncompliance.append(0)
        
        self.prevActionIndex = None
        self.prevReward = 0
        self.prevState = None
        self.epsilon *= 0.9


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
                    
        # TODO: Select action according to your policy

        action = self.selectBestAction()
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward == -1.0:
            self.red_light_violations[len(self.red_light_violations) - 1] += 1
        elif reward == -0.5:
            self.planner_noncompliance[len(self.planner_noncompliance) - 1] += 1
        

        # TODO: Learn policy based on state, action, reward
       
        if self.prevState != None:
            qMax = max(self.qLearnTable[self.state])
            prevQ = self.qLearnTable[self.prevState][self.prevActionIndex] 
            newQ = prevQ + self.alpha *(self.prevReward + self.gamma * (qMax - prevQ))
            self.qLearnTable[self.prevState][self.prevActionIndex] = newQ
            #print 'prevQ:', prevQ, 'newQ: ', newQ
            
        self.prevState = self.state
        self.prevActionIndex = self.env.valid_actions.index(action)
        self.prevReward = reward
        

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
 
        if self.planner.next_waypoint() == None :
             self.success_rate += 1

    def selectBestAction(self):
        #self.epsilon = self.epsilon * 0.999
        #print self.epsilon
        if self.state not in self.qLearnTable.keys():
            # new state, initialize to zero
            self.qLearnTable[self.state] = [0,0,0,0]
            #choose a random action
            return random.choice(self.env.valid_actions)
            
        if random.random() < self.epsilon:
            # explore by choosing a random action
            return random.choice(self.env.valid_actions)
        
        # find the best action for current state
        maxQ = max(self.qLearnTable[self.state])
        index = self.qLearnTable[self.state].index(maxQ)
        return self.env.valid_actions[index]

def run():
    """Run the agent for a finite number of trials."""
    import time

    alpha_range = [x / 100.0 for x in range(1,21)]
    gamma_range = [x / 100.0 for x in range(1,21)]
    
    max_result = 0
    best_alpha = 0
    best_gamma = 0

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'gridsearch_'+timestr +'.log'    
    with open(filename, 'a') as logfile:
        logfile.write("Alpha,Gamma,SuccessRate,Last20RedLightViolations,Last20PlannerNoncompliance\n" )
    
    for alpha in alpha_range:
        for gamma in gamma_range:
            success_rates = []
            last20_redlight_violations = []
            last20_planner_noncompliance = []
            for count in range(10):
                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent)  # create agent
                a.gamma = gamma
                a.alpha = alpha
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0.0000005, display=False)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                sim.run(n_trials=100)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                #plot_agent_performance(a.alpha, a.gamma,a.success_rate, a.red_light_violations, a.planner_noncompliance, count)
                
                sum_last20_redlight_violations = sum(a.red_light_violations[-20:])
                sum_last20_planner_noncompliance = sum(a.planner_noncompliance[-20:])
                
                success_rates.append(a.success_rate)
                last20_redlight_violations.append(sum_last20_redlight_violations)
                last20_planner_noncompliance.append(sum_last20_planner_noncompliance)
                
            mean_success = sum(success_rates)/float(len(success_rates))
            mean_last20redlight = sum(last20_redlight_violations)/float(len(last20_redlight_violations))
            mean_last20planner = sum(last20_planner_noncompliance)/float(len(last20_planner_noncompliance)) 
            print 'Mean success rate: ', mean_success
            print 'Mean last 20 red light violationse: ', mean_last20redlight
            print 'Mean last 20 planner_noncompliance: ', mean_last20planner
            with open(filename, 'a') as logfile:
                logfile.write("%.2f,%.2f,%d,%.2f,%.2f\n" % (alpha, gamma, mean_success, mean_last20redlight, mean_last20planner) )
     
def plot_agent_performance(alpha, gamma, success_rate, red_light_violations, planner_noncompliance, count):
    import matplotlib.pyplot as plt
    import time
    
    sum_last20_redlight_violations = sum(red_light_violations[-20:])
    sum_last20_planner_noncompliance = sum(planner_noncompliance[-20:])
    
    plt.plot(red_light_violations, label='red light violations', color='r')
    plt.plot(planner_noncompliance, label = 'planner noncompliance', color='b')
    plt.legend()
    plt.xlabel('Trials')
    plt.title('Enhanced Q-Learning, with deadline. Success rate %d%%' % success_rate)
    y_pos = max(red_light_violations + planner_noncompliance)*0.8
    plt.text(50, y_pos, r'$\alpha: %.3f, \gamma: %.3f$' % (alpha, gamma))
    
    y_pos *= 0.95
    plt.text(50, y_pos, "Last 20 red light violations: %d" % sum_last20_redlight_violations)
    y_pos *= 0.95
    plt.text(50, y_pos, "Last 20 planner noncompliance: %d" % sum_last20_planner_noncompliance)
    #plt.show() 
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('EnhancedQLearning_'+timestr +'_count%d_alpha%.2f_gamma%.2f_successRate%d.png' % (count,alpha, gamma, success_rate))
    plt.clf()

                

if __name__ == '__main__':
    run()
