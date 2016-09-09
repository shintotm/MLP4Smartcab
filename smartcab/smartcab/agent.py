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
        self.success = 0
        self.red_light_violations = []
        self.planner_noncompliance = []
        
        self.qLearnTable = {}
        self.prevActionIndex = None
        self.prevReward = 0
        self.prevState = None
        
        self.alpha = 0.5
        self.gamma = 0.1
        self.epsilon = 1
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
       
        self.red_light_violations.append(0)
        self.planner_noncompliance.append(0)
        
        self.prevActionIndex = None
        self.prevReward = 0
        self.prevState = None


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
             self.success += 1

    def selectBestAction(self):
        self.epsilon = self.epsilon * 0.999
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

    max_result = 0
    best_alpha = 0
    best_gamma = 0

    gammas = []
    gammas.append(1)
    for g in range(100):
        gammas.append(gammas[-1] * 0.96)

        alphas = []
    alphas.append(1)
    for a in range(100):
        alphas.append(alphas[-1] * 0.96)
    
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'gridsearch_'+timestr +'.log'    
    for alpha in alphas:

        for gamma in gammas:

            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            a.alpha = alpha
            a.gamma = gamma
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0.000005, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            print "Success: ", a.success, alpha, gamma
            with open(filename, 'a') as logfile:
                logfile.write("%d %.4f %.4f\n" % (a.success, alpha, gamma) )
            if a.success > max_result:
                max_result = a.success
                best_alpha = alpha
                best_gamma = gamma
                print max_result, alpha, gamma

    print 'best success rate:',max_result, ' alpha: ', best_alpha, ' gamma: ', best_gamma
    #import matplotlib.pyplot as plt
    #plt.plot(a.red_light_violations, label='red light violations', color='r')
    #plt.plot(a.planner_noncompliance, label = 'planner noncompliance', color='b')
    #plt.legend()
    #plt.xlabel('Trials')
    #plt.title('Q-Learning, with deadline. Success rate %d%%' % a.success)
    #plt.show()
    #plt.savefig('QLearningWithDeadline.png')



if __name__ == '__main__':
    run()
