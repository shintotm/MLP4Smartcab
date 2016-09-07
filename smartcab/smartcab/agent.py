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
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
       
        self.red_light_violations.append(0)
        self.planner_noncompliance.append(0)


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        # TODO: Select action according to your policy
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward == -1.0:
            self.red_light_violations[len(self.red_light_violations) - 1] += 1
        elif reward == -0.5:
            self.planner_noncompliance[len(self.planner_noncompliance) - 1] += 1

        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
 
        if self.planner.next_waypoint() == None :
             self.success += 1


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0005, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "Success: ", a.success

    import matplotlib.pyplot as plt
    plt.plot(a.red_light_violations, label='red light violations', color='r')
    plt.plot(a.planner_noncompliance, label = 'planner noncompliance', color='b')
    plt.legend()
    plt.xlabel('Trials')
    
    plt.title('Random actions, no deadline. Success rate %d%%' % a.success)
    #plt.show()
    plt.savefig('radomAction.png')


if __name__ == '__main__':
    run()
