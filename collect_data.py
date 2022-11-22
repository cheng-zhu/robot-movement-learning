from SteeringBehaviors import Wander
import SimulationEnvironment as sim
import pandas as pd
import numpy as np

def collect_training_data(total_actions):
    #set-up environment
    sim_env = sim.SimulationEnvironment()

    #robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    # network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision
    network_params = []


    for action_i in range(total_actions):
        progress = 100*float(action_i)/total_actions
        print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        #steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                #STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                #if current action is very new.
                if action_timestep < action_repeat * .3: #in case prior action caused collision
                    network_params[-1][-1] = collision #share collision result with prior action
                break


        # Update network_params.
        param = []
        for i in range(len(sensor_readings)):
            param.append(sensor_readings[i])
        param.append(action)
        param.append(0)

        network_params.append(param)


    arr = np.array(network_params)
    DF = pd.DataFrame(arr)
    DF.to_csv("training_data.csv", header=None, index=None)






if __name__ == '__main__':
    total_actions = 20000
    collect_training_data(total_actions)
