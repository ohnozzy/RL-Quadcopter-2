import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        single_state_size = 12
        

        self.state_size = self.action_repeat * single_state_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #if(self.sim.pose[2]<-1.0):
        #    return -10
        #manhattan = abs(self.sim.pose[:3] - self.target_pos).sum()
        #if (manhattan<0.1):
        #    return 10
        reward = .2*self.sim.pose[2]-.1*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()+0.1*self.sim.v[2]
        return reward
        #return 0

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if done:
                print('sim.next_timestep return done')
            reward += self.get_reward() 
            pose_all.append(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v)))
            
            
            #if(self.sim.pose[2]<-1.0):
            #    done = True
        next_state = np.concatenate(pose_all)
        #if reward>20:
        #    done = True
        if self.sim.pose[2]<1:
                #print('crash pose: ',self.sim.pose[2])
                done=True
                reward = -300
        if self.sim.pose[2]>self.target_pos[2]:
                print('reach pose: ',self.sim.pose[2])
                done=True
                reward = 1000
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v))] * self.action_repeat) 
        return state