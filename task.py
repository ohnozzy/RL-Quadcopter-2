import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose, init_velocities=None, 
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
        self.init_pose = init_pose
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        single_state_size = 9
        direction_vector_size = 3
        

        self.state_size = direction_vector_size+self.action_repeat * single_state_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.score_weight = np.array([1.0])
        self.score_L1_norm = 10*abs(self.score_weight).sum()
        self.shift_angle_range = np.vectorize(lambda x : x if x < np.pi else x - 2*np.pi)
        

        
    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        #if(self.sim.pose[2]<-1.0):
        #    return -10
        #manhattan = abs(self.sim.pose[:3] - self.target_pos).sum()
        #if (manhattan<0.1):
        #    return 10
        #position_score = np.tanh(self.sim.pose[2] - self.init_pose[2]) 
        #upward_score = np.tanh(0.1*self.sim.v[2])
        #upward_accel_score = np.tanh(0.01*self.sim.linear_accel[2])
        
        #sideway_penalty = np.exp(2*upward_score)*np.tanh(0.25*np.log(np.linalg.norm(self.sim.pose[:2] - self.init_pose[:2])+1))
        
        #angle_dif = self.shift_angle_range(abs(self.sim.pose[3:] - self.init_pose[3:]))
        
        
        #angular_penalty =np.tanh(0.1*np.linalg.norm(angle_dif))
        
        #move_aside_penalty = np.tanh(0.2*np.log(np.linalg.norm(self.sim.v[:2])+1))
        #rotate_aside_penalty = np.tanh(0.05*np.linalg.norm(self.sim.angular_v))
        
        #rotor_tor = np.tanh(0.001*np.linalg.norm(rotor_speeds))
        #uneven_rotor_speeds = np.tanh(0.5*np.std(rotor_speeds)/(max(np.mean(rotor_speeds),1)))
        #max_force = np.tanh(0.01*np.max(rotor_speeds))
        #reward = upward_reward -move_aside_penalty - sideway_penalty - uneven_rotor_speeds
        #return reward
        #score_v= np.array([upward_score, sideway_penalty, move_aside_penalty, angular_penalty, rotate_aside_penalty, rotor_tor])
        #reward =   score_v.dot(self.score_weight)/self.score_L1_norm
        #return reward/self.action_repeat, score_v
        #return 0
        pos_dif = self.target_pos[:3] - self.sim.pose[:3]
        pos_dif_norm = np.linalg.norm(pos_dif)
        score_v = np.array([pos_dif_norm])
        """velocity_norm = np.linalg.norm(self.sim.v);
        
        position_score = 2*(.5-np.tanh(0.1*pos_dif_norm))
        if pos_dif_norm > 0.1:
            velocity_score = 0.5*np.tanh(0.1*self.sim.v.dot(pos_dif)/pos_dif_norm)
            accel_score = 0.3*np.tanh(0.1*self.sim.linear_accel.dot(pos_dif)/pos_dif_norm)
        else:
            velocity_score = 1.0 - 0.5*np.tanh(0.1*velocity_norm)
            if velocity_norm >0.1:
                accel_score = 0.6-0.3*np.tanh(0.1*self.sim.linear_accel.dot(self.sim.v)/velocity_norm)
            else:
                accel_score = 1.0 - 0.1* np.tanh(0.1*np.linalg.norm(self.sim.linear_accel))
       
        score_v = np.array([position_score, velocity_score, accel_score])
        reward = score_v.dot(self.score_weight)/self.score_L1_norm
        return reward/self.action_repeat, score_v"""
        #angular = self.shift_angle_range(self.sim.pose[3:])
        if pos_dif_norm < 1.0:
            reward =  1.0 #-0.03*abs(angular[1])
        else:
            reward = 1.0/pos_dif_norm #-0.03*abs(angular[1])
        
        return 0.1*reward , score_v
    
    
    def stepNoReward(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        additional_component = np.zeros(len(self.score_weight))
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            pose_all.append(pose_all.append(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v,self.sim.linear_accel))))
            
            
            #if(self.sim.pose[2]<-1.0):
            #    done = True
        next_state = np.concatenate(pose_all)
        return next_state, done

    def actions2rotor_speed(self ,actions):
        speed = 425 + 25* actions
        return speed

    def step(self, actions):
        """Uses action to obtain next state, reward, done."""
        #reward = 0
        additional_component = np.zeros(len(self.score_weight))
        pose_all = []
        
        rotor_speeds = self.actions2rotor_speed(actions)
        for _ in range(5):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            #if done:
                #print('sim.next_timestep return done')
            if done:
                break
            
            #reward += r
            #additional_component += v
            #pose_all.append(self.sim.v[2])
            #pose_all.append(np.concatenate((self.sim.pose[2],self.sim.v[2])))
            #angular = self.shift_angle_range(self.sim.pose[3:])
            #if max(abs(self.sim.v[:3]))>100.0:
                #print('Early terminate(speed): %s %s' %(self.sim.pose,self.sim.v))
            #    done=True
            
            #if abs(angular[1])>np.pi/2.0 and reward<0.0:
                #print('Early terminate(orientation): %s %s' %(self.sim.pose,angular))
            #    done=True
            
            
            #if(self.sim.pose[2]<-1.0):
            #    done = True
        reward, v = self.get_reward(rotor_speeds)
        #z_hat = self.sim.v[2]
        #if reward>20:
        #    done = True
        #if done:
                #print('crash pose: ',self.sim.pose[2])
        #        if self.sim.time<self.sim.runtime:
        #            reward = -300
        direction = self.sim.pose[:3] - self.target_pos
        #next_state = np.concatenate((next_state,direction))
        next_state = np.concatenate((self.sim.pose[3:], self.sim.angular_v, self.sim.v, direction))
        #pos_dif = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        #if pos_dif>10:
        #    done = True
        #    reward = -1.5
        #        #print('reach pose: ',self.sim.pose[2])
        #        done=True
                #reward = 100
        return next_state, reward, done, v

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        direction = self.sim.pose[:3] - self.target_pos
        #single_state = np.concatenate((self.sim.pose[2], self.sim.v[2]))
        state = np.concatenate((self.sim.pose[3:], self.sim.angular_v, self.sim.v, direction))
        return state