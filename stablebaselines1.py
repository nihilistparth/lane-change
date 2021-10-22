# from _typeshed import Self
import glob
import os
import sys
import math
import time
from gym.spaces.discrete import Discrete
import numpy as np
import cv2
import random
from collections import deque
from random import randrange
import gym
from numpy.linalg.linalg import norm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C 
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter

# from tensorflow.contrib import seq2seq
# from tensorflow.contrib.rnn import DropoutWrapper
from gym import spaces

# from platform import python_version

# print(python_version())
# Global Setting
ROUNDING_FACTOR = 3
SECONDS_PER_EPISODE = 80
LIMIT_RADAR = 500
STATE_SPACE = 5
ACTION_SPACE =1
np.random.seed(32)
random.seed(32)
MAX_LEN = 600

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
'''
[114.8,128.9] 
[109.2,129.1]
[106.1,129.2]
[102.9,129.3]
'''
check_pts = [
            [96.00,129.3],
            [99.00,129.3],
            [102.9,129.3],
            [106.1,129.2],
            [110.2,129.1],
            [114.8,128.9], 
            ]
class CarlaVehicle(gym.Env):
    """
    class responsable of:
        -spawning the ego vehicle
        -destroy the created objects
        -providing environment for RL training
    """

    #Class Variables
    def __init__(self):
            super(CarlaVehicle,self).__init__() 
            self.client = carla.Client('localhost',2000)
            self.client.set_timeout(5.0)
            self.radar_data = deque(maxlen=MAX_LEN)
            self.state_space = np.array([STATE_SPACE])
            self.action_space = np.array([ACTION_SPACE])
            self.radar_space = np.array([LIMIT_RADAR])
            self.actor_list = []
            # self.action_space_arr = 
            self.action_space = spaces.Box(low=-1, high=1,shape=(1,), dtype=np.float32)  # steer else everythin in hardcoded
            # self.observation_space = spaces.Box(low=-1000, high=1000,shape=(LIMIT_RADAR,), dtype=np.float32)
            # self.observation_space = spaces.Tuple(spaces.Box(-1000, 1000, shape=(LIMIT_RADAR, ), dtype=np.float32),spaces.Box(-1000, 1000, shape=(5, ), dtype=np.float32))
            self.observation_space_dict = {
                'radar_data': spaces.Box(-20000, 20000, shape=(LIMIT_RADAR, ), dtype=np.float32),
                'speed'     : spaces.Box(low=0, high=100,shape=(1,), dtype=np.float32),
                'yaw'       : spaces.Box(low=-360, high=360,shape=(1,), dtype=np.float32),
                'lane'      : spaces.Discrete(2),
                'diff_angle': spaces.Box(low=-360, high=360,shape=(1,), dtype=np.float32),
                'norm'      : spaces.Box(low=0, high=100,shape=(1,), dtype=np.float32),  #[kmh, self.yaw_vehicle, lane, diff_angle, norm]                 
            }
            self.observation_space = spaces.Dict(self.observation_space_dict)
        

    def reset(self,Norender=False,loc = None):
        '''reset function to reset the environment before 
        the begining of each episode
        :params Norender: to be set true during training
        '''
        self.destroy()
        self.collision_hist = []
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        '''target_location: Orientation details of the target loacation
        (To be obtained from route planner)'''
        # self.target_waypoint = carla.Transform(carla.Location(x = 1.89, y = 117.06, z=0), carla.Rotation(yaw=269.63))
        self.given_rew = [0]* len(check_pts) # tocheck if reward given for that checkpoint or not
        self.chkpt_reward = 50               # this reward will be increased as ego approaches endpoint
        #Code for setting no rendering mode
        if Norender:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]
        self.static_prop = self.blueprint_library.filter("static.prop.streetbarrier")[0]
        self.tim =0

        #==================For learning Lane Change Maneuver===============
        # prop_1_loc = carla.Transform(carla.Location(x=69.970375, y=133.594208, z=0.014589), carla.Rotation(pitch=0.145088, yaw=-0.817204+90, roll=0.000000))
        # prop_2_loc = carla.Transform(carla.Location(x=67.970741, y=133.622726, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204+90, roll=0.000000))
        # prop_3_loc = carla.Transform(carla.Location(x=83, y=130.111465, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204+90, roll=0.000000))
        # prop_1 = self.world.spawn_actor(self.static_prop, prop_1_loc)
        # prop_2 = self.world.spawn_actor(self.static_prop, prop_2_loc)
        # prop_3 = self.world.spawn_actor(self.static_prop, prop_3_loc)
        # self.actor_list.append(prop_1)
        # self.actor_list.append(prop_2)
        # self.actor_list.append(prop_3)

        # These will be in left side of the road
        prop_lt_1_loc = carla.Transform(carla.Location(x=96, y=126, z=0.014589), carla.Rotation(pitch=0.145088, yaw=-0.817204, roll=0.000000))
        prop_lt_2_loc = carla.Transform(carla.Location(x=100, y=126, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_3_loc = carla.Transform(carla.Location(x=104, y=126, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_4_loc = carla.Transform(carla.Location(x=108, y=126, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_5_loc = carla.Transform(carla.Location(x=112, y=126, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_lt_1 = self.world.spawn_actor(self.static_prop, prop_lt_1_loc)
        prop_lt_2 = self.world.spawn_actor(self.static_prop, prop_lt_2_loc)
        prop_lt_3 = self.world.spawn_actor(self.static_prop, prop_lt_3_loc)
        prop_lt_4 = self.world.spawn_actor(self.static_prop, prop_lt_4_loc)
        prop_lt_5 = self.world.spawn_actor(self.static_prop, prop_lt_5_loc)

        # These will be the end points on the road
        prop_end_rt_loc = carla.Transform(carla.Location(x=104, y=133, z=0.014589), carla.Rotation(pitch=0.145088, yaw=-0.817204+90, roll=0.000000))
        prop_end_lt_loc = carla.Transform(carla.Location(x=114, y=129.5, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204+90, roll=0.000000))
        prop_start_lt_loc = carla.Transform(carla.Location(x=94, y=129.5, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204+90, roll=0.000000))
        prop_end_rt = self.world.spawn_actor(self.static_prop, prop_end_rt_loc)
        prop_end_lt = self.world.spawn_actor(self.static_prop, prop_end_lt_loc)
        prop_start_lt = self.world.spawn_actor(self.static_prop, prop_start_lt_loc)

        # It will bw on right side of the road
        prop_rt_1_loc = carla.Transform(carla.Location(x=98, y=136.5, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_2_loc = carla.Transform(carla.Location(x=102, y=136.5, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_1 = self.world.spawn_actor(self.static_prop, prop_rt_1_loc)
        prop_rt_2 = self.world.spawn_actor(self.static_prop, prop_rt_2_loc)

        prop_rt_3_loc = carla.Transform(carla.Location(x=108, y=133, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_4_loc = carla.Transform(carla.Location(x=112, y=133, z=0.009964), carla.Rotation(pitch=0.119906, yaw=-0.817204, roll=0.000000))
        prop_rt_3 = self.world.spawn_actor(self.static_prop, prop_rt_3_loc)
        prop_rt_4 = self.world.spawn_actor(self.static_prop, prop_rt_4_loc)

        self.actor_list.append(prop_lt_1)
        self.actor_list.append(prop_lt_2)
        self.actor_list.append(prop_lt_3)
        self.actor_list.append(prop_lt_4)
        self.actor_list.append(prop_lt_5)

        self.actor_list.append(prop_end_lt)
        self.actor_list.append(prop_end_rt)
        self.actor_list.append(prop_start_lt)

        self.actor_list.append(prop_rt_1)
        self.actor_list.append(prop_rt_2)
        self.actor_list.append(prop_rt_3)
        self.actor_list.append(prop_rt_4)

        
        #create ego vehicle the reason for adding offset to z is to avoid collision
        # off = random.uniform(0, 3)
        # init_loc = carla.Location(x = -145 + off, y = 7.5, z=2)
        # self.init_pos = carla.Transform(init_loc, carla.Rotation(yaw=-90))
        # left lane change
        # init_loc = carla.Location(x = 50.52, y = 135.91, z=1)
        init_loc = carla.Location(x = 92, y = 134, z=1)
        self.init_pos = carla.Transform(init_loc, carla.Rotation(pitch=0.094270, yaw=0, roll=-1.246277))
        self.next_lane_target = self.map.get_waypoint(init_loc).get_left_lane()
        self.target_waypoint = carla.Transform(carla.Location(x = 110, y = 129.5, z=0), carla.Rotation(yaw=0))
        print(f'next_lane_target-----------{self.next_lane_target}')

        self.vehicle = self.world.spawn_actor(self.bp, self.init_pos)
        time.sleep(1)
        self.actor_list.append(self.vehicle)
        self.lane_id_ego = 0
        self.lane_id_target = 0
        self.yaw_vehicle = 0
        self.yaw_target_road =  0

        #Create location to spawn sensors
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        #Create Collision Sensors
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.episode_start = time.time()

        #Create location to spawn sensors
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        #Radar Data Collectiom
        self.radar = self.blueprint_library.find('sensor.other.radar')
        self.radar.set_attribute("range", f"80")
        self.radar.set_attribute("horizontal_fov", f"60")
        self.radar.set_attribute("vertical_fov", f"25")

        #We will initialise Radar Data
        self.resetRadarData(80, 60, 25)

        self.sensor = self.world.spawn_actor(self.radar, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_radar(data))
        data = np.array(self.radar_data)
        arr = data[-LIMIT_RADAR:]
        obs = {
            'radar_data':arr,
            'speed'     :np.array([0],dtype=np.float32),
            'lane'      :0,
            'yaw'       :np.array([0],dtype=np.float32),
            'diff_angle':np.array([0],dtype=np.float32),
            'norm'      :np.array([19],dtype=np.float32),
        }
        return obs
        # return {"radar_data":arr,"speed": np.array([0]),"yaw": np.array([0]),"lane":0,"diff_angle": np.array([0]),"norm":np.array([19])}  
        # return data 
        # state_obs = []
        # state_obs = np.array(state_obs)
        # return  data[-LIMIT_RADAR:]),"speed":spaces.Box([0]),"yaw":spaces.Box([0]),"lane":spaces.Box([0]),"diff_angle":spaces.Box([0]),"norm":spaces.Box([19])}) #,np.array([0,0,0,0,19])}
        # return np.zeros(1)        #,[np.zeros(1)],[np.zeros(1)],[np.zeros(1)],[np.zeros(1)],[np.array([19])]) #,np.array([0,0,0,0,19])}
        
        # return {arr, np.array([0]),np.array([0]),0,np.array([0]) ,np.array([19]) }
        # return arr,np.array([0,0,0,0,19])
        # return arr,0,0,0,0,19
        # return spaces.Dict({"radar_data":spaces.Box([arr]),"speed": spaces.Box([np.array([0])]),"yaw": spaces.Box([np.array([0])]),"lane":spaces.discrete([0]),"diff_angle":spaces.Box(np.array([0])]),"norm":spaces.Box([np.array([19])])})  
       
        # return spaces.Dict({saarr,0,0,0,0,19}) #,np.array([0,0,0,0,19])}
        
        

    def resetRadarData(self, dist, hfov, vfov):
        # [Altitude, Azimuth, Dist, Velocity]
        alt = 2*math.pi/vfov
        azi = 2*math.pi/hfov

        vel = 0
        deque_list = []
        for _ in range(MAX_LEN//4):
            altitude = random.uniform(-alt,alt)
            deque_list.append(altitude)
            azimuth = random.uniform(-azi,azi)
            deque_list.append(azimuth)
            distance = random.uniform(10,dist)
            deque_list.append(distance)
            deque_list.append(vel)

        self.radar_data.extend(deque_list)


    #Process Camera Image
    def process_radar(self, radar):
        # To plot the radar data into the simulator
        self._Radar_callback_plot(radar)

        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # Parameters :: frombuffer(data_input, data_type, count, offset)
        # count : Number of items to read. -1 means all data in the buffer.
        # offset : Start reading the buffer from this offset (in bytes); default: 0.
        points = np.frombuffer(buffer = radar.raw_data, dtype='f4')
        points = np.reshape(points, (len(radar), 4))
        for i in range(len(radar)):
            self.radar_data.append(points[i,0])
            self.radar_data.append(points[i,1])
            self.radar_data.append(points[i,2])
            self.radar_data.append(points[i,3])


    # Taken from manual_control.py
    def _Radar_callback_plot(self, radar_data):
        current_rot = radar_data.transform.rotation
        velocity_range = 7.5 # m/s
        world = self.world
        debug = world.debug

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))

        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)
            
            norm_velocity = detect.velocity / velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            
            debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))


    #record a collision event
    def collision_data(self, event):
        self.collision_hist.append(event)

    
    def step(self, action):
        #Apply Vehicle Action
        self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))


    def destroy(self):
        """
            destroy all the actors
            :param self
            :return None
        """
        print('destroying actors')
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        print('done')

    def step(self,action_1):
            time.sleep(0.2)
            action = [0.5, float(action_1), 0, False]
            done = False
            #Calculate vehicle speed
            self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))
            kmh = self.get_speed()
            current_location = self.vehicle.get_location()
            current_transform = self.vehicle.get_transform()
            target_location = self.target_waypoint.location
            target_lane = self.next_lane_target
            cur_norm = self.compute_magnitude(target_location, current_location)
            current_position = []
            current_position.append(current_location.x)
            current_position.append(current_location.y)
            current_position.append(current_location.z)
            current_position = np.array(current_position)
            reward = 0
            # norm_dis = math.sqrt((current_location.x - self.end_pos[0])**2+(current_location.y - self.end_pos[1])**2 + (current_location.z - self.end_pos[2])**2)
            for i in range(len(check_pts)):
                if self.given_rew[i]==0:
                    norm_dis = math.sqrt((current_location.x - check_pts[i][0])**2 + (current_location.y - check_pts[i][1])**2)
                    if norm_dis < 3.00:
                        print("reached ",i,"th checkpoint")
                        reward += self.chkpt_reward * (i+1)
                        self.given_rew[i] = 1
            # vehicle_waypoint = self.map.get_waypoint(self.vehicle.get_location())
            self.lane_id_ego = self.map.get_waypoint(self.vehicle.get_location()).lane_id
            self.lane_id_target = self.next_lane_target.lane_id
            self.yaw_vehicle = current_transform.rotation.yaw
            self.yaw_target_road = self.next_lane_target.transform.rotation.yaw
            cur_diff_angle = abs(abs(int(179+self.yaw_vehicle))-abs(int(self.yaw_target_road)))
            # print(f'lane ego=={self.lane_id_ego}----lane target=={self.lane_id_target}----yaw ego=={self.yaw_vehicle}--yaw target=={self.yaw_target_road}')
        
            if (self.lane_id_ego!=self.lane_id_target):
                if(action[1]<0):
                    reward = reward + 30 # More Positive reward Comparative
                if(action[1]>0):
                    reward = reward - 10 # Less Negative reward Comparative
                if len(self.collision_hist) != 0:
                    done = True
                    reward = reward-1000
                reward+=0
            else:
                # Adding a large positive reward if the ego car is in target lane
                reward = reward+100
                if (cur_diff_angle > 5):
                    if(action[1]<0):
                        reward = reward - 30 # Less Negative reward Comparative
                    if(action[1]>0):
                        reward = reward + 50
                if len(self.collision_hist) != 0: # More Positive reward Comparative
                    done = True
                    reward = reward-1000

                # If the ego car is within range of target angle
                if(cur_diff_angle<=5):
                    done = True
                    # reward = reward+500
            
            if cur_norm <= 2:
                done = True
                reward = reward+500

            max_Norm = math.sqrt((110-92)**2 + (134-129.5)**2)  #18.55
            norm_reward = (max_Norm - cur_norm)*2

            reward += norm_reward
            data = np.array(self.radar_data)

            cur_lane = 0
            if self.lane_id_ego == self.lane_id_target:
                cur_lane = 1
        
            # return np.array([1]) , reward, done, {"none":1}
            # if done:
            #     self.destroy()
            info = {}
            arr = data[-LIMIT_RADAR:]
            yaw_v = np.array([self.yaw_vehicle],dtype=np.float32)
            kmh_v = np.array([kmh],dtype=np.float32)
            diff_angle_v = np.array([cur_diff_angle],dtype=np.float32)
            norm_v = np.array([cur_norm],dtype=np.float32)
            obs = {
            'radar_data':arr,
            'speed'     :kmh_v,
            'lane'      :cur_lane,
            'yaw'       :yaw_v,
            'diff_angle':diff_angle_v,
            'norm'      :norm_v,
            }
            return obs,reward,done,info
          
            # return {arr, np.array([kmh]),yaw_v,cur_lane,diff_angle_v ,norm_v }, reward, done, info
            # return arr,reward,done,info
            #return {"radar_data":arr,"speed": np.array([kmh]),"yaw":yaw_v,"lane": cur_lane,"diff_angle":diff_angle_v ,"norm": norm_v }, reward, done, info
            # return {"radar_data":arr,"speed": np.array([0]),"yaw": np.array([0]),"lane":0,"diff_angle": np.array([0]),"norm":np.array([19])}  

    def get_speed(self):
        """
            Compute speed of a vehicle in Kmh
            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Kmh
        """
        vel = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


    def compute_magnitude_angle(self, target_location, current_location, target_yaw, current_yaw):
        '''
        Compute relative angle and distance between a target_location and a current_location
        :param target_location: location of the target object
        :param current_location: location of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
        '''
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        current_yaw_rad = abs(current_yaw)

        target_yaw_rad = abs(target_yaw) 
        diff = current_yaw_rad-target_yaw_rad
        diff_angle = (abs((diff)) % 180.0)

        return (norm_target, diff_angle)

    def compute_magnitude(self, target_location, current_location):
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        return norm_target

if __name__ == '__main__':
    env = CarlaVehicle()
    # env = VecEnv(lambda: env, n_envs=1)
    check_env(env, warn=True, skip_render_check=True)
    ############################### FOR TESTING###############################################
    # model = PPO.load("ppo_carla_custom",env=env)                                           #         
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)  #
    # print("mean reward",mean_reward)                                                       #         
    ##########################################################################################

    ############################### FOR TRAINING ###########################################
    model = PPO('MultiInputPolicy', env,learning_rate =3e-4,n_steps = 200,batch_size=20, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1,tensorboard_log="./ppo_carla_tensorboard/").learn(100_000)
    model.save("ppo_carla_custom")
    print(env.observation_space.shape)
    

    obs = env.reset()
    tot_rew =0
    dones = False
    while dones== False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        tot_rew+=rewards
    print("total reward",tot_rew)
   
    env.destroy()
    n_eps = 5  
    for eps in range(n_eps):
        n_steps = 20
        obs = env.reset()
        time.sleep(1)
        for step in range(n_steps):
            try:
                action,_ = model.predict(obs,deterministic= True)
                print("Action: ", action)
                print("Step {}".format(step + 1))
                obs, reward, done, info = env.step(action)
                time.sleep(0.5)
                # print('obs=', obs, 'reward=', reward, 'done=', done)
                # env.render()
                if done:
                    print("Goal reached!", "reward=", reward)
                    # env.destroy()
                    break
            finally:
                # env.destroy()
                time.sleep(1)
                print("Done ep")
                time.sleep(1)
    