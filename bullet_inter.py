import numpy as np
import pybullet as p
import random
import time

class Bullet:
    """The Bullet class is an Abstract class from which tasks that will be 
    simulated inherit from.
    
    @author: Christopher A. DiCesare (2019-2020)
    """
    
    motor_names = [
        "right_hip_x",      # flexion/extension
        "right_hip_y",
        "right_hip_z",
        "right_knee_x",
        "right_knee_y",
        "right_knee_z",
        "right_ankle_x",    # dorsi-/plantar-flexion
        "right_ankle_y",
        "right_ankle_z",
        "left_hip_x",       # flexion/extension
        "left_hip_y",       
        "left_hip_z",
        "left_knee_x",
        "left_knee_y",
        "left_knee_z",
        "left_ankle_x",     # dorsi-/plantar-flexion
        "left_ankle_y",
        "left_ankle_z",   
    ]     
    
    def __init__(self, direct, ts, orn_init=None):
        """Constructor for objects of type Bullet."""        
        orn_init = orn_init or {}

        # Connect to the simulator.
        if direct:
            self.cid = p.connect(p.DIRECT)
            self.sleep_time = 0
        else:
            self.cid = p.connect(p.GUI)
            self.sleep_time = 0.05
        p.setTimeStep(ts, physicsClientId=self.cid)
        self.orn_init=orn_init  # default orientation

        # Set up the world, initialize the motors.
        self.initialize_world()
        self.initialize_motors(self.motor_names)
        self.set_up_initial_state(self.displ_init)
        
    # 'Abstract' methods
    def run_simulation(self):
        pass


    def initialize_world(self, plane_fname=".\\plane.urdf", model_fname='humanoid_symmetric.xml'):
        """Sets up the environment in which the humanoid will perform."""
        # Set gravity.
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        
        # Load a plane and the humanoid model to interact with it.
        p.loadURDF(plane_fname, physicsClientId=self.cid)
        self.model, = p.loadMJCF(model_fname,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, physicsClientId=self.cid)        
        

    def initialize_motors(self, motor_names):
        """Initializes the motors used in the simulation."""      
        jdict = {}
        for i in range( p.getNumJoints(self.model, physicsClientId=self.cid) ):
            info = p.getJointInfo(self.model, i, physicsClientId=self.cid)
            jname = info[1].decode("ascii")
            jdict[jname] = i
        self.motors = [jdict[n] for n in motor_names]
        self.motors.insert(0, 1) # add trunk flexion/extension
    
    
    def set_up_initial_state(self, displ_init):
        """Set up the initial state, then save to memory."""
        # Set initial displacement values for each of the joints.
        p.setJointMotorControlArray(self.model, self.motors, controlMode=p.POSITION_CONTROL, targetPositions=displ_init, physicsClientId=self.cid)
        
        # Simulate the fall to the ground.
        for i in range(30):
            time.sleep(self.sleep_time)
            p.stepSimulation(physicsClientId=self.cid)        
        
        # In order to use direct torque control, need to disable the motor.
        #p.setJointMotorControlArray(humanoid, motors, controlMode=p.POSITION_CONTROL, forces=[0 for n in motor_names])
        for i in range( p.getNumJoints(self.model) ):
            p.setJointMotorControl2(self.model, i, controlMode=p.POSITION_CONTROL, force=0, physicsClientId=self.cid)
            p.setJointMotorControl2(self.model, i, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.cid)  
        
        if self.orn_init:
            # Reset orientation.
            pos, orn = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)
            orn = np.array(orn); orn += self.orn_init['values']
            p.resetBasePositionAndOrientation(self.model, posObj=pos, ornObj=orn, physicsClientId=self.cid)             
        
        # Save the state for future use.
        self.state_id = p.saveState(physicsClientId=self.cid)


    def print_info(self):
        """Prints information about the model, simulation, etc."""
        for i in range( p.getNumJoints(self.model) ):
            info  = p.getJointInfo(self.model, i)
            jname = info[1].decode("ascii")
            print('Joint %u: %s' % (i, jname))
        for i in range( len(self.motors) ):
            info  = p.getJointInfo(self.model, self.motors[i])
            jname = info[1].decode("ascii")
            print('Motor %u: %s' % (i, jname))            


class Jumper(Bullet):
    """The Jumper class defines parameters for the drop vertical jump (DVJ) task
    to be simulated.
    
    @author: Christopher A. DiCesare (2019-2020)
    """    

    displ_init = np.array([
        -20,            # trunk extension/flexion
         30, -5, -5,    # right hip
        -20,  0,  0,    # right knee
        -20,  0,  0,    # right ankle
         30, -5, -5,    # left hip
        -20,  0,  0,    # left knee
        -20,  0,  0     # left ankle
    ]) * np.pi/180     
    
    def __init__(self, direct=False, ts=1/333):
        """Constructor for objects of type Jumper."""       
        super().__init__(direct, ts)
    
    
    def run_simulation(self, torques):
        """Runs the simulation given the specified torque inputs."""
        
        def get_jump_height():
            """Wrapper function to get base horizontal (bad) and vertical (good)
            jumping position."""
            pos, _ = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)
            hjump  = np.sqrt( pos[0] ** 2 + pos[1] ** 2 )
            vjump  = pos[2] - vjump_init
            
            return (hjump, vjump)
            
        
        # Load state.
        p.restoreState(self.state_id, physicsClientId=self.cid)
            
        # Apply downward force to the body to simulate landing.
        for i in range(30):
           time.sleep(self.sleep_time)
           p.applyExternalForce(self.model, -1, [0, 0, -300], [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
           p.stepSimulation(physicsClientId=self.cid)     
           
        # The initial state position is the one to beat. 
#        is_jumping = False
        vjump_init = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0][2]
        hjump_max, vjump_max  = -10, -10           
        
        # Apply torques at each time step.
        hip_ext_mult     =  4.5     # multiplier for hip extensor
        torques[:,0]    *= hip_ext_mult
        torques[:,9]    *= hip_ext_mult          
        trunk_ext_torque = [20]
        Fx = -50
        for i in range(torques.shape[0] + 100): # The +100 should be enough to let the 
            time.sleep(self.sleep_time)
            if i <= torques.shape[0]-1:
                torque_i = np.append(trunk_ext_torque, torques[i,:])
                p.applyExternalForce(self.model, -1, [Fx, 0, -300], [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
                p.setJointMotorControlArray(self.model, self.motors, controlMode=p.TORQUE_CONTROL, forces=torque_i, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)
            (hjump, vjump) = get_jump_height()
            if vjump > vjump_max:
#                is_jumping = True
                hjump_max  = hjump
                vjump_max  = vjump
#            if vjump < vjump_max and is_jumping:
#                break
                
#        return (hjump_max, vjump_max) 
                
        # Compute fitness.
        _lambda  = 0.25                                   # coefficient allowing for horizontal movement      
#        (vjump, hjump) = simulators[nsim].run_simulation(torques_i)
        fitness  = vjump_max - _lambda * hjump_max
        
        return fitness


class Lander(Bullet):
    """The Jumper class defines parameters for the single-leg drop landing (SLD) 
    task to be simulated.
    
    @author: Christopher A. DiCesare (2019-2020)
    """  
    
    displ_init = np.array([
        -30,            # trunk flexion/extension
        -15, 10, -5,    # right hip
        -20,  0,  0,    # right knee
        -20,  0,  0,    # right ankle
         20, 25,-25,    # left hip
       -120,  0,  0,    # left knee
        -20,  0,  0     # left ankle
    ]) * np.pi/180        
    
    def __init__(self, direct=False, ts=1/333):
        """Constructor for objects of type Lander."""
        super().__init__(direct, ts)


    def run_simulation(self, torques):
        """Runs the simulation given the specified torque inputs."""
        
        def get_velocity():
            """Wrapper function to get absolute velocity of the model."""
            vel, _ = p.getBaseVelocity(self.model)
            
            return np.sqrt( vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2 )


        # Torques for the left leg are 0.
        torques[:,9:] = 0
        
        # Load state.
        p.restoreState(self.state_id, physicsClientId=self.cid)
            
        # Apply downward force to the body to simulate landing.
        for i in range(30):
           time.sleep(self.sleep_time)
           p.applyExternalForce(self.model, -1, [0, 0, -150], [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
           p.stepSimulation(physicsClientId=self.cid) 
           
        # The initial state position is the goal. 
        vcog_init  = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0][2]
        
        # Apply torques at each time step.
        trunk_ext_torque = [0]
        torques[:, 0] *= np.linspace(2,0.5,101)[:torques.shape[0]]
        torques[:, 3] *= np.linspace(2,0.5,101)[:torques.shape[0]]
        torques[:, 6] *= 0.5        
        Fx = -50
        Fy =  50
        for i in range(torques.shape[0]):
            time.sleep(self.sleep_time)
            if i <= torques.shape[0]-1:
                torque_i = np.append(trunk_ext_torque, torques[i,:])
                p.applyExternalForce(self.model, -1, [Fx, Fy, -150], [0, 0, 0], flags=p.WORLD_FRAME,  physicsClientId=self.cid)
                p.setJointMotorControlArray(self.model, self.motors[10:], controlMode=p.POSITION_CONTROL, targetPositions=self.displ_init[10:], physicsClientId=self.cid)
                p.setJointMotorControlArray(self.model, self.motors,      controlMode=p.TORQUE_CONTROL,   forces=torque_i, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)         
        
        # Get final displacement, velocity of the COM.  
        vcog_final = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0][2]
        vcog_displ = np.abs(vcog_init - vcog_final)
#        vcog_displ = np.sqrt( np.sum( np.array(vcog_final) - np.array(vcog_init) ** 2 ) )
        vcog_veloc = get_velocity()        
        
#        return (vcog_displ, vcog_veloc) 
#    
        # Compute fitness.
#        fitness    = (0.1 / vcog_displ) * (0.01 / vcog_veloc)
        fitness    = 1 / (vcog_displ * vcog_veloc)
        
        return fitness     

                    
class Cutter(Bullet):
    """The Cutter class defines parameters for the single-leg lateral cut (CUT) 
    task to be simulated.
    
    @author: Christopher A. DiCesare (2019-2020)
    """    
    
    displ_init = np.array([
        -20,            # trunk extension/flexion
         30, -5, -5,    # right hip
        -20,  0,  0,    # right knee
        -20,  0,  0,    # right ankle
         30, -5, -5,    # left hip
        -20,  0,  0,    # left knee
        -20,  0,  0     # left ankle
    ]) * np.pi/180        
    
    def __init__(self, direct=False, ts=1/333):
        """Constructor for objects of type Cutter."""       
        orn_init={'values': np.array([0, 0.15, 0, 0])}
        super().__init__(direct, ts, orn_init)


    def run_simulation(self, torques):
        """Runs the simulation given the specified torque inputs."""
                      
        # Torques for the left leg are 0.
        torques[:,9:] = 0        
        
        # Load state.
        p.restoreState(self.state_id, physicsClientId=self.cid)
        for i in range( p.getNumJoints(self.model) ):
            p.setJointMotorControl2(self.model, i, controlMode=p.POSITION_CONTROL, force=0, physicsClientId=self.cid)
            p.setJointMotorControl2(self.model, i, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.cid)         
#        p.setJointMotorControlArray(self.model, self.motors, controlMode=p.POSITION_CONTROL, targetPositions=self.displ_init, physicsClientId=self.cid)
            
        # Apply downward force to the body to simulate landing.
        for i in range(60):
           time.sleep(self.sleep_time)
           p.applyExternalForce(self.model, -1, [0, 0, -150], [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
           p.stepSimulation(physicsClientId=self.cid) 
           
        # The target cut position is the goal (1.6 meter diagonally) 
#        vcog_init = p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0][2]
        vcog_goal        = np.array( p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0] )
        vcog_goal[:2]   += 1

        displ_init = np.array([
             20, 25,-25,    # left hip
           -120,  0,  0,    # left knee
            -20,  0,  0     # left ankle
        ]) * np.pi/180               
        
        # Apply torques at each time step.
        trunk_ext_torque = [0]
        torques[:, 0]   *= 2
        torques[:, 1]   *= 2
        torques[:, 3]   *= 2
        torques[:, 6]   *= 0.5
#        p.setJointMotorControlArray(self.model, self.motors[10:], controlMode=p.POSITION_CONTROL, targetPositions=displ_init, physicsClientId=self.cid)
#        torques[:,9:]    = np.transpose(np.transpose(torques[:,9:]) * np.linspace(1,0,101))          
        Fx, Fy =  50,  50  
#        Fx, Fy = 100, 100
#        Fx = 100
#        Fy = 100    
        for i in range(torques.shape[0]):
            time.sleep(self.sleep_time)
            if i <= torques.shape[0]-1:
                torque_i = np.append(trunk_ext_torque, torques[i,:])
                p.applyExternalForce(self.model, -1, [Fx, Fy, -150], [0, 0, 0], flags=p.WORLD_FRAME,  physicsClientId=self.cid)
#                p.setJointMotorControlArray(self.model, self.motors[10:], controlMode=p.POSITION_CONTROL, targetPositions=displ_init, physicsClientId=self.cid)
                p.setJointMotorControlArray(self.model, self.motors,      controlMode=p.TORQUE_CONTROL,   forces=torque_i, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)         
        
        # Get final displacement, velocity of the COM.  
        vcog_final = np.array( p.getBasePositionAndOrientation(self.model, physicsClientId=self.cid)[0] )
#        vcog_displ = np.abs(vcog_init - vcog_final)
        vcog_displ = np.sqrt( np.sum( (vcog_final - vcog_goal) ** 2 ) )
        
#        return (vcog_displ, vcog_veloc) 
#    
        # Compute fitness.
        fitness    = 1 / vcog_displ
        
        return fitness
        

if __name__ == '__main__':
    import pandas as pd
    
    varnames    = [
        'RHipMomentPROXIMALX',
        'RHipMomentPROXIMALY',
        'RHipMomentPROXIMALZ',
        'RKneeMomentPROXIMALX',
        'RKneeMomentPROXIMALY',
        'RKneeMomentPROXIMALZ',
        'RAnkleMomentPROXIMALX',
        'RAnkleMomentPROXIMALY',
        'RAnkleMomentPROXIMALZ'
    ] 
    
    # Test landing simulation.
    fname    = './sample_data/dvj_torques.csv'
    torques  = pd.read_csv(fname, delimiter=',')
    torques  = torques[varnames].values
    torques  = np.hstack([torques, torques])    # make it perfectly symmetrical bilateral strategy
    torques *= -1               # torques are EXTERNAL; make them INTERNAL       

    model   = Jumper(direct=False)
    # model   = Lander(direct=False)
    # model   = Cutter(direct=False)
    model.run_simulation(torques)
   