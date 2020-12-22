from multiprocessing import Array, Process
import numpy as np
import pybullet as p
from scipy.spatial.distance import cdist
import sys
import timeit
from basis import Basis, Population
import bullet as b

nproc = 4
simulators = []     # will contain the N simulators that will be used in parallel
for i in range(nproc):
    simulators.append(b.Cutter(direct=True))
#    simulators.append(b.Lander(direct=True))

def generate_torques(agent):
    """Generates the torque profile defined by the current waveform 
    parameters (i.e., wparams)."""
    ntorque  = np.max( agent[:,-1] ).astype(int)
    torques  = np.zeros([ntorque, 101])
    for i in range(ntorque):
        params_i = agent[agent[:,-1] == i+1, :-1]
        torque_i = Basis.gen_torque(params_i)
        torques[i,:] = torque_i

    return torques 
    

def compute_fitness(nsim, agents, idx, fitness):
    """Evaluates task performance, using parallelization (if specified).
    @param:     nsim    = The ID of the PyBullet instance (1 per processor).
    @param:     agents  = The agents to evaluate.
    @param:     idx     = The indices of the agents to iterate, simulate for (a la MapReduce).
    @param:     fitness = Shared memory array that will hold computed fitness of each agent.
    """

    for i in np.arange(idx[0], idx[1]+1):
        
        agent_i    = agents[i,:].reshape([-1,4])
        
        # Compute torques from agent.
        torques_i  = generate_torques(agent_i)  
        torques_i  = np.transpose(torques_i)                # make 101 x 9 matrix
        torques_i *= -1                                     # torques are EXTERNAL; make them INTERNAL
        norm_factor= 1.6 * 55                               # 1.6 m tall, 55 kg person
        torques_i *= norm_factor        
        
        torques_bl = np.hstack([torques_i, torques_i])      # DVJ is bilateral; duplicate the torques
        fitness_i  = simulators[nsim].run_simulation(torques_bl)
        
        # Compute the 'feasibility index'. This is the inverse of the
        # proportion of the profile where there is no activity at each
        # of the joints.
        _alpha     = 1                                      # coefficient allowing for feasibility of strategies
        thresh     = 0.5                                    # activity threshold
        # activity at each % of stance
        act_idx    = np.sum(                                
             np.abs(torques_i) <= np.max(np.abs(torques_i) * thresh, axis=0)
        )  
        feas_idx   = 1 - (act_idx / torques_i.size)         # entire stance phase     
        if fitness_i <= 0:
            fitness_i = 0
        else:
            fitness_i *= (feas_idx ** _alpha)   
        
        fitness[i] = fitness_i


class Evolve:
    """The Evolve class defines functionality for "evolving" a population of 
    candidate joint torque profiles, using a trained neural net as the 
    mapping between joint torque % and task performance (i.e., GRF profile).
    
    @author: Christopher A. DiCesare, CCHMC (2019-2020)
    """
    
    mean_fitness  = np.array([])   # will hold mean fitness at each epoch
    peak_fitness  = np.array([])   # will hold peak fitness at each epoch
    diversity     = False          # True to apply diversity function at each epoch
    n_epoch       = 0              # the number of the current generation
    tolerance_min = 0.0001         # the convergence tolerance
    norm_factors  = {}             # the mu, sigma used to compute dissimilarity.    
        
    def __init__(self, population=None, params=None):
        """Constructor for objects of type Evolve.
        @param:   population = An instance of Population containing the
                               population that will be evolved.
        @param:   params     = A dict of evolution parameters that will be applied.
        """
        if not population: return
        self.population = population          
        # Set default parameters.
        for key, val in params.items():
            setattr(self, key, val) 
      
        # Finally, evaluate performance for the initial generation.
        self.elapsed = 0
        self.evaluate_generation()
    
    
    def evaluate_generation(self):
        """Evaluate the current generation by finding the fitness of each
        agent in the population and sorting by descending fitness."""
        
        self.n_epoch += 1
        
        start_time    = timeit.default_timer()

        # Evaluate fitness, similarity.            
        fitness_cur   = self.compute_population_fitness()
        dissimilarity = self.compute_population_dissimilarity()
        if self.diversity:
            fitness_cur  *= dissimilarity                       
        
        # Sort by fitness.
        sorted_idx    = np.argsort(fitness_cur)      # ascending order
        self.fitness  = fitness_cur[sorted_idx]
        self.population.agents = self.population.agents[sorted_idx,:]

        # Keep track of mean, peak fitness.
        self.mean_fitness = np.append(self.mean_fitness,
            np.mean( self.fitness[int(self.population.agents.shape[0]/2):] )
        )
        self.peak_fitness = np.append(self.peak_fitness,  np.max(self.fitness))
        
        self.elapsed += (timeit.default_timer() - start_time)
        
        sys.stderr.write('\rEpoch %s/%u... | Fitness = %3.3f | Time (s): %3.1f' % (str(self.n_epoch).rjust(3), self.n_epoch_max, self.mean_fitness[self.n_epoch-1], self.elapsed))        
        sys.stderr.flush()
        
        
    def compute_population_fitness(self):
        """Evaluate fitness of the agents in the current generation."""
        """Use ballistic equations to compute the net effect of vertical
        jumping. This process is summarized as follows; for each agent:
            1) Compute the torque profile based on its basis torque
               parameters.       
            2) Map the torque profile to GRF using the NN function
               generated for the individual.
            3) Use ballistic equations (impulse-momentum) to determine
               COG behavior; assign a performance metric based on specified
               goal of the task (i.e., maximum vertical displacement,
               minimum horizontal displacement).
        """        
        
        nagents = self.population.agents.shape[0]     
        fitness = Array('d', nagents)
        
        # Simulation takes a while. Parallelize using available processors.
        procs     = []
        agent_idx = np.c_[
            np.linspace(0, nagents, self.nproc+1).astype(int)[:-1],
            np.linspace(0, nagents, self.nproc+1).astype(int)[1:]-1
        ]
        for i in range(self.nproc):
            proc = Process(target=compute_fitness, args=(i, self.population.agents, agent_idx[i,:], fitness))
            proc.start()
            procs.append(proc)
        for proc in procs: proc.join()
                                
        return np.array(fitness)
    
    
    def compute_population_dissimilarity(self):
        """Evaluate dissimilarity of the agents in the current population."""

        # Compile all agents into a N x (T x B x 3) matrix.
        n_agents = self.population.agents.shape[0]
        similarity_matrix = np.zeros([
            n_agents,
            self.population.agents[0].reshape([-1,4]).shape[0] * 3
        ])
        for i in range(n_agents):
            agent_i = self.population.agents[i].reshape([-1,4])
            similarity_matrix[i, :] = agent_i[:,:-1].flatten()
     
        # Assume the mean (dis)similarity of initial population is
        # maximally diverse. Make this measure of dissimilarity the 
        # normalizing factor for future generations
        if not self.norm_factors:
            self.norm_factors['mu']       = np.mean(similarity_matrix, axis=0)
            self.norm_factors['sigma']    =  np.std(similarity_matrix, axis=0)
                    
        matrix_norm = (similarity_matrix - self.norm_factors['mu']) / self.norm_factors['sigma']

        # Compute dissimilarity of population. Use to promote population diversity.
        dissimilarity = np.sum( cdist( matrix_norm, matrix_norm ), axis=0 ) / matrix_norm.shape[0]
        if not 'constant' in self.norm_factors.keys():
            self.norm_factors['constant'] = np.mean(dissimilarity)
        dissimilarity = dissimilarity / self.norm_factors['constant']

        return dissimilarity
    
    
    def evolve_generation(self):
        """Evolve the current population to the next generation."""

#        print('Evolving generation %s...' % str(self.n_epoch).rjust(3))
#        sys.stderr.write('Evolving... | ')
            
        # Select individuals for reproduction based on proportional 
        # fitness. This is selection with replacement (i.e., can have
        # more than one offspring).
        norm_fitness = self.fitness
        norm_fitness[norm_fitness < 0] = 0                                          # if negative, have no chance of reproducing 
        prob_vector  = norm_fitness / np.sum(norm_fitness)                          # probability vector has cumsum = 1 
        n_to_replace = int(self.p_to_replace * self.population.agents.shape[0])     # number of agents to replace
        reproduction = np.zeros([n_to_replace, 2], dtype=int)                       # pairs of agents that will produce offspring            
        for i in range(n_to_replace):
            n_agent_1 = np.sum( np.random.rand() >= np.cumsum(prob_vector) )
            n_agent_2 = np.sum( np.random.rand() >= np.cumsum(prob_vector) )
            if n_agent_1 == n_agent_2:    # can't reproduce with self
                while n_agent_1 == n_agent_2:
                    n_agent_2 = np.sum( np.random.rand() >= np.cumsum(prob_vector) )
            reproduction[i, :] = [n_agent_1, n_agent_2]
            
        # Reproduction, using single-point crossover and mutation.
        offspring = np.zeros([n_to_replace, self.population.agents.shape[1]])
        for i in np.arange(0, n_to_replace, 2):
            agent_1   = self.population.agents[reproduction[i,0],:].reshape([-1, 4])
            agent_2   = self.population.agents[reproduction[i,1],:].reshape([-1, 4])            
#            # MAJOR disruption (using multiple split points)
#            off_1     = np.zeros(agent_1.shape)
#            off_2     = np.zeros(agent_1.shape)
#            idx       = np.round(np.random.rand(agent_1.shape[0])).astype(bool)
#            off_1[ idx, :] = agent_1[ idx, :]
#            off_1[~idx, :] = agent_2[~idx, :]
#            off_2[~idx, :] = agent_1[~idx, :]
#            off_2[ idx, :] = agent_2[ idx, :]
            # MINOR disruption (using single split point)
            split_idx = np.random.randint(0, agent_1.shape[0]-1)
            off_1     = np.vstack([agent_1[:split_idx,:], agent_2[split_idx:,:]])
            off_2     = np.vstack([agent_2[:split_idx,:], agent_1[split_idx:,:]])           
            # MUTATION (amplify/suppress single nucleotide of single agent)
            if np.random.rand() <= self.mutation_rate:
                idx_row = np.random.randint(0, off_1.shape[0]-1)
                idx_col = np.random.randint(0, 2)
                amp     = 1.0 
                if idx_col == 1:
                    means    = np.random.normal(0.5, 0.25)
                    if np.random.rand() >= 0.5:
                        amp  = -1.0 
                    mutation = amp * means
                else:   # 0 or 2
                    means    = np.random.normal(0, 15)
                    mutation = int( amp * means )
                if np.random.rand() <= 0.5:
                    off_1[idx_row, idx_col] += mutation
                else:
                    off_2[idx_row, idx_col] += mutation  
            # Record offspring.
            offspring[i,:]   = off_1.reshape(off_1.size)
            offspring[i+1,:] = off_2.reshape(off_2.size)                    
            
        # Replacement, using elitism and proportional replacement, by
        # selecting according to inverse probability WITHOUT replacement.
        # Steps of this procedure:
        #    1) Remove agents with fitness level of 0; NO chance of
        #       reproducing. Adjust n_to_replace accordingly.
        #    2) Compute inverse proportionality according to fitness.
        #    3) Randomly select, WITHOUT replacement, the n_to_replace.
        
        # Step 1.
        n_unfit = np.sum( norm_fitness == 0 )                   # will automatically be replaced in next generation
        if n_unfit >= n_to_replace:
            n_unfit = n_to_replace            
        n_to_replace -= n_unfit;                                # these replacements need to be randomly selected
        
        # Steps 2 and 3.
        norm_fitness = norm_fitness[norm_fitness > 0]           # only those agents that have a possibility of reproducing    
        repl_fitness = norm_fitness[:-self.n_to_save][::-1]     # replacement probability is inverse of fitness  
        replacement  = np.zeros(n_to_replace, dtype=int)
        for i in range(n_to_replace):
            prob_vector = repl_fitness / np.sum(repl_fitness)
            n_agent     = np.sum( np.random.rand() >= np.cumsum(prob_vector) )               
            replacement[i] = n_agent
            # Remove agent (making 0 gives it no probability of being selected again).
            repl_fitness[n_agent] = 0
        # Add n_unfit to repl_fitness to get indices of replacements, 
        # then add indices of those who have 0 fitness.
        replacement += n_unfit
        replacement  = np.append(np.arange(0, n_unfit), np.sort(replacement))
        
        # Replace; evaluate new performance.
        self.population.agents[replacement,:] = offspring
        self.evaluate_generation()

    
    def plot_fitness(self):
        """Plot mean and peak fitness as a function of evolution."""
        pass
#        f  = figure('Position', [1, 1, 840, 420]); hold on
#        ax = gca();
#        plot(this.mean_fitness, 'r-', 'LineWidth', 2)
#        plot(this.peak_fitness, 'b-', 'LineWidth', 2)
#        set(ax, 'XLim', [1, this.n_epoch], 'FontSize', 12)
#        ax.XTick = [1, ax.XTick];
#        legend({'Mean Fitness', 'Peak Fitness'}, 'location', 'northwest', 'FontSize', 12);
#        xlabel('Epoch', 'FontSize', 16)
#        ylabel('Fitness', 'FontSize', 16)            
#   
#        return f
    
    
    def plot_inputs(self, agent_idx):
        """Plots the specified agent's inputs."""
        # Generate the agent's torque profile. 
        agent  = self.population['agents'][-1,:]    # best performing agent
        inputs = self.generate_input_from_agent(agent)
        
        # Generate the figure.
        import matplotlib.pyplot as plt
        plt.figure()#'Position', [1 1 1200 600]);
        varnames = self.population['torque_names']
        ax_names = [v[1:].replace('PROXIMAL', ' ').replace('X', '(Sagittal)').replace('Y', '(Frontal)').replace('Z', '(Transverse)').replace('Moment', ' Moment') for v in varnames]
        x = np.arange(0,101)
        for i in range(inputs.shape[0]):
            ax = plt.subplot(2, 3, i)
            plt.plot(x, inputs[i,:], 'r-', 'LineWidth', 2)
            ax.set_xlim([0, 100])
    #        set(gca, 'FontSize', 12, 'XLim', [0 100], 'XTick', [0 100])
            ax.xlabel('Stance (%)', fontsize=14)
            ax.ylabel(ax_names[i], fontsize=14)


if __name__ == '__main__':
    import os
    import pickle
    from scipy.io import loadmat
    
#    # Generate population.
#    torque_names  = [
#        'RHipMomentPROXIMALX',   'RHipMomentPROXIMALY',   'RHipMomentPROXIMALZ',  
#        'RKneeMomentPROXIMALX',  'RKneeMomentPROXIMALY',  'RKneeMomentPROXIMALZ', 
#        'RAnkleMomentPROXIMALX', 'RAnkleMomentPROXIMALY', 'RAnkleMomentPROXIMALZ'
#    ]  
#    n_basis       = np.array([5, 5, 5, 2, 3, 3, 2, 3, 3])       # number of basis torques per joint torque (9 DOF)
#    lambda_tmp    = np.round( 111 / np.sqrt(n_basis) )
#    lambda_bounds = np.c_[lambda_tmp, lambda_tmp/2].astype(int) # lambda (duration) mean and SD
#    torque_bounds = loadmat('torque_bounds.mat', variable_names=torque_names)
#    population    = Population(torque_names, n_basis, torque_bounds, lambda_bounds, n_agents=200)
    
    # Save/load population for comparison purposes.
    outname = os.path.join('.', 'sample_population.dat')  
    #with open(outname, 'wb') as f:
    #    pickle.dump(population, f)  
    with open(outname, 'rb') as f:
        population = pickle.load(f)    
    
    try:
        p.disconnect()
    except:
        pass
    
    # Evolve the Population.
    params = {
       'n_to_save':          5,   	# The number of top performers in each generation to save (elitism).
       'p_to_replace':       0.2,   # In a delete-n-last configuration, replace this proportion of the population with offspring. 
       'n_epoch_max':      100,     # The maximum number of generations.
       'mutation_rate':      0.01,  # Rate of mutation.
       'nproc':           nproc,    # Number of processors to use
    }
    evolve = Evolve(population, params)
    fitness_delta = evolve.tolerance_min             # current fitness change
    while evolve.n_epoch < evolve.n_epoch_max and fitness_delta >= evolve.tolerance_min:
        evolve.evolve_generation()
        fitness_delta = np.mean( np.diff( evolve.mean_fitness[np.max([0, evolve.n_epoch-5]):evolve.n_epoch] ) )
#    p.disconnect()  
        
#    for i in range(nproc):
#        p.disconnect(simulators[i].cid)
        
    # Save the evolved population w. pickle.
    
    if not os.path.exists(os.path.join('.', 'output')):
        os.mkdir(os.path.join('.', 'output'))
    outname = os.path.join('output', 'evolve_pybullet_0000.dat')
    with open(outname, 'wb') as f:
        pickle.dump(evolve, f)

