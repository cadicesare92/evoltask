import numpy as np
from scipy.io import loadmat

def generate_waveform(params):
    """Generate normalized waveform (i.e., between 0 - 100). 
       @param:   params = vector of 3 params: kappa = delay, alpha = amplitude, 
                          lambda = duration.
    """
    # Unpack parameters.
    _kappa  = int(params[0])
    _alpha  =     params[1]  
    _lambda = int(params[2])
    
    kappa_min  = 5                  # defines the minimum point before 0 the waveform can be activated (i.e., -5)
    # Kappa (onset) needs to occur in visible window.
    if  _kappa > 100:           
        _kappa = 95
    if  _kappa < -kappa_min:
        _kappa = -kappa_min
        
    # Define some parameters that bound the resultant waveform. The waveform 
    # can be activated before and have a duration that surpasses the 0-100#
    # time frame; define these features.
       
    window_min = 10                                     # defines the minimum window (i.e., duration needs to last at least 5 frames; more if <0% onset)
    lambda_min = np.max([window_min, (6 - _kappa)])     # defines the minimum duration (i.e., if <0% onset, duration has to last at least until 5% stance)
    window_max = kappa_min * 2 + 101                    # defines the maximum window (visible and invisible portions, bounded by kappa_min)
    lambda_max = (window_max - kappa_min - _kappa - 1)  # maximum duration as defined by lambda
           
    # Perform some checks on the duration.   
    if  _lambda < lambda_min:       # needs to be of minimum duration
        _lambda = lambda_min	
    if  _lambda > lambda_max:       # needs to be less than maximum duration
        _lambda = lambda_max
    
    # Define the waveform, accounting for <0% and >100% stance. Then,
    # remove data outside of the 0-100% range for resultant waveform.
    bkappa = _kappa + kappa_min                         # bound kappa by kappa_min
    x      = np.arange(bkappa, bkappa + _lambda + 1)    # define x range of waveform
    y      = np.zeros(window_max)                       # define y range of waveform (i.e., 201 samples; -50%, 0-100%, and +50% stance)  
    y[x]   = (_alpha/2) * np.sin( (2*np.pi / _lambda) * (x - bkappa - (_lambda/4)) ) + (_alpha/2)   # calculate y range of waveform
    
    # Remove data outside of the 0-100% range.
    y_filt = y
    y_filt = np.delete(y_filt, np.arange(0,kappa_min))  # cut off beginning
    y_filt = y_filt[:-kappa_min]                        # cut off ending
    
    return (y_filt, y)


class Basis:
    """
    The Basis class defines the stochastic gradient descent procedures for 
    computing the optimal number of basis torques for each number of basis 
    torques.
    
    @author: Christopher A. DiCesare, CCHMC (2019)
    """
    def __init__(self, torque_names, n_basis, torque_bounds, lambda_bounds):
        """Constructor for objects of type Basis.
        @param:   torque_names  = The names of the torques
        @param:   n_basis       = A list of the number of basis torques per joint torque
        @param:   torque_bounds = Sample data for each torque
        @param:   lambda_bounds = The duration (mean + SD) of each basis torque
        """
#        self.m_max      =  6;  	# maximum number of basis torques per joint torque
        self.lambda_min = 10;   	# minimum lambda (duration)       
#        n_basis[n_basis > self.m_max] = self.m_max;   # ensure n_basis is below specified maximum
        self.torques = {}
        for i, tname in enumerate(torque_names):
            self.torques[tname] = {
                'mean':   np.mean(torque_bounds[tname], axis=0),
                'sd':     np.std(torque_bounds[tname],  axis=0),
                'lambda': lambda_bounds[i, :],
                'params': np.zeros([n_basis[i], 3])         
            }
        self.initialize_params()   
        
        
    def initialize_params(self):
        """Initialize the waveform parameters. This process is summarized as follows:
            1) For each of the T x M basis torques, randomly generate
               the kappa (onset) and lambda (duration parameters).
            2) Randomly sample from the alpha (magnitude) distribution
               at time t = kappa + 1/2 * lambda.
        """
        for tname in self.torques.keys():                               # iterate through each torque
            torque   = self.torques[tname]
            n_basis  = torque['params'].shape[0]
            _kappa   = np.sort( np.random.choice(np.arange(-5,81), 5, replace=False) )    # kappa needs to be selected WITHOUT replacement
            _lambda_mu    = torque['lambda'][0]                         # mean lambda (duration) of the torques
            _lambda_sigma = torque['lambda'][1]                         # SD lambda                                           
            for i in range(n_basis):                                    # generate n_basis basis torques                                    
                _lambda = np.max([                                      # randomize lambda
                    np.random.normal(_lambda_mu, _lambda_sigma),
                    self.lambda_min
                ]).astype(int)
                n_apex  = _kappa[i] + (_lambda/2).astype(int)           # idx of the apex of the waveform                    
                if n_apex <   0: n_apex =   0
                if n_apex > 100: n_apex = 100
                _alpha_mu    = torque['mean'][n_apex]                   # mean alpha (magnitude) of the torques
                _alpha_sigma = torque['sd'][n_apex]                     # SD alpha
                _alpha  = np.random.normal(_alpha_mu, _alpha_sigma)     # randomize alpha 
                self.torques[tname]['params'][i, :] = [_kappa[i], _alpha, _lambda]        # assign params to current torque         
        
    
    def gen_genotype(self):
        """Generate N-long (N = T x B x 4) string of parameters for the agent."""
        genotype = np.array([])
        for i, tname in enumerate(self.torques.keys()):
            params_i = np.c_[
                self.torques[tname]['params'],
                np.ones([self.torques[tname]['params'].shape[0], 1]) * (i+1)
            ]
            params_i = np.reshape(params_i, params_i.size)              # reshape to single-dimension
            genotype = np.hstack([genotype, params_i])
            
        return genotype
        
        
    def gen_torques(self):
        """Generates joint torques from superposition of M basis torque waveforms, 
        using currently defined parameters.
        """
        pass
#        varnames  = fieldnames(this.torques);
#        n_torques = numel(varnames);
#        fit = zeros(numel(n_torques), 101);
#        for i = 1:n_torques
#            params_i  = this.torques.(varnames{i}).params;
#            params_i  = reshape(params_i', 1, []);              # get as single string of parameters
#            fit_i     = this.gen_torque(params_i);
#            fit(i, :) = fit_i;
#
#        return fit

    @staticmethod
    def gen_torque(params):
        """Generate a single waveform from a set of parameters."""
#        if params.shape[0] == 1:
#            params = reshape(params', [], length(params)/3)';	# get as M x 3 matrix of parameters        
        n_basis = params.shape[0]
        fit     = np.zeros([n_basis, 101])
        for i in range(n_basis):
            params_j = params[i, :]
            fit_i, _ = generate_waveform(params_j) 
            fit[i,:] = fit_i                         
        fit = np.sum(fit, axis=0)

        return fit   


class Population:
    """
    The class Population generates a population of agents (i.e., joint-torque
    based coordination strategies) that will be evolved.
    
    @author: Christopher A. DiCesare, CCHMC, UM (2019-2020)
    """
    
    def __init__(self, torque_names, n_basis, torque_bounds, lambda_bounds, n_agents=1000):
        """Constructor for objects of type Population. 
        @param:   torque_names  = The names of the torques
        @param:   n_basis       = A list of the number of basis torques per joint torque
        @param:   torque_bounds = Sample data for each torque
        @param:   lambda_bounds = The duration (mean + SD) of each basis torque
        @param:   n_agents      = The number of agents to generate.
        """
        
        self.torque_names  = torque_names           
            
        # Initialize the cell array containing the candidate torque
        # profiles. Each 'agent' is an N x M x 4 matrix, with the first
        # three columns being the kappa, alpha, and lambda,
        # respectively, with the fourth column indicating which torque
        # the row belongs to (e.g., 1 = RHipMomentPROXIMALX).        
        for i in range(n_agents):
            basis = Basis(torque_names, n_basis, torque_bounds, lambda_bounds)
            genotype = basis.gen_genotype()
            if i == 0:
                agents = genotype
            else:
                agents = np.vstack([agents, genotype])
        
        self.agents = agents
        
        

if __name__ == '__main__':  
    torque_names  = [
        'RHipMomentPROXIMALX',   'RHipMomentPROXIMALY',   'RHipMomentPROXIMALZ',  
        'RKneeMomentPROXIMALX',  'RKneeMomentPROXIMALY',  'RKneeMomentPROXIMALZ', 
        'RAnkleMomentPROXIMALX', 'RAnkleMomentPROXIMALY', 'RAnkleMomentPROXIMALZ'
    ]  
    n_basis       = np.array([5, 5, 5, 2, 3, 3, 2, 3, 3])       # number of basis torques per joint torque (9 DOF)
    lambda_tmp    = np.round( 111 / np.sqrt(n_basis) )
    lambda_bounds = np.c_[lambda_tmp, lambda_tmp/2].astype(int) # lambda (duration) mean and SD
    torque_bounds = loadmat('torque_bounds.mat', variable_names=torque_names)
    
    # Generate a sample Population of agents.
    population    = Population(torque_names, n_basis, torque_bounds, lambda_bounds)

