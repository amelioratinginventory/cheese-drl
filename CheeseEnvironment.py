from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ray
import time
import scipy.stats as st
import scipy.integrate as sigr
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB
from ray.rllib.env.env_context import EnvContext
from stochastic.processes.diffusion import vasicek as vsk

from scipy.interpolate import RegularGridInterpolator, interp1d

import statsmodels.distributions.copula.api as cop

#-----------------------------------#
class CheeseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, config=EnvContext):
        
        self.render_mode = config["render_mode"] if "render_mode" in config else None
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        #----------------------------------------------------------------------------------------------------------------------------------------------------------#
        #READ_IN PROBLEM CONFIGURATION FROM DICTIONARY
        
        #problem size
        self.numAges = config["numAges"] if "numAges" in config else 5   #the number of age classes in the inventory system
        self.ages = [i for i in range(self.numAges)]
        self.nProducts = config["nProducts"] if "nProducts" in config else 2   #the number of products
        self.products = [p for p in range(self.nProducts)]
        self.nProcesses = self.nProducts+1
        self.processes = [p for p in range(self.nProcesses)]
        self.targetAges = config["targetAges"] if "targetAges" in config else [1,3]   #the minimum storage time of the considered products
        self.maxInventory = config["maxInventory"] if "maxInventory" in config else 50   #the maximum inventory in each age class
    
        #demand distributions 
        self.demand_means = config["demand_means"] if "demand_means" in config else [10.0,7.0]
        self.demand_covs = config["demand_covs"] if "demand_covs" in config else [0.15,0.15]
        self.demand_elasticity = config["demand_elasticity"] if "demand_elasticity" in config else [-0.27, -0.52]  
        self.salvage = config["salvage"] if "salvage" in config else [0.5,0.5]
        
        #price processes (purchase and sales)
        self.price_means = config["price_means"] if "price_means" in config else [170.0, 250.0, 350.0]
        self.speed = config["speed"] if "speed" in config else [0.1, 0.1, 0.1]
        self.vol = config["vol"] if "vol" in config else [10,10,10]
        self.price_covs = config["price_covs"] if "price_covs" in config else [np.sqrt(self.vol[p]**2/(1-(1-self.speed[p])**2))/self.price_means[p] for p in range(self.nProcesses)]
        self.correlation_matrix = config["correlation_matrix"] if "correlation_matrix" in config else [[1.0,0.7,0.7],[0.7,1.0,0.95],[0.7,0.95,1.0]]
        self.priceDistributions = [st.norm(self.price_means[i], self.price_covs[i]*self.price_means[i]) for i in range(self.nProcesses)]
        self.multi_norm = st.multivariate_normal(mean=np.zeros(self.nProcesses), cov=self.correlation_matrix)
        self.norm = st.norm(0,1)
        
        #other reward function parameters
        self.holdingCosts = config["holdingCosts"] if "holdingCosts" in config else 2.5
        
        #evaportaion rate
        self.evaporation = config["evaporation"] if "evaporation" in config else ([0.04 for _ in range(4)] + [0.015 for _ in range(4)] + [0.01 for _ in range(2)])

        self.min_ppf = config["min_ppf"] if "min_ppf" in config else 1e-12
        self.max_ppf = config["max_ppf"] if "max_ppf" in config else 1.0 - 1e-12
        #maximum production amount
        
        #time horizon for each RL episode - initialize at 0
        self.max_horizon = config["horizon"] if "horizon" in config else 5000
        self.n_steps = 0

        #set initial price to mean of price distribution
        self.prices = deepcopy(self.price_means)

        #forbid outdating
        self.avoid_outdating = config["avoid_outdating"] if "avoid_outdating" in config else True

        #allow blending
        self.allow_blending = config["allow_blending"] if "allow_blending" in config else True
        self.blending_range = config["blending_range"] if "blending_range" in config else None
        #define issuance flexibility
        if self.allow_blending:
            if self.blending_range is None:
                self.ageRange = config["ageRange"] if "ageRange" in config else [[self.targetAges[p]] for p in self.products]
            else:
                self.ageRange = [[i for i in range(self.targetAges[p],self.targetAges[p]+self.blending_range)] for p in self.products]
                self.ageRange[self.nProducts-1].append(self.targetAges[self.nProducts-1]+self.blending_range)
        else:
            self.ageRange = [[self.targetAges[p]] for p in self.products]
        self.issuance_ages = [i for i in range(self.numAges) if any(i in self.ageRange[p] for p in self.products)]
        
        #assert that there are no duplicate age classes in the age ranges across products
        assert sum(len(self.ageRange[p]) for p in self.products) == len(self.issuance_ages)


        #select actions according to heuristic (circumvents RL agent)
        self.simulate_heuristic = config["simulate_heuristic"] if "simulate_heuristic" in config else False
        
        self.use_cdfs_for_regularization = False
        self.cdf_buffer = None

        #possible production levels for issuance heuristic
        self.use_issuance_model = config["use_issuance_model"] if "use_issuance_model" in config else False
        self.production_step_size = config["production_step_size"] if "production_step_size" in config else 0.05
        self.production_step_size_lp = config["production_step_size_lp"] if "production_step_size_lp" in config else 0.05
        self.price_step_size = config["price_step_size"] if "price_step_size" in config else 0.05
        assert abs(self.production_step_size_lp / self.production_step_size - round(self.production_step_size_lp / self.production_step_size)) < 1e-9
        self.time_horizon_redux = config["time_horizon_redux"] if "time_horizon_redux" in config else 0
        self.demand_max = config["demand_max"] if "demand_max" in config else [self.demandDistributions[p].ppf(self.max_ppf) for p in self.products]
        
        self.production_levels = {p: [round(i,2) for i in np.arange(0,self.demand_max[p]+self.production_step_size,self.production_step_size)] for p in self.products}
        self.demand_max = [self.production_levels[p][-1] for p in self.products]
        print("SALES BOUND: ", self.demand_max)
        self.production_levels_lp = {p: [round(i,2) for i in np.arange(0,self.demand_max[p], self.production_step_size_lp)] for p in self.products}
        for p in self.products:
            if self.demand_max[p] not in self.production_levels_lp[p]:
                self.production_levels_lp[p] += [self.demand_max[p]]
        self.price_levels = {p: [round(i,2) for i in np.arange(self.priceDistributions[p].ppf(self.min_ppf),self.priceDistributions[p].ppf(self.max_ppf),self.price_step_size)] for p in range(self.nProcesses)}

        #check common random number setting
        self.use_common_random_numbers = config["use_common_random_numbers"] if "use_common_random_numbers" in config else False
        if self.use_common_random_numbers:
            self.use_cdfs_for_regularization = True
            self.cdf_buffer = {i: self.multi_norm.rvs() for i in range(self.max_horizon)}

        #----------------------------------------------------------------------------------------------------------------------------------------------------------#
        
        #accumulated evaporation losses
        self.evaporation_remains_per_age_class = [np.prod([1-self.evaporation[a] for a in range(i+1)]) for i in self.ages]
        
        #expected revenues (piecewise-linear approximation for lookahead LPs)
        if "expected_revenue" in config:
            self.expected_revenue = config["expected_revenue"]
        else:
            self.expected_revenue = {p: {l: {g: self.expected_revenue_function(p,l,g) for g in self.price_levels[p+1]} for l in self.production_levels[p]} for p in self.products}
        
        self.revenue_interpolators = {p: RegularGridInterpolator((self.production_levels[p], self.price_levels[p+1]), np.array([[self.expected_revenue[p][l][g] for g in self.price_levels[p+1]] for l in self.production_levels[p]]), method="linear") for p in self.products}
        self.price_interpolators = {p: {l: interp1d(self.price_levels[p+1], np.array([self.expected_revenue[p][l][g] for g in self.price_levels[p+1]])) for l in self.production_levels_lp[p]} for p in self.products}
        
        if "slope" in config:
            self.slope = config["slope"]
        else:
            self.slope = {p: {l: {g: self.slope_function(p,l,g) for g in self.price_levels[p+1]} for l in self.production_levels[p]} for p in self.products}

        #parameters for scaling rewards; #initial inventory = mean demands
        if "upper_bound" in config and config["upper_bound"] is not None:
            self.max_reward = config["upper_bound"]["max_reward"]
            self.inventory_position = np.array(config["upper_bound"]["inventory_position"])
        else:
            ub = upper_bound(self, discr_step=self.production_step_size)
            self.max_reward = ub["max_reward"]
            self.inventory_position = np.array(ub["inventory_position"])
            # self.max_reward = np.dot(self.demand_means, self.price_means[1:])
            # self.inventory_position = np.array([sum(self.demand_means[p] for p in self.products if self.targetAges[p]>=i) for i in self.ages])
        self.init_inventory = deepcopy(self.inventory_position)
        self.min_reward = 0
        self.reward_lb = config["reward_lb"] if "reward_lb" in config else -1.0
        self.reward_ub = config["reward_ub"] if "reward_ub" in config else 1.0
        print("MAX REWARD: ", self.max_reward)
        print("MIN_REWARD: ", self.min_reward)

        #set lookahead horizon
        self.n_time_steps = self.numAges + 1 - self.time_horizon_redux
        self.time_periods = [i for i in range(self.n_time_steps)]
        self.price_pipelines = np.array([self.prices for _ in self.time_periods])

        #initialize heuristic LP
        self.create_heuristic_lp()

        #derive heuristic action from lookahead model
        self.heuristic_model.optimize()
        self.heuristic_action = [self.heuristic_model.getVarByName("inv[0,1]").X] + [self.heuristic_model.getVarByName(f"iss[{a},0]").X for a in self.issuance_ages]

        print("starting inventory: ", self.inventory_position)
        
        #create initial purchasing volume 
        self.just_purchased = self.heuristic_action[0]
        
        if self.use_issuance_model:
            #initialize lookahead LP for issuance/production
            self.action_space = spaces.Box(low=np.full((len(self.products)+1,),-1.0), high=np.full((len(self.products)+1,),1.0), shape=(len(self.products)+1,))
        else:
            if self.allow_blending:
                if self.avoid_outdating:
                    #if outdating is prevented, the last age class does not have to be included in the action space
                    self.action_space = spaces.Box(low=np.full((len(self.issuance_ages),),-1.0), high=np.full((len(self.issuance_ages),),1.0), shape=(len(self.issuance_ages),))
                else:
                    #otherwise issuance volumes from all applicable age classes are included in the action space
                    self.action_space = spaces.Box(low=np.full((len(self.issuance_ages)+1,),-1.0), high=np.full((len(self.issuance_ages)+1,),1.0), shape=(len(self.issuance_ages)+1,))
            else:
                self.action_space =  spaces.Box(low=np.full((self.nProducts,),-1.0), high=np.full((self.nProducts,),1.0), shape=(self.nProducts,))
        if self.use_issuance_model:
            original_space = spaces.Dict({
                "prices": spaces.Box(low=0.0, high=1.0, shape = (self.nProducts+1,)),
                "inventory": spaces.Box(low=np.full((self.numAges,),0.0), high=np.full((self.numAges,),1.0), shape=(self.numAges,)),
                "max_production": spaces.Box(low=np.full((self.nProducts,),0.0), high=np.full((self.nProducts,),1.0), shape=(self.nProducts,)),
            })
            self.max_production = np.array([self.demand_max[p] for p in self.products])
        else:
            original_space = spaces.Dict({
                "prices": spaces.Box(low=0.0, high=1.0, shape = (self.nProducts+1,)),
                "inventory": spaces.Box(low=np.full((self.numAges,),0.0), high=np.full((self.numAges,),1.0), shape=(self.numAges,)),
            })
        
        self.observation_space = original_space
      
    #function for calculating expected revenues given production volume 
    def expected_revenue_function(self, p: int, x: float, gamma: float):
        x = min(self.demand_max[p], x)
        new_mean = self.demand_means[p] * (1 + (gamma-self.price_means[p+1])/self.price_means[p+1]*self.demand_elasticity[p])
        demand_distribution = st.norm(new_mean, self.demand_covs[p]*new_mean)
        var = (self.demand_covs[p]*new_mean)**2
        fixed_factor_pdf = 1.0/(np.sqrt(2*np.pi)*self.demand_covs[p]*new_mean) 
        return sigr.quad(lambda d: fixed_factor_pdf * np.exp(-(d-new_mean)**2/(2*var)) * gamma * (d + (x-d) * self.salvage[p]), 0, x)[0] + x * sigr.quad(lambda d: fixed_factor_pdf * np.exp(-(d-new_mean)**2/(2*var)) * gamma, x, demand_distribution.ppf(self.max_ppf))[0] 
    
    #slope of expected revenue given production volume
    #calculate slope of expected revenue function
    def slope_function(self, p: int, x: float, gamma: float):
        x = min(self.demand_max[p], x)
        new_mean = self.demand_means[p] * (1 + (gamma-self.price_means[p+1])/self.price_means[p+1]*self.demand_elasticity[p])
        demand_distribution = st.norm(new_mean, self.demand_covs[p]*new_mean)
        var = (self.demand_covs[p]*new_mean)**2
        fixed_factor_pdf = 1.0/(np.sqrt(2*np.pi)*self.demand_covs[p]*new_mean) 
        return demand_distribution.cdf(x) * self.salvage[p] * gamma + (1-demand_distribution.cdf(x)) * gamma

    def get_heuristic_action(self):
        #update the heuristic model to new inventory and price levels
        self.update_heuristic_model()
        #solve heuristic model
        self.heuristic_model.optimize()
        #get purchasing volume 
        purchasing = self.heuristic_model.getVarByName("inv[0,1]").X
        #get issuance decisions
        issuance = {a: self.heuristic_model.getVarByName(f"iss[{a},0]").X for a in self.issuance_ages}
        return purchasing, issuance
    
    #internal getter for state
    def  _get_obs(self):
        if self.use_issuance_model:
            return {"prices": np.array([self.priceDistributions[p].cdf(self.prices[p]) for p in self.processes]), "inventory": self.inventory_position/self.maxInventory, "max_production": self.max_production/self.demand_max}
        else:
            return {"prices": np.array([self.priceDistributions[p].cdf(self.prices[p]) for p in self.processes]), "inventory": self.inventory_position/self.maxInventory} 

    def _get_state_from_obs(self, obs):
        self.inventory_position = obs["inventory"] * self.maxInventory
        self.prices = [self.priceDistributions[p].ppf(obs["prices"][p]) for p in self.processes]
        
    #reset function does nothing --> infinite horizon
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        print("FINISH EPISODE")
        self.n_steps = 0
        return self._get_obs(), {}

    #step function: if specified, use lookahead LP to approximate issuance decisions
    def step(self, action):
        # if self.use_issuance_model:
        #     return self.step_continuous_issuance_lp(action)
        # else:
        return self.step_continuous(action)
        
    def step_continuous(self, action):

        self.n_steps += 1
        if self.n_steps > self.max_horizon:
            self.n_steps = 1
       
        #get action from heuristic
        if self.simulate_heuristic:
            action = self.get_heuristic_action()
            purchasing = deepcopy(action[0])
            issuance = {i: action[1][i] for i in self.issuance_ages}
        #map neural network output to implementable action
        elif not self.allow_blending:
            #mask purchasing actions using maximum inventory level
            purchasing = ((action[0]+1)/2) * (self.maxInventory)
            issuance = {i: ((action[1+self.issuance_ages.index(i)]+1)/2) * self.inventory_position[i] for i in self.issuance_ages[:-1]}
            issuance[self.targetAges[self.nProducts-1]] = self.inventory_position[self.targetAges[self.nProducts-1]]
        else:
            if self.use_issuance_model:
                #mask purchasing actions using maximum inventory level
                purchasing = ((action[0]+1)/2) * (self.maxInventory)
                production = [((action[1+p]+1)/2) * self.max_production[p] for p in self.products]
                if self.avoid_outdating:
                    production[self.products[-1]] = max(production[self.products[-1]], min(self.demand_max[self.products[-1]],self.inventory_position[-1]))
                #implement FIFO issuance for the oldest product and LIFO issuance for the younger products based on production action
                issuance = self.get_issuance_from_production(production)
            else:
                #mask purchasing actions using maximum inventory level
                purchasing = ((action[0]+1)/2) * (self.maxInventory)
                #mask issuance actions using current inventory levels
                if self.avoid_outdating:
                    issuance = {i: ((action[1+self.issuance_ages.index(i)]+1)/2) * self.inventory_position[i] for i in self.issuance_ages[:-1]}
                    issuance[self.issuance_ages[-1]] = self.inventory_position[self.issuance_ages[-1]]
                else:
                    issuance = {i: ((action[1+i-self.issuance_ages.index(i)]+1)/2) * self.inventory_position[i] for i in self.issuance_ages}
        for i in self.ages:
            if i not in self.issuance_ages:
                issuance[i] = 0
            
        #derive production/production volumes and post-decision inventory
        production = [max(0,min(self.demand_max[p],sum(issuance[a] * self.evaporation_remains_per_age_class[a] for a in self.ageRange[p]))) for p in self.products]
        new_inventory = np.nan_to_num(np.array([max(0,self.inventory_position[a] - issuance[a]) for a in self.ages]))
        outdating = new_inventory[-1]
        
        #inventories age by one period
        self.inventory_position = np.nan_to_num(np.concatenate(([purchasing],new_inventory[:-1])))
        
        #compute and normalize reward
        purchasing_cost = purchasing*self.prices[0]	
        revenue_product = [self.revenue_interpolators[p]((production[p], self.prices[p+1])) for p in self.products]
        revenue = sum(revenue_product[p] for p in self.products)
        holding_cost = sum(self.inventory_position)*self.holdingCosts

        reward = revenue - purchasing_cost - holding_cost
        norm_reward = (reward - self.min_reward)/(self.max_reward-self.min_reward) * (self.reward_ub-self.reward_lb) + self.reward_lb  

        #sample new prices and update price pipeline
        prev_prices = deepcopy(self.prices)
        self.sample_prices()
        if self.use_issuance_model:
            self.update_max_production()
        
        #update state
        observation = self._get_obs()

        return observation, norm_reward, False, self.n_steps == self.max_horizon, {"revenue":revenue, "purchasing_cost": purchasing_cost, "holding_cost":holding_cost, "purchasing":purchasing, "production":production, "issuance":[issuance[a] for a in self.ages],  "outdating": outdating, "inventory": self.inventory_position, "prices":prev_prices}

    def step_continuous_issuance_lp(self, action):
        # start_time = time.time()
        
        self.n_steps += 1
        if self.n_steps > self.max_horizon:
            self.n_steps = 1
            #print("STARTING STATE ENV: ", self.price, " ", self.inventory_position)
        
        production = None
        starting_inventory = self.inventory_position
        #get purchasing action from heuristic
        if self.simulate_heuristic:
            action = self.get_heuristic_action()
            purchasing = deepcopy(action[0])
            issuance = {i: action[1][i-self.issuance_ages[0]] for i in self.issuance_ages}
            for i in self.ages:
                if i not in self.issuance_ages:
                    issuance[i] = 0

        #derive purchasing (and possibly production) volumes from neural network output  
        else:
            purchasing = ((action[0]+1)/2) * self.maxInventory
            purchasing = np.nan_to_num(purchasing)
            self.just_purchased = purchasing
            
            # print("TIME TO READ ACTION: ", time.time() - start_time)
            # start_time = time.time()
            #update issuance LP to new inventory level and purchasing level
            self.update_issuance_model()
            # print("TIME TO UPDATE ISSUANCE MODEL: ", time.time() - start_time)
            # start_time = time.time()
            #optmize model
            self.issuance_model.optimize()
            # print("KAPPA: ", self.issuance_model.KappaExact)
            # print("TIME TO SOLVE ISSUANCE MODEL: ", time.time() - start_time)
            # start_time = time.time()

            #get issuance decisions
            issuance = {a: self.issuance_model.getVarByName(f"iss[{a},0]").X for a in self.issuance_ages}
            for i in self.ages:
                if i not in self.issuance_ages:
                    issuance[i] = 0
                
        #derive production/production volumes and post-decision inventory
        production = [max(0,min(self.demand_max[p],sum(issuance[a] * self.evaporation_remains_per_age_class[a] for a in self.ageRange[p]))) for p in self.products]
        new_inventory = np.nan_to_num(np.array([max(0,self.inventory_position[a] - issuance[a]) for a in self.ages]))
        outdating = new_inventory[-1]

        #inventories age by one period
        self.inventory_position = np.nan_to_num(np.concatenate(([purchasing],new_inventory[:-1])))

        #compute and normalize rewards
        purchasing_cost = purchasing*self.prices[0]
        revenue_product = [self.revenue_interpolators[p]((production[p], self.prices[p+1])) for p in self.products]
        revenue = sum(revenue_product[p] for p in self.products)
        holding_cost = sum(self.inventory_position)*self.holdingCosts

        
        reward = revenue - purchasing_cost - holding_cost
        norm_reward = (reward - self.min_reward)/(self.max_reward-self.min_reward) * (self.reward_ub-self.reward_lb) + self.reward_lb  

        # print("TIME TO DERIVE REWARDS: ", time.time() - start_time)
        # start_time = time.time()

        #sample new prices and update price pipeline
        prev_prices = deepcopy(self.prices)
        self.sample_prices()
                    
        #update state        
        observation = self._get_obs()

        # print("TIME TO FINALIZE STEP: ", time.time() - start_time)
        return observation, norm_reward, False, self.n_steps == self.max_horizon, {"revenue":revenue, "purchasing_cost": purchasing_cost, "holding_cost":holding_cost, "purchasing":purchasing, "production":production, "issuance":[issuance[a] for a in self.ages],  "outdating": outdating, "inventory": self.inventory_position, "prices":prev_prices}

    def sample_prices(self):
        #sample random shocks
        if self.use_cdfs_for_regularization:
            shocks = self.cdf_buffer[self.n_steps-1]
        else:
            shocks = self.multi_norm.rvs()
        #apply shocks
        self.prices = [max(0,self.prices[p] + self.speed[p] * (self.price_means[p] - self.prices[p]) + self.vol[p] * shocks[p]) for p in self.processes]
        #update price pipeline
        if self.simulate_heuristic:
            self.update_price_pipelines()
    
    def update_price_pipelines(self):
        for p in self.processes:
            self.price_pipelines[0,p] = self.prices[p]
            for i in self.time_periods[:-1]:
                self.price_pipelines[i+1,p] = self.price_pipelines[i,p] + self.speed[p] * (self.price_means[p] - self.price_pipelines[i,p])      

    def update_max_production(self):
        for p in self.products:
            self.max_production[p] = min(self.demand_max[p], sum(self.inventory_position[a] * self.evaporation_remains_per_age_class[a] for a in self.ageRange[p]))
     
    def get_issuance_from_production(self, production): 
        #FIFO issuance for the oldest product
        issuance = {i: 0 for i in self.ages}
        volume_remaining = {p: production[p] for p in self.products}
        for i in reversed(self.ageRange[self.products[-1]]):
            if self.avoid_outdating and i == self.ages[-1]:
                issuance[i] = self.inventory_position[i]
            else:
                issuance[i] = min(self.inventory_position[i], volume_remaining[self.products[-1]]/self.evaporation_remains_per_age_class[i])
            volume_remaining[self.products[-1]] -= issuance[i] * self.evaporation_remains_per_age_class[i]
            if volume_remaining[self.products[-1]] <= 0:
                break

        #LIFO issuance for the younger products
        for p in reversed(self.products[:-1]):
            for i in self.ageRange[p]:
                issuance[i] = min(self.inventory_position[i], volume_remaining[p]/self.evaporation_remains_per_age_class[i])
                volume_remaining[p] -= issuance[i] * self.evaporation_remains_per_age_class[i]
                if volume_remaining[p] <= 0:
                    break
        return issuance

    #update LP constraints to new inventory    
    def update_inventory_constraints(self, model):  
        for a in self.ages:
            model.remove(model.getConstrByName("start"+str(a)))
            inv = model.getVarByName(f"inv[{a},0]")
            model.addLConstr(inv == self.inventory_position[a], name="start"+str(a))
    
    def update_objective_function(self, model):
        for p in self.products:
            for l in self.production_levels_lp[p]:
                for t in self.time_periods:
                    self.obj_revenue_pipeline[p][l][t] = self.revenue_interpolators[p]((l,self.price_pipelines[t,p+1]))
                    production = model.getVarByName(f"prod[{p},{l},{t}]")
                    production.Obj = self.obj_revenue_pipeline[p][l][t]
        for a in self.issuance_ages:
            for t in self.time_periods[a+1:]:
                issuance = model.getVarByName(f"iss[{a},{t}]")
                issuance.Obj = -self.holdingCosts*(a+1) - self.price_pipelines[t-a-1,0]
        model.update()

    #update issuance LP to new inventory and purchasing input
    def update_issuance_model(self):
        self.issuance_model.update()
        self.update_inventory_constraints(self.issuance_model)
        self.issuance_model.remove(self.issuance_model.getConstrByName("startp"))
        inv = self.issuance_model.getVarByName("inv[0,1]")
        self.issuance_model.addLConstr(inv == self.just_purchased, name="startp")
        self.update_objective_function(self.issuance_model)
        self.issuance_model.update()

    #update heuristic LP to new inventory and price
    def update_heuristic_model(self):
        self.heuristic_model.update()
        self.update_inventory_constraints(self.heuristic_model)
        self.update_objective_function(self.heuristic_model)
        self.heuristic_model.update()

    #create a lookahead linear program for taking issuance/production decisions given the current state and purchasing action as inputs
    def create_issuance_lp(self):
        self.issuance_model = gb.Model()
        #disable printout of Gurobi solution process
        self.issuance_model.setParam('OutputFlag', 0)
        self.issuance_model.setParam('Method', 1)
        
        #prepare indices of decision variables
        tuplelist_iss = []
        tuplelist_prod = []
        for a in self.issuance_ages:
            for t in self.time_periods:
                tuplelist_iss.append((a,t))
        for p in self.products:
            for l in self.production_levels_lp[p]:
                for t in self.time_periods:
                    tuplelist_prod.append((p,l,t))


        #define decision variables
        inv = self.issuance_model.addVars(self.numAges, self.n_time_steps, name="inv")
        iss = self.issuance_model.addVars(tuplelist_iss, name="iss")
        out = self.issuance_model.addVars(self.n_time_steps, name="out")
        prod = self.issuance_model.addVars(tuplelist_prod, lb=[0.0 for i in range(len(tuplelist_prod))], ub=[1.0 for i in range(len(tuplelist_prod))], name="prod")
      
        print(self.n_time_steps)
        print(self.numAges)
        print(self.nProducts)
        print(len(tuplelist_prod))
        print(len(tuplelist_iss))

        #starting inventory
        for a in self.ages:
            self.issuance_model.addLConstr(inv[a,0] == self.inventory_position[a], name="start"+str(a))

        #purchasing volume from actor network
        self.issuance_model.addLConstr(inv[0,1] == self.just_purchased, name="startp")
        
        #maximum purchasing/inventory capacity restrictions
        for t in range(1,self.n_time_steps):
            self.issuance_model.addLConstr(inv[0,t] <= (self.maxInventory))

        for t in self.time_periods:
            #outdating
            self.issuance_model.addLConstr(out[t] >= inv[self.numAges-1,t] - iss[self.numAges-1,t])
            for p in self.products:
                self.issuance_model.addLConstr(gb.quicksum(prod[p,l,t] for l in self.production_levels_lp[p]) <= 1.0)
                #relate production to issuance volumes
                self.issuance_model.addLConstr(gb.quicksum(prod[p,l,t] * l for l in self.production_levels_lp[p]) <= gb.quicksum(iss[a,t] * self.evaporation_remains_per_age_class[a] for a in self.ageRange[p]))
            for a in self.ages:
                #inventory balance
                if a in self.issuance_ages:
                    self.issuance_model.addLConstr(iss[a,t] <= inv[a,t])
                if t>0 and a>0:
                    if a in self.issuance_ages[1:]:
                        self.issuance_model.addLConstr(inv[a,t] == (inv[a-1,t-1] - iss[a-1,t-1]))
                    else:
                        self.issuance_model.addLConstr(inv[a,t] == inv[a-1,t-1])
                
        #set objective value       
        obj = gb.quicksum(prod[p,l,t] * self.obj_revenue_pipeline[p][l][t] for p in self.products for l in self.production_levels_lp[p] for t in self.time_periods) \
            + gb.quicksum(iss[a,t]*sum((-self.holdingCosts) for _ in range(a+1)) for a in self.issuance_ages for t in self.time_periods) \
            - gb.quicksum(iss[a,t]*self.price_pipelines[t-a-1,0] for a in self.issuance_ages for t in self.time_periods[a+1:]) \
            - gb.quicksum(iss[a,t]*self.price_means[0] for a in self.issuance_ages for t in self.time_periods[:a+1]) \
            + gb.quicksum(out[t]*(-self.holdingCosts*self.numAges - self.price_means[0]) for t in self.time_periods)
        
        self.issuance_model.setObjective(obj, GRB.MAXIMIZE)

    #create a lookahead linear program for taking all decisions
    def create_heuristic_lp(self):
        self.heuristic_model = gb.Model()
        #disable printout of Gurobi solution process
        self.heuristic_model.setParam('OutputFlag', 0)
        self.heuristic_model.setParam('Method', 1)
        
        #prepare indices of decision variables
        tuplelist_iss = []
        tuplelist_prod = []
        
        for a in self.issuance_ages:
            for t in self.time_periods:
                tuplelist_iss.append((a,t))
        for p in self.products:
            for l in self.production_levels_lp[p]:
                for t in self.time_periods:
                    tuplelist_prod.append((p,l,t))

        #define decision variables
        inv = self.heuristic_model.addVars(self.numAges, self.n_time_steps, name="inv")
        iss = self.heuristic_model.addVars(tuplelist_iss, name="iss")
        out = self.heuristic_model.addVars(self.n_time_steps, name="out")
        prod = self.heuristic_model.addVars(tuplelist_prod, lb=[0.0 for i in range(len(tuplelist_prod))], ub=[1.0 for i in range(len(tuplelist_prod))], name="prod")

        self.obj_revenue_pipeline = {p: {l: {t: self.revenue_interpolators[p]((l,self.price_pipelines[t,p+1])) for t in self.time_periods} for l in self.production_levels_lp[p]} for p in self.products}

        #starting inventory
        for a in self.ages:
            self.heuristic_model.addLConstr(inv[a,0] == self.inventory_position[a], name="start"+str(a))
        
        #maximum purchasing/inventory capacity restrictions
        for t in range(1,self.n_time_steps):
            self.heuristic_model.addLConstr(inv[0,t] <= (self.maxInventory))

        for t in self.time_periods:
            #outdating
            if self.allow_blending:
                if self.numAges-1 in self.issuance_ages:
                    self.heuristic_model.addLConstr(out[t] >= inv[self.numAges-1,t] - iss[self.numAges-1,t])
                else:
                    self.heuristic_model.addLConstr(out[t] >= inv[self.numAges-1,t])
            else:
                self.heuristic_model.addLConstr(out[t] >= 0)
            for p in self.products:
                self.heuristic_model.addLConstr(gb.quicksum(prod[p,l,t] for l in self.production_levels_lp[p]) <= 1.0)
                #relate production to issuance volumes
                self.heuristic_model.addLConstr(gb.quicksum(prod[p,l,t] * l for l in self.production_levels_lp[p]) <= gb.quicksum(iss[a,t] * self.evaporation_remains_per_age_class[a] for a in self.ageRange[p]))
            for a in self.ages:
                #inventory balance
                if a in self.issuance_ages:
                    self.heuristic_model.addLConstr(iss[a,t] <= inv[a,t])
                if t>0 and a>0:
                    if a-1 in self.issuance_ages[1:]:
                        self.heuristic_model.addLConstr(inv[a,t] == (inv[a-1,t-1] - iss[a-1,t-1]))
                    else:
                        self.heuristic_model.addLConstr(inv[a,t] == inv[a-1,t-1])
                
        #set objective value       
        obj = gb.quicksum(prod[p,l,t] * self.obj_revenue_pipeline[p][l][t] for p in self.products for l in self.production_levels_lp[p] for t in self.time_periods) \
            + gb.quicksum(iss[a,t]*sum((-self.holdingCosts) for _ in range(a+1)) for a in self.issuance_ages for t in self.time_periods) \
            - gb.quicksum(iss[a,t]*self.price_pipelines[t-a-1,0] for a in self.issuance_ages for t in self.time_periods[a+1:]) \
            - gb.quicksum(iss[a,t]*self.price_means[0] for a in self.issuance_ages for t in self.time_periods[:a+1]) \
            + gb.quicksum(out[t]*(-self.holdingCosts*self.numAges - self.price_means[0]) for t in self.time_periods)
    
        self.heuristic_model.setObjective(obj, GRB.MAXIMIZE)

    
    def env_creator(env_config):
        return CheeseEnv(env_config)
    
    #one step simulation given policy
    def simulate_one_step(self, policy=None):
        if policy == None:
            action = self.action_space.sample()
        else:
            action = policy.compute_single_action(self._get_obs(), explore=False)
        return self.step(action) #, explore=False
    
    def simulate_starting_state_eval(self, nsteps=1, plot=False):
        heuristic_setting = deepcopy(self.simulate_heuristic)
        self.simulate_heuristic = True
        for _ in range(nsteps):
            next_state, reward, truncated, done, info = self.simulate_one_step(None)
        self.simulate_heuristic = heuristic_setting
        return self.prices, self.inventory_position
        
    #simulator function for evaluating policy
    def simulate_n_steps(self, nsteps=1, policy=None, plot=False, warm_up=100):
        
        self.use_cdfs_for_regularization = False

        #warm-up
        warm_up_steps = warm_up
        while(warm_up_steps > 0):
            if self.render_mode == 'human':
                action = np.array([int(i) for i in input("write down action in the format: <purchasing> <production product 1> ... <production product |W|>").split(" ")[:self.nProducts+1]])
                print("action taken:", action)
                next_state, reward, truncated, done, info = self.step(action) 
            else:
                next_state, reward, truncated, done, info = self.simulate_one_step(policy)
            warm_up_steps -= 1 

        self.n_steps = 0
        rewards = np.array([])
        purchasing = np.array([])
        revenues = np.array([])
        purchasing_costs = np.array([])
        holding_costs = np.array([])
        prices = np.array([])

        production = {p: np.array([]) for p in self.products}
        issuance = {a: np.array([]) for a in self.ages}
        inventories = {a: np.array([]) for a in self.ages}
        outdating = np.array([])
        iterations = 0
        while(nsteps > 0):
            #print("inventory position: ", self.inventory_position)
            if self.render_mode == 'human':
                action = np.array([int(i) for i in input("write down action in the format: <purchasing> <production product 1> ... <production product |W|>").split(" ")[:self.nProducts+1]])
                print("action taken:", action)
                next_state, reward, truncated, done, info = self.step(action) 
            else:
                next_state, reward, truncated, done, info = self.simulate_one_step(policy)
            
            #print(info)
            rewards = np.append(rewards,reward)
            purchasing = np.append(purchasing, info["purchasing"])
            revenues = np.append(revenues, info["revenue"])
            purchasing_costs = np.append(purchasing_costs, info["purchasing_cost"])
            holding_costs = np.append(holding_costs, info["holding_cost"])
            prices = np.append(prices, info["prices"])
            for p in self.products:
                production[p] = np.append(production[p], info["production"][p])
            for a in self.ages:
                issuance[a] = np.append(issuance[a], info["issuance"][a])
                inventories[a] = np.append(inventories[a], info["inventory"][a])
            outdating = np.append(outdating, info["outdating"])

            iterations+=1
            if iterations % 500 == 0:
                denormalized_rewards = [((r-self.reward_lb)/(self.reward_ub - self.reward_lb)) * (self.max_reward - self.min_reward) + self.min_reward for r in rewards]
                print(f"{iterations} STEPS SIMULATED")
                print("average reward: ", np.mean(rewards))
                print("average reward w/o normalization: ", np.mean(denormalized_rewards))
                print("90 percent confidence average reward: ", st.norm.interval(0.9, loc=np.mean(denormalized_rewards), scale=np.std(denormalized_rewards)/np.sqrt(len(denormalized_rewards))))
                print("average purchasing: ", np.mean(purchasing))
                print("purchasing variance: ", np.var(purchasing))
                print("average production: ", [np.mean(production[p]) for p in self.products])
                print("average inventory structure: ", [np.mean(inventories[a]) for a in self.ages])
                print("average outdating: ", np.mean(outdating))
                print("average revenue: ", np.mean(revenues))
                print("average purchasing costs: ", np.mean(purchasing_costs))
                print("average holding costs: ", np.mean(holding_costs))
                print("average price: ", np.mean(prices))
                print("price std: ", np.std(prices))
            nsteps-=1

        if plot:
            fig = plt.hist(purchasing)
            plt.show()
            fig = plt.plot(prices)
            plt.show()
        return np.mean(rewards)
    
    def evaluate_vs_heuristic(self, seeds, policy=None, starting_prices=None, starting_inv=None, replications=1000, do_heuristic=True):
        if policy==None:
            raise ValueError("Policy must be provided for evaluation")
        #create cdfs using seeds (matrix size depends on replications)
        cdfs = {}

        print("STARTING PRICES: ", starting_prices)
        print("STARTING INVENTORY: ", starting_inv)

        for seed in seeds:
            np.random.seed(seed)
            cdfs[seed] = {i: np.array(self.multi_norm.rvs()) for i in range(replications)}
        #evaluate policy
        rewards_DRL = np.array([])
        for seed in seeds:
            #np.random.seed(seed)
            print("EVALUATING POLICY WITH SEED: ", seed)
            rewards_DRL = np.append(rewards_DRL, [self.simulate_w_cdfs(cdfs=cdfs[seed], policy=policy, initial_prices=starting_prices, initial_inventory=starting_inv)[1]])

        print("DRL average reward: ", np.mean(rewards_DRL))

        if do_heuristic:
            #evaluate heuristic
            rewards_heuristic = np.array([])
            for seed in seeds:
                #np.random.seed(seed)
                print("EVALUATING HEURISTIC WITH SEED: ", seed)
                rewards_heuristic = np.append(rewards_heuristic, [self.simulate_w_cdfs(cdfs=cdfs[seed], policy=None, initial_prices=starting_prices, initial_inventory=starting_inv)[1]])

            print("HEURISTIC average reward: ", np.mean(rewards_heuristic))
        else:
            rewards_heuristic = None

        return rewards_DRL, rewards_heuristic

    def get_heuristic_average(self, final_interval_width=0.4):
        
        self.use_cdfs_for_regularization = False
        
        simulate_heuristic_setting = self.simulate_heuristic
        self.simulate_heuristic = True
        interval_width = final_interval_width + 1
        rewards = np.array([])
        while interval_width > final_interval_width:
            for _ in range(500):
                next_state, reward, truncated, done, info = self.step(self.action_space.sample()) 
                rewards = np.append(rewards,reward)
            new_interval = st.norm.interval(0.9,loc=np.mean(rewards),scale=np.std(rewards)/np.sqrt(len(rewards)))
            interval_width = new_interval[1]-new_interval[0]
            print(f"average reward & interval width after {len(rewards)} steps: ", np.mean(rewards), " ", interval_width)

        avg_heuristic_reward = np.mean(rewards)
        print("heuristic average reward: ", avg_heuristic_reward)
        self.simulate_heuristic = simulate_heuristic_setting
        self.reset()

        return avg_heuristic_reward
    
    def simulate_data_for_regression(self, policy=None, data_size=200_000, warm_up_length = 1000):
        if policy is None:
            self.simulate_heuristic = True
        for _ in range(warm_up_length):
            self.simulate_one_step(policy)

        self.use_cdfs_for_regularization = False
        
        state = self._get_obs()
        feature_matrix = np.concatenate((state["prices"],state["inventory"]))
        reward_array = np.array([])
        state,reward,_,_,info = self.simulate_one_step(policy)
        response_matrix = np.concatenate(([info["purchasing"]],info["production"],info["issuance"]))
        reward_array = np.append(reward_array, reward)
        for i in range(data_size):
            if (i+1)%1000 == 0:
                print(f"SIMULATING DATA FOR REGRESSION... {i} STEPS COMPLETED")
            feature_matrix = np.vstack((feature_matrix, np.concatenate((state["prices"],state["inventory"]))))
            state,reward,_,_,info = self.simulate_one_step(policy)
            reward_array = np.append(reward_array, reward)
            response_matrix = np.vstack((response_matrix,np.concatenate(([info["purchasing"]],info["production"],info["issuance"]))))

        return feature_matrix, response_matrix, reward_array

    def simulate_w_cdfs(self, cdfs=None, policy=None, initial_prices = None, initial_inventory=None, warm_up_length=1000):
        
        if cdfs is None:
            self.cdf_buffer = {i: self.multi_norm.rvs() for i in self.max_horizon}
        else:
            self.cdf_buffer = cdfs
        cdf_length = len(cdfs)
        print("CDF LENGTH: ", cdf_length)
        start_state = self._get_obs()
        self.n_steps = 0

        heuristic_setting = deepcopy(self.simulate_heuristic)

        if policy is None:
            self.simulate_heuristic = True
        else:
            self.simulate_heuristic = False
        if initial_prices is not None:
            self.prices = initial_prices
        if initial_inventory is not None:
            self.inventory_position = initial_inventory
        
        self.use_cdfs_for_regularization = True
        
        for _ in range(warm_up_length):
            next_state, reward, truncated, done, info = self.simulate_one_step(policy)
        
        print("WARM-UP FINISHED")

        state = self._get_obs()
        rewards = np.array([])
        feature_matrix = np.concatenate((state["prices"],state["inventory"]))
        state,reward,_,_,info = self.simulate_one_step(policy)
        response_matrix = np.concatenate(([info["purchasing"]],info["production"],info["issuance"]))
        rewards = np.append(rewards, reward)
        for _ in range(cdf_length-warm_up_length-1):
            feature_matrix = np.vstack((feature_matrix, np.concatenate((state["prices"],state["inventory"]))))
            state, reward, _, _, info = self.simulate_one_step(policy)
            rewards = np.append(rewards, reward)
            response_matrix = np.vstack((response_matrix,np.concatenate(([info["purchasing"]],info["production"],info["issuance"]))))
            if (len(rewards)+1) % 1000 == 0:
                print(f"SIMULATING WITH CDFS... {len(rewards)+1} STEPS COMPLETED")
        
        self.use_cdfs_for_regularization = False
        self.n_steps = 0
        self.reset()
        self._get_state_from_obs(start_state)
        mean_reward = np.mean(rewards)

        self.simulate_heuristic = heuristic_setting

        return feature_matrix, response_matrix, rewards


#--------------------------------------#
def upper_bound(env:CheeseEnv, discr_step=0.1):
    if abs(discr_step / env.production_step_size - round(discr_step / env.production_step_size)) > 1e-9:
        raise ValueError(f"discr_step {discr_step} is not a multiple of env.production_step_size {env.production_step_size}")

    #create outer approximation of concave function for expected reward
    for p in env.products:
        env.expected_revenue[p][round(env.demand_max[p]+discr_step, ndigits=2)] = env.expected_revenue[p][env.demand_max[p]]
        env.slope[p][round(env.demand_max[p]+discr_step, ndigits=2)] = {g: 0.0 for g in env.price_levels[p+1]}
    tangent_points = {p: [round(i,ndigits=2) for i in np.arange(env.production_levels[p][0],round(env.production_levels[p][-1]+discr_step, ndigits=2),discr_step)] for p in env.products}
    print("TANGENT POINTS: ", tangent_points[0])

    break_points = {p: {g: {} for g in env.price_levels[p+1]} for p in env.products}
    for w in env.products:
        for g in env.price_levels[w+1]:
            break_points[w][g][0] = 0.0
            for l in range(len(tangent_points[w])-1):
                if env.slope[w][tangent_points[w][l]][g] != env.slope[w][tangent_points[w][l+1]][g] and (env.expected_revenue[w][tangent_points[w][l+1]][g] - env.expected_revenue[w][tangent_points[w][l]][g] + env.slope[w][tangent_points[w][l]][g]*tangent_points[w][l] - env.slope[w][tangent_points[w][l+1]][g]*tangent_points[w][l+1])/(env.slope[w][tangent_points[w][l]][g]-env.slope[w][tangent_points[w][l+1]][g]) < tangent_points[w][l+1] and (env.expected_revenue[w][tangent_points[w][l+1]][g] - env.expected_revenue[w][tangent_points[w][l]][g] + env.slope[w][tangent_points[w][l]][g]*tangent_points[w][l] - env.slope[w][tangent_points[w][l+1]][g]*tangent_points[w][l+1])/(env.slope[w][tangent_points[w][l]][g]-env.slope[w][tangent_points[w][l+1]][g]) > tangent_points[w][l]:
                    break_points[w][g][l+1] = (env.expected_revenue[w][tangent_points[w][l+1]][g] - env.expected_revenue[w][tangent_points[w][l]][g] + env.slope[w][tangent_points[w][l]][g]*tangent_points[w][l] - env.slope[w][tangent_points[w][l+1]][g]*tangent_points[w][l+1])/(env.slope[w][tangent_points[w][l]][g]-env.slope[w][tangent_points[w][l+1]][g])
                    
    expected_revenue_break_points = {p: {g: {} for g in env.price_levels[p+1]} for p in env.products} 
    for p in env.products:
        for g in env.price_levels[p+1]:
            expected_revenue_break_points[p][g][break_points[p][g][0]] = env.expected_revenue[p][tangent_points[p][0]][g]
            for l in break_points[p][g]:
                if l > 0:
                    expected_revenue_break_points[p][g][break_points[p][g][l]] = env.expected_revenue[p][tangent_points[p][l-1]][g] + env.slope[p][tangent_points[p][l-1]][g] * (break_points[p][g][l] - tangent_points[p][l-1])
            expected_revenue_break_points[p][g][max(break_points[p][g].values())] = env.expected_revenue[p][env.demand_max[p]][g]
    
    #discretize price levels
    price_levels_discretized = {p: env.price_levels[p] for p in env.processes}
    price_probabilities_discretized = {p: {} for p in env.processes}
    print("PRICE LEVELS DISCRETIZED: ", price_levels_discretized)
    for p in env.processes[1:]:
        price_probabilities_discretized[p] = {i: env.priceDistributions[p].cdf(i+discr_step) - env.priceDistributions[p].cdf(i) for i in price_levels_discretized[p]}
        price_probabilities_discretized[p][price_levels_discretized[p][-1]] = 1 - env.priceDistributions[p].cdf(price_levels_discretized[p][-1])  
        price_probabilities_discretized[p][price_levels_discretized[p][0]] = env.priceDistributions[p].cdf(price_levels_discretized[p][1])

    price_probabilities_discretized[0] = {i: env.priceDistributions[0].cdf(i) - env.priceDistributions[0].cdf(i-discr_step) for i in price_levels_discretized[0]}
    price_probabilities_discretized[0][price_levels_discretized[0][-1]] = 1 - env.priceDistributions[0].cdf(price_levels_discretized[0][-2])
    price_probabilities_discretized[0][price_levels_discretized[0][0]] = env.priceDistributions[0].cdf(price_levels_discretized[0][0])

    for p in env.processes:
        print(p)
        print(sum(price_probabilities_discretized[p].values()))
    #create upper bound model 
    upper_bound_model = gb.Model()
    #disable printout of Gurobi solution process
    upper_bound_model.setParam('OutputFlag', 0)

    #define decision variables
    ff_indices = [(p,g,b) for p in env.products for g in env.price_levels[p+1] for b in break_points[p][g].values()]
    print("FF INDICES: ", [ff_indices[i] for i in range(len(ff_indices)) if ff_indices[i][0] == 1 and ff_indices[i][1] == 500.0])
    ff = upper_bound_model.addVars(ff_indices, name="ff", lb=0.0, ub=1.0)
    purchasing = upper_bound_model.addVars(price_levels_discretized[0], name="purchasing", lb=0.0, ub=env.maxInventory)
    inv = upper_bound_model.addVars(env.numAges, name="inv", lb=0.0, ub=env.maxInventory)
    iss = upper_bound_model.addVars(env.nProducts, env.numAges, name="iss", lb=0.0, ub=env.maxInventory)
    out = upper_bound_model.addVar(name="out", lb=0.0, ub=env.maxInventory)

    print("Upper Bound Model: Variable Creation Done!")

    #OBJECTIVE FUNCTION
    obj = gb.quicksum(ff[p,g,l] * expected_revenue_break_points[p][g][l] * price_probabilities_discretized[p+1][g] for p in env.products for g in price_levels_discretized[p+1] for l in break_points[p][g].values())  - gb.quicksum(purchasing[p]*p*price_probabilities_discretized[0][p] for p in price_levels_discretized[0]) - gb.quicksum(inv[a]*env.holdingCosts for a in env.ages)
    upper_bound_model.setObjective(obj, GRB.MAXIMIZE)

    print("Upper Bound Model: Objective Added!")

    #CONSTRAINTS
    #inventory balance
    for a in env.ages[1:]:
        upper_bound_model.addLConstr(inv[a] == (inv[a-1]-gb.quicksum(iss[p,a-1] for p in env.products)))
    #inventory in first age class is determined by purchasing behavior
    upper_bound_model.addLConstr(inv[0] == gb.quicksum(purchasing[l] * price_probabilities_discretized[0][l] for l in price_levels_discretized[0]))
    #outdating is determined by issuance from last age class
    upper_bound_model.addLConstr(out == inv[env.numAges-1] - gb.quicksum(iss[p,env.numAges-1] for p in env.products))
    
    print("Upper Bound Model: Inventory Constraints Added!")

    #production & issuance
    for p in env.products:
        for g in env.price_levels[p+1]:
            #production proportions add up to 1
            upper_bound_model.addLConstr(gb.quicksum(ff[p,g,l] for l in break_points[p][g].values()) == 1)
        #production amounts cannot exceed issuance amounts per product
        upper_bound_model.addLConstr(gb.quicksum(ff[p,g,l]*l*price_probabilities_discretized[p+1][g] for g in price_levels_discretized[p+1] for l in break_points[p][g].values()) <= gb.quicksum(iss[p,a]*env.evaporation_remains_per_age_class[a] for a in env.ages))
        #target ages need to be respected
        upper_bound_model.addLConstr(gb.quicksum(iss[p,a] * a * env.evaporation_remains_per_age_class[a] for a in env.ages) >= env.targetAges[p] * gb.quicksum(iss[p,a] * env.evaporation_remains_per_age_class[a] for a in env.ages))
        #tie issuance to products' age ranges
        upper_bound_model.addLConstr(gb.quicksum(iss[p,a] for a in env.ages if a not in env.ageRange[p]) <= 0)


    print("Upper Bound Model: Product Constraints Added!")

    upper_bound_model.optimize()

    opt_inv = [upper_bound_model.getVarByName(f"inv[{i}]").X for i in env.ages]
    print("OPTIMAL INVENTORY STRUCTURE: ", opt_inv)
    opt_iss = [sum(upper_bound_model.getVarByName(f"iss[{p},{a}]").X for p in env.products) for a in env.ages]
    print("OPTIMAL ISSUANCE VOLUMES: ", opt_iss)
    opt_prod = [sum(upper_bound_model.getVarByName(f"iss[{p},{a}]").X * env.evaporation_remains_per_age_class[a] for a in env.ages) for p in env.products]
    print("OPTIMAL PRODUCTION VOLUMES: ", opt_prod)
    opt_purchasing = sum(upper_bound_model.getVarByName(f"purchasing[{l}]").X * price_probabilities_discretized[0][l] for l in price_levels_discretized[0])
    print("OPTIMAL PURCHASING: ", opt_purchasing)

    opt_cost_p = sum(upper_bound_model.getVarByName(f"purchasing[{l}]").X * l * price_probabilities_discretized[0][l] for l in price_levels_discretized[0])
    print("OPTIMAL PURCHASING COST: ", opt_cost_p)
    opt_rev = sum(upper_bound_model.getVarByName(f"ff[{p},{g},{l}]").X * expected_revenue_break_points[p][g][l] * price_probabilities_discretized[p+1][g] for p in env.products for g in price_levels_discretized[p+1] for l in break_points[p][g].values())
    print("OPTIMAL EXPECTED REVENUES: ", opt_rev)
    opt_cost_h = sum(upper_bound_model.getVarByName(f"inv[{a}]").X * env.holdingCosts for a in env.ages)
    print("OPTIMAL HOLDING COSTS: ", opt_cost_h)


    res_dict = {"max_reward": upper_bound_model.ObjVal, "inventory_position": opt_inv, "issuance": opt_iss, "production": opt_prod, "purchasing": opt_purchasing, "purchasing_cost": opt_cost_p, "holding_cost": opt_cost_h, "expected_revenues": opt_rev}

    return res_dict