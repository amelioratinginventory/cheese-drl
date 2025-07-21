#--------------------------------------#
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import ray
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import string
import pprint
from pathlib import Path
import time

import tensorflow as tf
import argparse
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from numpy import inf
import scipy.integrate as sigr
import ray.rllib.algorithms.apo as apo
import ray.rllib.algorithms.ppo as ppo 
from ray import air, tune
from ray.tune.registry import register_env, register_trainable
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import ResultGrid

from ray.rllib.utils import try_import_tf
from ray.rllib.utils.schedules.polynomial_schedule import PolynomialSchedule
from ray.tune.logger import pretty_print
from ray.air.config import CheckpointConfig
from MultiAgentCheeseEnvironment import CheeseEnv as env
from MultiAgentCheeseEnvironment import upper_bound as ub_function
from ray.rllib.algorithms.apo.apo_tf_policy import APOTF2Policy
from ray.rllib.utils.checkpoints import get_checkpoint_info
from stochastic.processes.diffusion import vasicek as vsk

import wandb
import openpyxl
import os

#----------------------------------------#
def data_from_xlsx_named_range(xlsx_file, range_name):
   cell_name = xlsx_file.defined_names[range_name].value
   ws, reg = cell_name.split('!')
   if ws.startswith("'") and ws.endswith("'"):
      ws = ws[1:-1]
   region = xlsx_file[ws][reg]
   data = [cell.value for row in region for cell in row]
   return data

#------------------------------------------#

def main():
   
   #disable GPU for training
   #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

   problem_id = "cheese_110_production"
   pid_clean=problem_id[0:10]
   storage_path = f"{os.getcwd()}/cheese/{problem_id}/training_results/"

   #ray training settings
   horizon = 5500
   truncation_length = 500
   num_workers = 100
   minibatch_size = 0.1
   recovery_checkpoint = None#f"{os.getcwd()}/cheese/{problem_id}/training_results/full_blending_full_nn_none_lp_100w/checkpoint_000415"
   start_iteration = 0 if recovery_checkpoint is None else int(recovery_checkpoint[-3:])-1
   
   if recovery_checkpoint is not None:
      run_id = recovery_checkpoint.split("training_results/")[1].split("/")[0]
      checkpoint_dir = recovery_checkpoint.split("checkpoint_")[0]
   use_common_random_numbers = False
   training_iterations = 500

   #load problem configuration given problem_id
   path_config = Path(f"{os.getcwd()}\\cheese\\{pid_clean}\\config.json") 
   if path_config.is_file():
      with open(path_config, 'r') as f:
         problem_config = json.load(f)
      numAges = problem_config["numAges"]
      nProducts = problem_config["nProducts"]
      targetAges = problem_config["targetAges"]
      ageRange = problem_config["ageRange"]
      maxInventory = problem_config["maxInventory"]
      evaporation = problem_config["evaporation"]
      demand_elasticity = problem_config["demand_elasticity"]
      demand_means = problem_config["demand_means"]
      demand_covs = problem_config["demand_covs"]
      salvage = problem_config["salvage"]
      price_means = problem_config["price_means"]
      speed = problem_config["speed"]
      vol = problem_config["vol"]
      price_covs = problem_config["price_covs"]
      correlation_matrix = problem_config["correlation_matrix"]
      holding_cost = problem_config["holdingCosts"]
      production_step_size = problem_config["production_step_size"]
      demand_max = problem_config["demand_max"]
      min_ppf = problem_config["min_ppf"]
      max_ppf = problem_config["max_ppf"]
      expected_revenue = {p: {float(l): {float(g): problem_config["expected_revenue"][str(p)][str(l)][str(g)] for g in problem_config["expected_revenue"][str(p)][str(l)]} for l in problem_config["expected_revenue"][str(p)]} for p in range(nProducts)}
      #slope = {p: {float(l): problem_config["slope"][str(p)][str(l)] for l in problem_config["slope"][str(p)]} for p in range(nProducts)}
      upper_bound = problem_config["upper_bound"] if "upper_bound" in problem_config else None
      #set problem configuration
   else:   
      # problem size parameters
      numAges = 10
      nProducts = 2
      targetAges = [3,7]
      ageRange = [[i for i in range(targetAges[p],targetAges[p+1])] for p in range(nProducts-1)]
      ageRange.append([i for i in range(targetAges[-1],numAges)])
      # ageRange = None
      maxInventory = 30
      
      #set evaporation to 4 % in the first four periods, 1.5% in the second four periods and 1% in the last five periods
      evaporation = [0.04 for _ in range(4)] + [0.015 for _ in range(4)] + [0.01 for _ in range(2)]
     
      #demand distribution parameters
      demand_elasticity = [-0.27, -0.52]
      demand_means = [10.0, 7.0]
      demand_covs = [0.15, 0.15]
      demand_distribution_base = [st.norm(demand_means[i], demand_covs[i]*demand_means[i]) for i in range(nProducts)]
      salvage = [0.3,0.3]
      
      #price distribution parameters
      price_means = [170,250,350]
      if problem_id[9] == "0":
         speed = [0.05,0.05,0.05]
      elif problem_id[9] == "1":
         speed = [0.1,0.1,0.1]
      elif problem_id[9] == "2":
         speed = [0.25,0.25,0.25]
      elif problem_id[9] == "3":
         speed = [1.0,1.0,1.0] 
      
      vol_multipliers = {"0": [1,1,1], "1":[2,1,1], "2":[1,2,2], "3":[2,2,2]}
      
      if problem_id[9] == "0":
         vol = [np.sqrt(975*vol_multipliers[problem_id[7]][i]/19.0) for i in range(nProducts+1)]
      elif problem_id[9] == "1":
         vol = [np.sqrt(1900*vol_multipliers[problem_id[7]][i]/19.0) for i in range(nProducts+1)]
      elif problem_id[9] == "2":
         vol = [np.sqrt(4375*vol_multipliers[problem_id[7]][i]/19.0) for i in range(nProducts+1)]
      elif problem_id[9] == "3":
         vol = [np.sqrt(10000*vol_multipliers[problem_id[7]][i]/19.0) for i in range(nProducts+1)]
      
      price_covs = [np.sqrt((vol[i]**2)/(1-(1-speed[i])**2))/price_means[i] for i in range(nProducts+1)]
      
      if problem_id[8] == "0":
         corr_purchase_sales = 0.7
      elif problem_id[8] == "1":
         corr_purchase_sales = 0.3
      elif problem_id[8] == "2":
         corr_purchase_sales = 0.9
      
      correlation_matrix = [[1.0,corr_purchase_sales,corr_purchase_sales],[corr_purchase_sales,1.0,0.95],[corr_purchase_sales,0.95,1.0]]
      
      
      min_ppf = 1e-12
      max_ppf = 1-1e-12
      demand_max = [demand_distribution_base[p].ppf(max_ppf) for p in range(nProducts)]
      production_step_size = 0.1
      upper_bound = None
      
      holding_cost = 2.5 

   #uncertainty distribution parameters
   priceDistributions = [st.norm(price_means[i], price_covs[i]*price_means[i]) for i in range(nProducts+1)]
   multi_norm = st.multivariate_normal(mean=[0,0,0], cov=correlation_matrix)
   
   #preprocessing: expected revenue calculation
   production_levels = {p: [round(i,2) for i in np.arange(0,demand_max[p]+production_step_size,production_step_size)] for p in range(nProducts)}

   price_step_size = 0.1
   price_levels = {p: [round(i,2) for i in np.arange(max(0,priceDistributions[p+1].ppf(min_ppf)),priceDistributions[p+1].ppf(max_ppf),price_step_size)] for p in range(nProducts)}
  
   production_step_size_lp = 0.1

   print("PRICE MIN PPF: ", [priceDistributions[p+1].ppf(min_ppf) for p in range(nProducts)])


   #get expected revenue for each product, production level and current sales price
   def expected_revenue_function(p: int, x: float, gamma: float):
      x = min(demand_max[p], x)
      new_mean = demand_means[p] * (1 + (gamma-price_means[p+1])/price_means[p+1]*demand_elasticity[p])
      demand_distribution = st.norm(new_mean, demand_covs[p]*new_mean)
      var = (demand_covs[p]*new_mean)**2
      fixed_factor_pdf = 1.0/(np.sqrt(2*np.pi)*demand_covs[p]*new_mean) 
      return sigr.quad(lambda d: fixed_factor_pdf * np.exp(-(d-new_mean)**2/(2*var)) * gamma * (d + (x-d) * salvage[p]), 0, x)[0] + x * sigr.quad(lambda d: fixed_factor_pdf * np.exp(-(d-new_mean)**2/(2*var)) * gamma, x, demand_distribution.ppf(max_ppf))[0] 
    
   #calculate slope of expected revenue function
   def slope_function(p: int, x: float, gamma: float):
      x = min(demand_max[p], x)
      new_mean = demand_means[p] * (1 + (gamma-price_means[p+1])/price_means[p+1]*demand_elasticity[p])
      demand_distribution = st.norm(new_mean, demand_covs[p]*new_mean)
      var = (demand_covs[p]*new_mean)**2
      fixed_factor_pdf = 1.0/(np.sqrt(2*np.pi)*demand_covs[p]*new_mean) 
      return demand_distribution.cdf(x) * salvage[p] * gamma + (1-demand_distribution.cdf(x)) * gamma
         
   if True: #not path_config.is_file():
      exp_rev_path = Path(f"{os.getcwd()}\\cheese\\{pid_clean}\\expected_revenue.json") 
      if not exp_rev_path.is_file():
         expected_revenue = {p: {l: {g: 0 for g in price_levels[p]} for l in production_levels[p]} for p in range(nProducts)}
         slope = {p: {l: {g: 0 for g in price_levels[p]} for l in production_levels[p]} for p in range(nProducts)}
         for p in range(nProducts):
            for l in production_levels[p]:
               for g in price_levels[p]:
                  expected_revenue[p][l][g] = expected_revenue_function(p,l,g)
                  slope[p][l][g] = slope_function(p,l,g)
               print(f"PRODUCT: {p}, LEVEL: {l}")#, EXP_REV: {expected_revenue[p][l]}, SLOPE: {slope[p][l]}")
         with open(exp_rev_path, 'w') as f:
            json.dump({"expected_revenue": expected_revenue, "slope":slope}, f)  
      else:
         with open(exp_rev_path, 'r') as f:
            res = json.load(f)
         expected_revenue = {p: {l: {g: res["expected_revenue"][str(p)][str(l)][str(g)] for g in price_levels[p]} for l in production_levels[p]} for p in range(nProducts)}
         slope = {p: {l: {g: res["slope"][str(p)][str(l)][str(g)] for g in price_levels[p]} for l in production_levels[p]} for p in range(nProducts)}

   #determine use of linear program for issuance decisions
   use_issuance_model = False

       # define obs, action space and polices to get a multiagent env
    # create purchase and issuance agents
   
   # MULTI-AGENT WITHOUT GROUPING:
   if problem_id[11:]=="1perAgent":
      multi_agent_setting = "multi_all"
      products = [p for p in range(nProducts)]
      issuance_ages = [i for i in range(numAges) if any(i in ageRange[p] for p in products)]
      print("ISSUANCE AGES: ", issuance_ages)
      num_agents = 1 + len(issuance_ages) - 1  # purchase agent + issuance agents -1 -> last age does not need agent
      # list of all agent ids -> ["purchase_agent", "issuance_agent_1", ..., "issuance_agent_6"] = from age 3-8 (excluded last one)
      agent_ids = []
      for n in range(num_agents):
         if n == 0:  # purchase agent is the first agent
            agent_ids.append("purchase_agent")
         else:  # enumerate each issuance agent -> issuance_agent_1 for age 3 ...
            agent_ids.append(f"issuance_agent_{issuance_ages[n-1]}")
      policies = agent_ids
      policies_to_train = agent_ids
   # # MULTI-AGENT WITH GROUPING:
   elif problem_id[11:]=="group":
      multi_agent_setting = "multi_group"
      num_agents = 1 + nProducts    #  purchase agent + product agents
      agent_ids = []
      for n in range(num_agents):
         if n == 0:
            agent_ids.append("purchase_agent")
         else:
            agent_ids.append(f"production_agent_{n-1}")
      policies = agent_ids
      policies_to_train = agent_ids
   elif problem_id[11:]=="production":
      multi_agent_setting = "multi_production"
      num_agents = 1 + nProducts
      agent_ids = []
      for n in range(num_agents):
         if n == 0:
            agent_ids.append("purchase_agent")
         else:
            agent_ids.append(f"production_agent_{n-1}")
      policies = agent_ids
      policies_to_train = agent_ids

   #create config dict which is passed to environment init 
   AIE_config = {"numAges":numAges, "nProducts":nProducts, "targetAges":targetAges, "ageRange":ageRange, "maxInventory":maxInventory, "evaporation":evaporation, 
    	         "demand_elasticity": demand_elasticity, "demand_means":demand_means, "demand_covs":demand_covs, "salvage":salvage, "price_means": price_means, "speed": speed,
               "vol": vol, "price_covs": price_covs, "correlation_matrix": correlation_matrix, "priceDistributions":priceDistributions, "holdingCosts":holding_cost, "expected_revenue":expected_revenue, "slope": slope,
               "min_ppf":min_ppf, "max_ppf":max_ppf, "production_step_size":production_step_size, "production_step_size_lp":production_step_size_lp, "price_step_size":price_step_size, "upper_bound":upper_bound, "demand_max": demand_max,
               "action_space_design":"box_continuous", "render_mode":'rgb_array', "horizon":horizon, "simulate_heuristic":False, "use_common_random_numbers": use_common_random_numbers, "multi_agent_setting": multi_agent_setting,
               "reward_lb":-1.0, "reward_ub":1.0, "use_issuance_model":use_issuance_model}  

   ray.init(num_cpus=num_workers+1, num_gpus=0)
   
   eval_runs = 10
   eval_length = 500
   eval_buffer = {i: {j: list(multi_norm.rvs()) for j in range(eval_length)} for i in range(eval_runs)}

   register_env("CheeseEnvironment", lambda config: env(config))
   test_env = env(AIE_config)
   ray.rllib.utils.check_env(test_env)

   eval_starting_prices, eval_starting_inv = test_env.simulate_starting_state_eval(50)
   # eval_heuristic = []
   # for i in range(eval_runs):
   #    eval_heuristic += [test_env.simulate_w_cdfs(cdfs=eval_buffer[i], policy=None, initial_prices=eval_starting_prices, initial_inventory=eval_starting_inv, warm_up_length=0)[2]]

   mean_eval_heuristic = 0#np.mean(eval_heuristic)
   print("HEURISTIC AVERAGE EVAL: ", mean_eval_heuristic)    

   path_ub = Path(f"{os.getcwd()}\\cheese\\{problem_id}\\upper_bound.json")
   if not path_config.is_file():
      # with open (path_ub, 'r') as f:
      #    res = json.load(f)
      # ub_json = {"max_reward": res["max_reward"], "inventory_position": res["inventory_position"]}
      ub = ub_function(test_env)
      print("UB: ", ub)
      JSON_config = {"numAges":numAges, "nProducts":nProducts, "targetAges":targetAges, "ageRange":ageRange, "maxInventory":maxInventory, "evaporation":evaporation, 
    	         "demand_elasticity": demand_elasticity, "demand_means":demand_means, "demand_covs":demand_covs, "salvage":salvage, "price_means": price_means,
               "speed": speed, "vol": vol, "price_covs": price_covs, "correlation_matrix": correlation_matrix,
               "demand_max": demand_max, "holdingCosts":holding_cost, "expected_revenue":expected_revenue, "slope": slope,
               "min_ppf":min_ppf, "max_ppf":max_ppf, "production_step_size":production_step_size, "price_step_size":price_step_size, "upper_bound":ub}  
      print("WRITE TO CONFIG FILE")
      with open(path_config, 'w') as f:
         json.dump(JSON_config, f)

   register_trainable("APO", apo.APO)
   assert tf.executing_eagerly()

   #restore algorithm from checkpoint if required
   if recovery_checkpoint is not None:
      algo = Algorithm.from_checkpoint(recovery_checkpoint)
      with open(checkpoint_dir+"best_checkpoint.json", 'r') as f:
         checkpoint_stats = json.load(f)
      nCheckpoints = 3
      best_average_reward = checkpoint_stats["avg_reward"]
      best_reward_estimate = checkpoint_stats["reward_estimate"]
      best_eval = checkpoint_stats["eval"]
      #CHANGE THIS WHEM CHECKPOINTING
      average_reward_estimate_checkpoint = 0.59889
      bias_estimate_checkpoint = -0.019291
      def update_apo_estimates(w):
         for k in w.policy_map.keys():
            # print("UPDATING WORKER")
            # print(w)
            # print(w.policy_map[k].average_reward_estimate)
            w.policy_map[k].average_reward_estimate = average_reward_estimate_checkpoint
            w.policy_map[k].bias_estimate = bias_estimate_checkpoint
            #print(w.policy_map[k].average_reward_estimate)
      
      algo.workers.foreach_worker(
        func=update_apo_estimates
      )
      for k in algo.workers.local_worker().policy_map.keys():
         print(f"LOCAL WORKER AVERAGE REWARD {k}: ", algo.workers.local_worker().get_policy(k).average_reward_estimate)
      print(f"RECOVERED ALGORITHM from checkpoint {recovery_checkpoint}")
      print("START ITERATION: ", start_iteration)
   else:
      print("TEST ENV ACTION SPACE: ", test_env.action_space)
      print("SAMPLE ACTION: ", test_env.action_space.sample())
      init_average_reward_estimate = test_env.simulate_n_steps(500, None, plot = False, warm_up = 20) #warmup 20
      print("initial average reward estimate from random policy: ", init_average_reward_estimate)

      adv_sampl = AIE_config["use_adversarial_sampling"] if "use_adversarial_sampling" in AIE_config else False
      #heuristic_average = test_env.get_heuristic_average(final_interval_width = 0.02)

      use_bias_normalization = False
      if use_bias_normalization:
         heuristic_average = test_env.get_heuristic_average()
      else:
         heuristic_average = None

      config = apo.APOConfig().environment("CheeseEnvironment", env_config=AIE_config)
      #config = ppo.PPOConfig().environment("AmelioratingInventory", env_config=AIE_config)
      config.reporting(metrics_num_episodes_for_smoothing=num_workers*10)
      config.rollouts(num_rollout_workers=num_workers, rollout_fragment_length='auto', batch_mode="complete_episodes")
      config.framework("tf2")
      config.multi_agent(policies=policies, policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id), policies_to_train=policies_to_train)
      config.training(lr=7e-5, model={"vf_share_layers": False, "fcnet_hiddens": [128,128]}, use_gae = True, lambda_=0.93, gamma=1.0, sgd_minibatch_size = int(minibatch_size*num_workers*horizon), num_sgd_iter=30, apo_step_size=0.2, bias_factor=0.4, use_bias_normalization = use_bias_normalization, heuristic_average=heuristic_average, init_average_reward_estimate = init_average_reward_estimate, shuffle_sequences = True, train_batch_size = num_workers*horizon, truncation_length=truncation_length, clip_param=0.2)
      #config.training(lr=1e-4, model={"vf_share_layers": False, "fcnet_hiddens": [64,64]}, use_gae = True, lambda_=0.9, gamma=0.99, sgd_minibatch_size = int(minibatch_size*num_workers*horizon), num_sgd_iter=30, shuffle_sequences = True, train_batch_size = num_workers*horizon, truncation_length=truncation_length)

      blending_setting = "full" 
      nn_setting = "p" if use_issuance_model else "full"
      lp_setting = "x" if use_issuance_model else "none"
      run_id = f"{blending_setting}_blending_{nn_setting}_nn_{lp_setting}_lp_{num_workers}w"
      
      print("RUN ID: ", run_id)

      param_space_config = config.to_dict()
      algo = config.build()
      nCheckpoints = 3
      best_average_reward = [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_reward_estimate = [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_eval = [(-np.Inf, None) for _ in range(nCheckpoints)]
      
   wandb.init(project=problem_id, name=run_id)
   if not os.path.isdir(storage_path+run_id):
      os.mkdir(storage_path+run_id)

   print("IS MULTI AGENT: ", algo.get_config().is_multi_agent())

   #track time per iteration
   #time_per_iteration = []

   for i in range(start_iteration, training_iterations):
      # start = time.time()
      print(f"TRAINING ITERATION {i}")
      train_results = algo.train()
      logger_dict = {}
      logger_dict["episode_reward_mean"] = train_results["episode_reward_mean"]
      if run_id.split("_")[-1] != "ppo":
         logger_dict["average_reward_estimate"] = train_results["info"]["learner"][f"{'purchase_agent' if num_agents > 1 else 'default_policy'}"]["learner_stats"]["average_reward_estimate"]
         logger_dict["bias_estimate"] = train_results["info"]["learner"][f"{'purchase_agent' if num_agents > 1 else 'default_policy'}"]["learner_stats"]["bias_estimate"]
      logger_dict["rewards"] = train_results["hist_stats"]["episode_reward"]
      for policy in train_results["info"]["learner"]:
         logger_dict[f"entropy_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["entropy"]
         logger_dict[f"kl_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["kl"]
         logger_dict[f"vf_explained_var_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["vf_explained_var"]
         logger_dict[f"total_loss_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["total_loss"]
         logger_dict[f"policy_loss_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["policy_loss"]
         logger_dict[f"vf_loss_{policy}"] = train_results["info"]["learner"][policy]["learner_stats"]["vf_loss"]
      wandb.log(logger_dict)
      #wandb.log({"episode_reward_mean":episode_reward_mean, "rewards":rewards, "entropy":entropy, "kl":kl, "vf_explained_var":vf_explained_var, "total_loss":total_loss, "policy_loss":policy_loss, "vf_loss":vf_loss})
      if i>start_iteration and (logger_dict["episode_reward_mean"] > best_average_reward[0][0]):
         if best_average_reward[0][1] is not None and best_average_reward[0][1] not in [best_reward_estimate[i][1] for i in range(nCheckpoints)] and best_average_reward[0][1] not in [best_eval[i][1] for i in range(nCheckpoints)]:
            algo.delete_checkpoint(best_average_reward[0][1])
         best_average_reward[0] = (logger_dict["episode_reward_mean"], checkpoint)
         best_average_reward.sort(key=lambda x: x[0])
      if i>start_iteration and run_id.split("_")[-1] != "ppo" and logger_dict["average_reward_estimate"] > best_reward_estimate[0][0]:
         if best_reward_estimate[0][1] is not None and best_reward_estimate[0][1] not in [best_average_reward[i][1] for i in range(nCheckpoints)] and best_reward_estimate[0][1] not in [best_eval[i][1] for i in range(nCheckpoints)]:
            algo.delete_checkpoint(best_reward_estimate[0][1])
         best_reward_estimate[0] = (logger_dict["average_reward_estimate"], checkpoint)
         best_reward_estimate.sort(key=lambda x: x[0])
      #delete checkpoint if none of the both outer if-loops is true
      if i>start_iteration and (logger_dict["episode_reward_mean"] < best_average_reward[0][0] and logger_dict["average_reward_estimate"] < best_reward_estimate[0][0]):
         if checkpoint_deletable and checkpoint not in [best_average_reward[i][1] for i in range(nCheckpoints)] and checkpoint not in [best_reward_estimate[i][1] for i in range(nCheckpoints)]:
            algo.delete_checkpoint(checkpoint)
      #save checkpoint
      checkpoint = algo.save(storage_path+run_id)
      checkpoint_deletable = True
      #evaluate policy every 5 times and save best checkpoint
      if i>=100 and i%5 == 0:
         eval_algo = []
         for j in range(eval_runs):
            eval_algo += [test_env.simulate_w_cdfs(cdfs=eval_buffer[j], policy=algo, initial_prices=eval_starting_prices, initial_inventory=eval_starting_inv, warm_up_length=0)[2]]
         eval_mean = np.mean(eval_algo)
         print(f"ITERATION {i}, EVALUATION: {eval_mean}")
         print(f"EVAL HEURISTIC: {mean_eval_heuristic}")
         if eval_mean > best_eval[0][0]:
            best_eval[0] = (eval_mean, checkpoint)
            best_eval.sort(key=lambda x: x[0])
            checkpoint_deletable = False
            if best_eval[0][1] is not None and best_eval[0][1] not in [best_average_reward[i][1] for i in range(nCheckpoints)] and best_eval[0][1] not in [best_reward_estimate[i][1] for i in range(nCheckpoints)]:
               algo.delete_checkpoint(best_eval[0][1])
         if eval_mean > mean_eval_heuristic:
            print("EVALUATION BETTER THAN HEURISTIC")
         
      with open(storage_path+run_id+"/best_checkpoint.json", 'w') as f:
         json.dump({"reward_estimate": best_reward_estimate, "avg_reward": best_average_reward, "eval": best_eval, "eval_heuristic":mean_eval_heuristic, "eval_buffer": eval_buffer}, f)
      # end = time.time()
      # time_per_iteration.append(end-start)
   algo.stop()
   # best_result = results.get_best_result(metric="episode_reward_mean", scope="all")
   # best_trial = results._experiment_analysis.get_best_trial(metric="info/learner/default_policy/learner_stats/average_reward_estimate", scope="all")
   # best_checkpoint = results._experiment_analysis.get_best_checkpoint(best_trial, metric ="info/learner/default_policy/learner_stats/average_reward_estimate")

   print("CHECKPOINT: ", best_reward_estimate)

   #print time per iteration to json   
   # with open(storage_path+"/time_per_iteration.json", 'w') as f:
   #    json.dump(time_per_iteration, f)
   
   data_size = 50_000

   if best_reward_estimate is not None:
      algorithm_path = best_eval[-1][1] 
      checkpoint_info = get_checkpoint_info(algorithm_path)
      #raise ValueError(f"Checkpoint info: {checkpoint_info}")
      state = Algorithm._checkpoint_info_to_algorithm_state(
         checkpoint_info = checkpoint_info,
         policy_ids = None,
         policy_mapping_fn=None,
         policies_to_train=None,
      )
      state["config"]["num_workers"] = 1
      policy = Algorithm.from_state(state)
      print("Policy loaded")
      test_env = env(AIE_config)
      features, responses, rewards = test_env.simulate_data_for_regression(policy, data_size=data_size)
      regression_data = {"features":features.tolist(), "responses":responses.tolist(), "rewards": rewards.tolist()}
      with open(storage_path+run_id+"/regression_data.json", 'w') as f:
               json.dump(regression_data, f)
 
#----------------------------------------#
if __name__ == '__main__':
	main()
