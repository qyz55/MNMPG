import datetime
import os
import pprint
import time
import threading
import torch as th
import yaml
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.rl_utils import save_batch, save_q
from os.path import dirname, abspath
import os.path as osp

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda:
        th.cuda.set_device(args.device_num)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        args.tb_logs = tb_exp_direc
        # args.latent_role_direc = os.path.join(tb_exp_direc, "{}").format('latent_role')
        logger.setup_tb(tb_exp_direc)
        #dump config to the tb directory
        with open(os.path.join(tb_exp_direc, "config.yaml"), "w") as f:
            yaml.dump(_config, f, default_flow_style=False)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    th.autograd.set_detect_anomaly(True)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

#    args.own_feature_size = env_info["own_feature_size"] #unit_type_bits+shield_bits_ally
    #if args.obs_last_action:
    #    args.own_feature_size+=args.n_actions
    #if args.obs_agent_id:
    #    args.own_feature_size+=args.n_agents

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if args.learner == "hierarchical_rode_learner":
        scheme.update({
            "role_avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "roles": {"vshape": (1,), "group": "agents", "dtype": th.long}
        })
    if args.learner == "hierarchical_noise_q_learner":
        scheme.update(
            {"noise": {"vshape": (args.noise_dim,)}}
        )
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    if args.q_net_ensemble:
        mac = [mac_REGISTRY[args.mac](buffer.scheme, groups, args) for _ in range(args.ensemble_num)]
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        if args.runner=="meta_noise":
            runner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    if args.meta_h:
        last_meta_T = -args.meta_h_interval - 1
        meta_buffer = ReplayBuffer(scheme, groups, args.batch_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    use_rode=True if args.learner == "hierarchical_rode_learner" else False
    meta_start_t = 0
    if args.learner == "hierarchical_rode_learner":
        meta_start_t = args.role_action_spaces_update_start
    if args.save_batch_interval > 0:
        last_save_batch = -args.save_batch_interval - 1
    whole_q_list = []
    if args.save_q_all:
        q_list_ind = 0
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        # if args.meta_h:
        #     episode_batch, batch_log_p, mean_step_returns = runner.run(test_mode=False, meta_mode=True)
        # else:
        #     episode_batch, _ = runner.run(test_mode=False) #[8,181,10,1] for actions
        episode_batch, _ = runner.run(test_mode=False, use_rode=use_rode) #[8,181,10,1] for actions
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size) and args.meta_h and \
            (runner.t_env - last_meta_T) / args.meta_h_interval >= 1.0 and runner.t_env >= meta_start_t:
            repeat_times = args.batch_size // runner.batch_size
            # meta_buffer.insert_episode_batch(episode_batch)
            batch_log_p_all = []
            mean_step_returns_all = []
            for _ in range(repeat_times):
                               #[8]
                # episode_batch, batch_log_p, mean_step_returns = runner.run_meta(test_mode=False, meta_mode=True)
                # batch_log_p_all.append(batch_log_p)
                episode_batch, _, mean_step_returns = runner.run_meta(test_mode=False, meta_mode=True, use_rode=use_rode)
                mean_step_returns_all += mean_step_returns
                buffer.insert_episode_batch(episode_batch[0])
                meta_buffer.insert_episode_batch(episode_batch)
            #[32]
            # batch_log_p_all = th.cat(batch_log_p_all, dim=0)
            for _ in range(repeat_times):
                episode = prep_ep_and_train(meta_buffer, args, learner, episode, runner.t_env, whole_q_list)
            mean_step_returns_new_all = []
            for _ in range(repeat_times):
                episode_batch_new, mean_step_returns_new= runner.run_meta(test_mode=False, use_rode=use_rode)
                buffer.insert_episode_batch(episode_batch_new[0])
                mean_step_returns_new_all += mean_step_returns_new
            #need to get batch_log_p_here
            batch_log_p_all = runner.get_log_p(meta_buffer)
            learner.train_meta(batch_log_p_all, mean_step_returns_all, mean_step_returns_new_all, runner.t_env)
            for _ in range(repeat_times):
                episode = prep_ep_and_train(buffer, args, learner, episode, runner.t_env, whole_q_list)
            last_meta_T = runner.t_env
        elif buffer.can_sample(args.batch_size):
            prep_ep_and_train(buffer, args, learner, episode, runner.t_env, whole_q_list)
            # episode_sample = buffer.sample(args.batch_size) #[32,181,10,1] for actions

            # # Truncate batch to only filled timesteps
            # max_ep_t = episode_sample.max_t_filled()
            # episode_sample = episode_sample[:, :max_ep_t]

            # if episode_sample.device != args.device:
            #     episode_sample.to(args.device)

            # learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            save_batch_flag = False
            discount = 1.0 if args.t_max // 5 <= runner.t_env else 10.0
            if args.save_batch_interval > 0 and (runner.t_env - last_save_batch) / (args.save_batch_interval // discount) >= 1.0:
                save_batch_flag = True
                last_save_batch = runner.t_env
            for i in range(n_test_runs):
                if args.runner=="meta" or args.runner=="meta_noise":
                    runner.run_meta(test_mode=True, use_rode=use_rode)
                else:
                    runner.run(test_mode=True, use_rode=use_rode)
                if save_batch_flag:
                    save_batch(runner.batch, osp.join(args.tb_logs, "batch"), runner.t_env, i)
            if args.noise_bandit:
                for _ in range(n_test_runs):
                    runner.run_meta(test_mode=True, test_uniform=True)


        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run if args.runner != "meta" and args.runner != "meta_noise" else 1

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
        if args.save_q_all and len(whole_q_list) >= 4000:
            save_q(whole_q_list, osp.join(args.tb_logs, "q"), q_list_ind)
            whole_q_list.clear()
            q_list_ind += 1

    if args.save_q_all and len(whole_q_list) > 0:        
        save_q(whole_q_list, osp.join(args.tb_logs, "q"), q_list_ind)
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

def prep_ep_and_train(buffer, args, learner, episode, t_env, whole_q_list):
    for _ in range(args.repeat_training_times):
        for i in range(args.ensemble_num):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            if args.save_q_all:
                if args.learner == "hierarchical_q_learner":
                    q_all, mix_q_all, termed, skill_all = learner.train(episode_sample, t_env, episode, chosen_index = i, return_q_all=True)
                    whole_q_list.append({"q":q_all, "mix_q":mix_q_all, "termed":termed, "skill":skill_all})
                else:
                    q_all, mix_q_all, termed = learner.train(episode_sample, t_env, episode, chosen_index = i, return_q_all=True)
                    whole_q_list.append({"q":q_all, "mix_q":mix_q_all, "termed":termed})
            else:
                learner.train(episode_sample, t_env, episode, chosen_index = i)
    episode += args.batch_size_run if args.runner != "meta" and args.runner != "meta_noise" else 1
    return episode
