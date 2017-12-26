import copy
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import vrep
from arguments import get_args
from kfac import KFACOptimizer
from model import MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

# DO VREP THING HERE

scene_name = 'quad_innerloop.ttt'
quad_name = 'Quadricopter'
propellers = ['rotor1thrust', 'rotor2thrust', 'rotor3thrust', 'rotor4thrust']


def vrep_exit(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    exit(0)


# Start V-REP connection
try:
    vrep.simxFinish(-1)
    print("Connecting to simulator...")
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID == -1:
        print("Failed to connect to remote API Server")
        vrep_exit(clientID)
    # Setup V-REP simulation
    print("Setting simulator to async mode...")
    vrep.simxSynchronous(clientID, True)
    dt = .001
    vrep.simxSetFloatingParameter(clientID,
                                  vrep.sim_floatparam_simulation_time_step,
                                  dt,  # specify a simulation time step
                                  vrep.simx_opmode_oneshot)

    # Load V-REP scene
    print("Loading scene...")
    vrep.simxLoadScene(clientID, scene_name, 0xFF, vrep.simx_opmode_blocking)
    # Initialize rotors
    print("Initializing propellers...")
    for i in range(len(propellers)):
        vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)
except KeyboardInterrupt:
    vrep_exit(clientID)

err, quad_handle = vrep.simxGetObjectHandle(clientID, quad_name, vrep.simx_opmode_blocking)

# Initialize quadrotor position and orientation
vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_streaming)


def reset():
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    time.sleep(0.1)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    err, pos = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    return np.array(euler)


def envstep(action):
    # Get old quadrotor state
    err, pos_old = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler_old = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

    # Set propeller thrusts
    print("Setting propeller thrusts...")
    propeller_vels = action

    # Send propeller thrusts
    print("Sending propeller thrusts...")
    [vrep.simxSetFloatSignal(clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in
     zip(propellers, propeller_vels)]

    # Trigger simulator step
    print("Stepping simulator...")
    vrep.simxSynchronousTrigger(clientID)

    # Get new quadrotor state
    err, pos_new = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler_new = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

    observation = np.array(euler_new)

    reward_alpha = np.float32(-np.fabs(euler_new[0])) if not check_quad_flipped(euler_new) else np.float32(-10.)
    reward_beta = np.float32(-np.fabs(euler_new[1])) if not check_quad_flipped(euler_new) else np.float32(-10.)
    reward_gamma = np.float32(-np.fabs(euler_new[2])) if not check_quad_flipped(euler_new) else np.float32(-5.)

    reward_alpha_acc = np.float32(-np.fabs(euler_new[0] - euler_old[0])) if not check_quad_flipped(
        euler_new) else np.float32(-10.)
    reward_beta_acc = np.float32(-np.fabs(euler_new[1] - euler_old[1])) if not check_quad_flipped(
        euler_new) else np.float32(-10.)
    reward_gamma_acc = np.float32(-np.fabs(euler_new[2] - euler_old[2])) if not check_quad_flipped(
        euler_new) else np.float32(-5.)

    reward = reward_alpha + reward_beta + reward_gamma + reward_alpha_acc * 10. + reward_beta_acc * 10. \
             + reward_gamma_acc * 10.

    done = False
    info = np.array(euler_new)

    return observation, reward, done, info


def check_quad_flipped(euler):
    if abs(euler[0]) > 2. or abs(euler[1]) > 2.:
        print("Quad flipped")
        return True
    else:
        return False


# END VREP THING HERE

def main():
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    observation_space = np.zeros((3, 1))
    action_space = np.zeros((4, 1))

    obs_shape = np.shape(observation_space)
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    # if observation_space == 3:
    #     actor_critic = CNNPolicy(obs_shape[0], action_space, args.recurrent_policy)
    # else:
    #     assert not args.recurrent_policy, \
    #         "Recurrent policy is not implemented for the MLP controller"
    #     actor_critic = MLPPolicy(obs_shape[0], action_space)
    actor_critic = MLPPolicy(obs_shape[0], action_space)

    action_shape = np.shape(action_space)[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = np.shape(observation_space)[0]
        obs = torch.from_numpy(obs).float()
        print(np.shape(obs))
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next obs
            obs, reward, done, info = envstep(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward,
                            masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                Variable(rollouts.masks[:-1].view(-1, 1)),
                Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages,
                                                                  args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages,
                                                                     args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{"
                ":.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           final_rewards.mean(),
                           final_rewards.median(),
                           final_rewards.min(),
                           final_rewards.max(), dist_entropy.data[0],
                           value_loss.data[0], action_loss.data[0]))
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass


if __name__ == "__main__":
    main()
