import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import vrep

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

scene_name = 'quad_innerloop.ttt'
quad_name = 'Quadricopter'
propellers = ['rotor1thrust', 'rotor2thrust', 'rotor3thrust', 'rotor4thrust']


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input, hidden, output):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, hidden)

        self.bn1 = nn.BatchNorm2d(input)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.linear3 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(self.bn1(x)).view(-1, self.hidden)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def select_action(model, state):
    return int(model(Variable(state, volatile=True)).data.max(1)[1].numpy())


def is_valid_state(pos_start, pos_current, euler_current):
    valid = True
    if pos_current[2] < 1.0:
        valid = False
    diff = np.fabs(pos_current - pos_start)
    if np.amax(diff[:3]) > 0.7:
        valid = False
    if check_quad_flipped(euler_current):
        valid = False
    return valid


def conduct_action(forces, action):
    if action <= 3:
        forces[action] = forces[action] + 1. if forces[action] < 20 else forces[action]
    else:
        forces[action - 4] = forces[action - 4] - 1. if forces[action - 4] >= 1 else 0.
    return forces


def reset(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    time.sleep(0.1)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)


ByteTensor = torch.ByteTensor


def DQN_update(model, memory, batch_size, GAMMA, optimizer):
    if len(memory) < 10:
        return
    elif len(memory) < batch_size:
        batch_size = len(memory)
    else:
        batch_size = batch_size
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size, 1))

    next_state_values[non_final_mask, 0] = model(non_final_next_states).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def vrep_exit(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    exit(0)


def check_quad_flipped(euler):
    if abs(euler[0]) > 0.2 or abs(euler[1]) > 0.2:
        print("Quad flipped")
        return True


def main():
    # Start V-REP connection
    try:
        vrep.simxFinish(-1)
        print("Connecting to simulator...")
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if clientID == -1:
            print("Failed to connect to remote API Server")
            vrep_exit(clientID)
    except KeyboardInterrupt:
        vrep_exit(clientID)
        return

    # Setup V-REP simulation
    print("Setting simulator to async mode...")
    vrep.simxSynchronous(clientID, True)
    dt = .0005
    vrep.simxSetFloatingParameter(clientID,
                                  vrep.sim_floatparam_simulation_time_step,
                                  dt,  # specify a simulation time step
                                  vrep.simx_opmode_oneshot)

    # Load V-REP scene
    print("Loading scene...")
    vrep.simxLoadScene(clientID, scene_name, 0xFF, vrep.simx_opmode_blocking)

    # Get quadrotor handle
    err, quad_handle = vrep.simxGetObjectHandle(clientID, quad_name, vrep.simx_opmode_blocking)

    # Initialize quadrotor position and orientation
    vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_streaming)

    # Start simulation
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

    # Initialize rotors
    print("Initializing propellers...")
    for i in range(len(propellers)):
        vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)

    # Get quadrotor initial position and orientation
    err, pos_old = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler_old = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

    pos_old = np.asarray(pos_old)
    euler_old = np.asarray(euler_old)

    pos_start = pos_old

    # hyper parameters

    n_input = 6
    n_action = 8
    hidden = 64
    memory = ReplayMemory(10000)
    batch_size = 512
    learning_rate = .01
    eps = 0.1
    gamma = 0.9
    loss = 0.0
    criterion = nn.MSELoss()
    refresh_freq = 2000

    net = DQN(n_input, hidden, n_action)

    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    state = [0 for i in range(n_input)]
    state = torch.from_numpy(np.asarray(state, dtype=np.float32)).view(-1, n_input)

    f = 0.
    propeller_vels = [f, f, f, f]

    net.eval()
    while (vrep.simxGetConnectionId(clientID) != -1):
        for iter in range(refresh_freq):

            # epsilon greedy
            r = random.random()
            if r > eps:
                action = select_action(net, state)
            else:
                action = random.randrange(n_action)

            # Set propeller thrusts
            print("Setting propeller thrusts...")
            propeller_vels = conduct_action(propeller_vels, action)
            # Send propeller thrusts
            print("Sending propeller thrusts...")
            [vrep.simxSetFloatSignal(clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in
             zip(propellers, propeller_vels)]

            # Trigger simulator step
            print("Stepping simulator...")
            vrep.simxSynchronousTrigger(clientID)

            # Get quadrotor initial position and orientation
            err, pos_new = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
            err, euler_new = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

            pos_new = np.asarray(pos_new)
            euler_new = np.asarray(euler_new)

            valid = is_valid_state(pos_start, pos_new, euler_new)
            if valid:
                next_state = np.concatenate([pos_new - pos_old, euler_new - euler_old])
            else:
                next_state = None

            next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).view(-1,
                                                                                         n_input) if next_state is not \
                                                                                                     None else None

            reward = np.float32(-np.linalg.norm(pos_new - pos_old))

            memory.push(state, torch.from_numpy(np.asarray([[action]])), next_state,
                        torch.from_numpy(np.asarray([[reward]])))

            state = next_state
            pos_old = pos_new
            euler_old = euler_new
            print(propeller_vels)
            print("\n")

            # Perform one step of the optimization (on the target network)

            DQN_update(net, memory, batch_size, gamma, optim)

            if not valid:
                # reset
                reset(clientID)
                print("Loading scene...")
                vrep.simxLoadScene(clientID, scene_name, 0xFF, vrep.simx_opmode_blocking)

                # Setup V-REP simulation
                print("Setting simulator to async mode...")
                vrep.simxSynchronous(clientID, True)
                dt = .0005
                vrep.simxSetFloatingParameter(clientID,
                                              vrep.sim_floatparam_simulation_time_step,
                                              dt,  # specify a simulation time step
                                              vrep.simx_opmode_oneshot)
                # Get quadrotor handle
                err, quad_handle = vrep.simxGetObjectHandle(clientID, quad_name, vrep.simx_opmode_blocking)

                # Initialize quadrotor position and orientation
                vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
                vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_streaming)

                # Start simulation
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

                # Get quadrotor initial position and orientation
                err, pos_old = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
                err, euler_old = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

                pos_old = np.asarray(pos_old)
                euler_old = np.asarray(euler_old)
                pos_start = pos_old

                state = [0 for i in range(n_input)]
                state = torch.from_numpy(np.asarray(state, dtype=np.float32)).view(-1, n_input)

                f = 0.
                propeller_vels = [f, f, f, f]


if __name__ == '__main__':
    main()
