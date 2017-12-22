import vrep

scene_name = 'quad_innerloop.ttt'
quad_name = 'Quadricopter'
propellers = ['rotor1thrust', 'rotor2thrust', 'rotor3thrust', 'rotor4thrust']


def main():
    # Start V-REP connection
    try:
        vrep.simxFinish(-1)
        print("Connecting to simulator...")
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if clientID == -1:
            print("Failed to connect to remote API Server")
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            vrep.simxFinish(clientID)
    except KeyboardInterrupt:
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(clientID)
        return

    # Setup V-REP simulation
    print("Setting simulator to async mode...")
    vrep.simxSynchronous(clientID, True)
    dt = 0.01
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

    # Get quadrotor initial position and orientation
    err, pos = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)

    # Start simulation
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

    # Initialize rotors
    print("Initializing propellers...")
    for i in range(len(propellers)):
        vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)

    while (vrep.simxGetConnectionId(clientID) != -1):
        # Set propeller thrusts
        print("Setting propeller thrusts...")
        propeller_vels = [10.0, 10.0, 10.0, 10.0]
        for i in range(10):
            # Send propeller thrusts
            print("Sending propeller thrusts...")
            [vrep.simxSetFloatSignal(clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in
             zip(propellers, propeller_vels)]

            # Trigger simulator step
            print("Stepping simulator...")
            vrep.simxSynchronousTrigger(clientID)

            # Get new orientation
            err, euler = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
            print(euler)

            print("\n")


if __name__ == '__main__':
    main()
