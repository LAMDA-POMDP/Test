@everywhere using Roomba

max_speed = 5.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 1:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
let k = 0
    global bumper_roomba_gen
    function bumper_roomba_gen()
        pomdp = RoombaPOMDP(sensor=Bumper(), mdp=RoombaMDP(config=k%3+1, aspace=action_space, v_max=max_speed))
        k += 1
        return pomdp
    end
end
# Use lidar
pomdp = bumper_roomba_gen

# Belief updater
num_particles = 30000 # number of particles in belief
pos_noise_coeff = 0.3
ori_noise_coeff = 0.1
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles, BumperResampler(num_particles, m, pos_noise_coeff, ori_noise_coeff)), num_particles)

grid = RectangleGrid(range(-25, stop=15, length=201),
                    range(-20, stop=5, length=101),
                    range(0, stop=2*pi, length=61),
                    range(0, stop=1, length=2)) # Create the interpolating grid
# grid = RectangleGrid(range(-25, stop=15, length=101),
#                    range(-20, stop=5, length=51),
#                    range(0, stop=2*pi, length=31),
#                    range(0, stop=1, length=2)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000)

running = RunningSolver()

# For AdaOPS
@everywhere Base.convert(::SVector{4,Float64}, s::RoombaState) = SVector{4,Float64}(s)
grid = StateGrid(range(-25, stop=15, length=7)[2:end-1],
                range(-20, stop=5, length=5)[2:end-1],
                range(0, stop=2*pi, length=4)[2:end-1],
                [1.])
splfu_bounds = AdaOPS.IndependentBounds(SemiPORollout(running), FOValue(approx_solver), check_terminal=true)
flfu_bounds = AdaOPS.IndependentBounds(FORollout(running), FOValue(approx_solver), check_terminal=true)
adaops_list = [:default_action=>[running], 
                    :bounds=>[splfu_bounds],
                    :delta=>[0.0],
                    :grid=>[grid],
                    :m_init=>[10],
                    :sigma=>[2, 3, 5],
                    :zeta=>[0.1, 0.2],
                    :overtime_warning_threshold=>[Inf],
                    :bounds_warnings=>[false,],
		    ]

adaops_list_labels = [["Running",], 
                    ["(SemiPO_Running, MDP)"],
                    [0.0],
                    ["FullGrid"],
                    [10],
                    [2, 3, 5],
                    [0.1, 0.2],
                    [Inf],
                    [false],
            ]

# ARDESPOT
bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(running), ARDESPOT.FullyObservableValueUB(approx_solver), check_terminal=true)
ardespot_list = [:default_action=>[running,], 
                :bounds=>[bounds,],
                :lambda=>[0.1, 0.3],
                :K=>[200],
                :bounds_warnings=>[false,],
                ]
ardespot_list_labels = [["Running",], 
                ["(Running, MDP)",],
                [0.1, 0.3],
                [200],
                [false],
                ]

# For POMCPOW
mdp_estimator = FOValue(approx_solver)
pomcpow_list = [:default_action=>[running,], 
                :estimate_value=>[mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :k_observation=>[1.0, 2.0],
                :alpha_observation=>[1.0, 1/300],
                :criterion=>[MaxUCB(100.)]]
pomcpow_list_labels = [["Running",], 
                        ["MDPValue"],
                        [100000,], 
                        [1.0,], 
                        [1.0, 2.0],
                        [1.0, 1/300],
                        ["UCB 100"]]

# Solver list
solver_list = [
                AdaOPSSolver=>adaops_list, 
                DESPOTSolver=>ardespot_list,
                POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    adaops_list_labels, 
                    ardespot_list_labels,
                    pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                "ARDESPOT",
                "POMCPOW",
                ]

                
episodes_per_domain = 300
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=3,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=300,
                    belief_updater=belief_updater,
                    experiment_label="BumperRoomba3*300",
                    full_factorial_design=true)