@everywhere using AA228FinalProject

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

# Use Bumper
pomdp = bumper_roomba_gen

# Belief updater
num_particles = 50000 # number of particles in belief
v_noise_coeff = 0.3
om_noise_coeff = 0.1
belief_updater = (m)->RoombaParticleFilter(m, num_particles, v_noise_coeff, om_noise_coeff)

# grid = RectangleGrid(range(-25, stop=15, length=201),
#                    range(-20, stop=5, length=126),
#                    range(0, stop=2*pi, length=61),
#                    range(-1, stop=1, length=3)) # Create the interpolating grid

grid = RectangleGrid(range(-25, stop=15, length=41),
                   range(-20, stop=5, length=26),
                   range(0, stop=2*pi, length=13),
                   range(-1, stop=1, length=3)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000)

@everywhere struct BumperRoombaBounds{M}
    fib::AlphaVectorPolicy
    blind::AlphaVectorPolicy
    m::M
    states::Vector{Int}
end

@everywhere function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::BumperRoombaBounds, pomdp::RoombaPOMDP, b::WPFBelief, W::Vector{Vector{Float64}}, obs::Vector{Bool}, max_depth::Int, bounds_warning::Bool)
    resize!(bd.states, n_particles(b))
    for (i, s) in enumerate(particles(b))
        bd.states[i] = convert_s(Int, s, bd.m)
    end
    n_states = AA228FinalProject.n_states(bd.m)
    @inbounds for i in eachindex(W)
        # belief_vec = sparsevec(bd.states, W[i]/sum(W[i]), n_states)
        belief_vec = WeightedParticleBelief(bd.states, W[i])
        U[i] = value(bd.fib, belief_vec)
        L[i] = value(bd.blind, belief_vec)
    end
    return L, U
end

@everywhere struct BumperRoombaBoundsSolver <: Solver end

@everywhere function POMDPs.solve(s::BumperRoombaBoundsSolver, m::BumperPOMDP)
    mdp = AA228FinalProject.mdp(m)
    discrete_m = RoombaPOMDP(sensor=m.sensor, mdp=RoombaMDP(config=mdp.config, aspace=mdp.aspace, v_max=mdp.v_max, sspace=DiscreteRoombaStateSpace(41, 26, 20)))
    fib = solve(FIBSolver(), discrete_m)
    blind = solve(BlindPolicySolver(), discrete_m)
    BumperRoombaBounds(fib, blind, discrete_m, Int[])
end

# For AdaOPS
@everywhere Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)

l = 8
w = 5
t = 4
grid = StateGrid(range(-25, stop=15, length=l+1)[2:end-1],
                    range(-20, stop=5, length=w+1)[2:end-1],
                    range(0, stop=2*pi, length=t+1)[2:end-1])

adaops_list = [
                :bounds=>[BumperRoombaBoundsSolver()],
                :delta=>[0.3],
                :grid=>[grid],
                :max_occupied_bins=>[convert(Int, 11*l*w*t/20)],
                :m_min=>[10, 30]
		    ]

adaops_list_labels = [
                    ["(Blind, FIB)"],
                    [0.0],
                    ["FullGrid"],
                    [(5*8-3*6)*6],
                    [100, 200, 300],
            ]

# ARDESPOT
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
pomcpow_list = [
                :estimate_value=>[mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :k_observation=>[1.0, 2.0],
                :alpha_observation=>[1.0, 1/300],
                :criterion=>[MaxUCB(100.)]]
pomcpow_list_labels = [
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
                    experiment_label="BumperRoomba3_300",
                    full_factorial_design=true)
