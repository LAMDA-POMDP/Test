@everywhere using RoombaPOMDPs

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
@everywhere Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)
let l = 40, w = 25, t = 24, ζ=0.001, η=0.005, v_noise_coeff=0.02, om_noise_coeff=0.01, grid = StateGrid(range(-25, stop=15, length=l+1)[2:end-1],
                    range(-20, stop=5, length=w+1)[2:end-1],
                    range(0, stop=2*pi, length=t+1)[2:end-1])
    n_init = ceil(Int, KLDSampleSize(convert(Int, 11*l*w*t/20), ζ, η))
    global belief_updater = (m)->RoombaParticleFilter(m, n_init, v_noise_coeff, om_noise_coeff, KLDResampler(grid, n_init, ζ, η))
end
# num_particles = 50000 # number of particles in belief
# v_noise_coeff = 0.3
# om_noise_coeff = 0.1
# belief_updater = (m)->RoombaParticleFilter(m, num_particles, v_noise_coeff, om_noise_coeff)

grid = RectangleGrid(range(-25, stop=15, length=121),
                   range(-20, stop=5, length=76),
                   range(0, stop=2*pi, length=61),
                   range(-1, stop=1, length=3)) # Create the interpolating grid

# grid = RectangleGrid(range(-25, stop=15, length=41),
#                    range(-20, stop=5, length=26),
#                    range(0, stop=2*pi, length=13),
#                    range(-1, stop=1, length=3)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000)

@everywhere struct BumperRoombaBounds{M}
    qmdp::AlphaVectorPolicy
    blind::AlphaVectorPolicy
    m::M
    states::Vector{Int}
end

@everywhere function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::BumperRoombaBounds, pomdp::RoombaPOMDP, b::WPFBelief, W::Vector{Vector{Float64}}, obs::Vector{Bool}, max_depth::Int, bounds_warning::Bool)
    resize!(bd.states, n_particles(b))
    for (i, s) in enumerate(particles(b))
        bd.states[i] = convert_s(Int, s, bd.m)
    end
    @inbounds for i in eachindex(W)
        belief = WeightedParticleBelief(bd.states, W[i])
        U[i] = value(bd.qmdp, belief)
        L[i] = value(bd.blind, belief)
    end
    return L, U
end

@everywhere function ARDESPOT.bounds(bd::BumperRoombaBounds, pomdp::RoombaPOMDP, b::ARDESPOT.ScenarioBelief)
    resize!(bd.states, n_particles(b))
    for (i, s) in enumerate(particles(b))
        bd.states[i] = convert_s(Int, s, bd.m)
    end
    belief = ParticleCollection(bd.states)
    return value(bd.blind, belief), value(bd.qmdp, belief)
end

@everywhere struct BumperRoombaBoundsSolver <: Solver
    verbose::Bool
end
BumperRoombaBoundsSolver() = BumperRoombaBoundsSolver(false)

@everywhere function POMDPs.solve(s::BumperRoombaBoundsSolver, m::BumperPOMDP)
    mdp = RoombaPOMDPs.mdp(m)
    discrete_m = RoombaPOMDP(sensor=m.sensor, mdp=RoombaMDP(config=mdp.config, aspace=mdp.aspace, v_max=mdp.v_max, sspace=DiscreteRoombaStateSpace(121, 76, 61)))
    qmdp = solve(QMDPSolver(verbose=s.verbose), discrete_m)
    blind = solve(BlindPolicySolver(verbose=s.verbose), discrete_m)
    BumperRoombaBounds(qmdp, blind, discrete_m, Int[])
end

# For AdaOPS
@everywhere Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)

l = 16
w = 10
t = 1
grid = StateGrid(range(-25, stop=15, length=l+1)[2:end-1],
                    range(-20, stop=5, length=w+1)[2:end-1],
                    range(0, stop=2*pi, length=t+1)[2:end-1])

adaops_list = [
                :bounds=>[BumperRoombaBoundsSolver(true)],
                :delta=>[0.3],
                :grid=>[grid],
                :max_occupied_bins=>[convert(Int, 11*l*w*t/20)],
                :m_min=>[10, 30]
		    ]

adaops_list_labels = [
                    ["(Blind, QMDP)"],
                    [0.0],
                    ["FullGrid"],
                    [(5*8-3*6)*6],
                    [100, 200, 300],
            ]

# ARDESPOT
random = RandomSolver()
ardespot_list = [:default_action=>[random,], 
                :bounds=>[BumperRoombaBoundsSolver(true),],
                :lambda=>[0.1, 0.3],
                :K=>[300],
                :bounds_warnings=>[false,],
                ]
ardespot_list_labels = [["Random",], 
                ["(Blind, QMDP)",],
                [0.1, 0.3],
                [300],
                [false],
                ]

# For POMCPOW
mdp_estimator = FOValue(approx_solver)
pomcpow_list = [
                :estimate_value=>[mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :k_observation=>[1.0],
                :alpha_observation=>[1.0],
                :criterion=>[MaxUCB(1.), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)]]
pomcpow_list_labels = [
                        ["MDPValue"],
                        [100000,], 
                        [1.0,], 
                        [1.0,],
                        [1.0,],
                        ["UCB 1","UCB 10","UCB 100","UCB 1000"]]

# Solver list
solver_list = [
                #AdaOPSSolver=>adaops_list, 
                #DESPOTSolver=>ardespot_list,
                POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    #adaops_list_labels, 
                    #ardespot_list_labels,
                    pomcpow_list_labels,
                    ]
solver_labels = [
                #"ADAOPS",
                #"ARDESPOT",
                "POMCPOW",
                ]

                
episodes_per_domain = 250
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=4,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=100,
                    belief_updater=belief_updater,
                    experiment_label="BumperRoomba4_250",
                    full_factorial_design=true)
