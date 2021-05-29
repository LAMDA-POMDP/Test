# @everywhere using Roomba
@everywhere using RoombaPOMDPs

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
let k = 0
    global lidar_roomba_gen
    function lidar_roomba_gen()
        pomdp = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP(config=k%3+1, aspace=action_space, v_max=max_speed))
        k += 1
        return pomdp
    end
end
# Use lidar
pomdp = lidar_roomba_gen

# Belief updater
@everywhere Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)
let l = 40, w = 25, t = 24, ζ=0.001, η=0.005, v_noise_coeff=0.02, om_noise_coeff=0.01, grid = StateGrid(range(-25, stop=15, length=l+1)[2:end-1],
                    range(-20, stop=5, length=w+1)[2:end-1],
                    range(0, stop=2*pi, length=t+1)[2:end-1])
    n_init = ceil(Int, KLDSampleSize(convert(Int, 11*l*w*t/20), ζ, η))
    global belief_updater = (m)->RoombaParticleFilter(m, n_init, v_noise_coeff, om_noise_coeff, KLDResampler(grid, n_init, ζ, η))
end
# num_particles = 50000 # number of particles in belief
# belief_updater = (m)->BasicParticleFilter(m, LidarResampler(num_particles, m, v_noise_coeff, om_noise_coeff), num_particles)

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

@everywhere struct LidarRoombaBounds{M}
    m::M
    mdp::ValueIterationPolicy
    values::Vector{Float64}
    time_pen::Float64
    discount::Float64
end

@everywhere struct LidarRoombaBoundsSolver <: Solver
    verbose::Bool
end
@everywhere LidarRoombaBoundsSolver() = LidarRoombaBoundsSolver(false)
@everywhere function POMDPs.solve(sol::LidarRoombaBoundsSolver, m::RoombaPOMDP)
    mdp = RoombaPOMDPs.mdp(m)
    mdp = RoombaMDP(config=mdp.config, aspace=mdp.aspace, v_max=mdp.v_max, sspace=DiscreteRoombaStateSpace(121, 76, 61))
    mdp_policy = solve(ValueIterationSolver(verbose=sol.verbose), mdp)
    LidarRoombaBounds(mdp, mdp_policy, Float64[], mdp.time_pen * (1-discount(m)^19) / (1-discount(m)), discount(m)^20)
end

@everywhere function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::LidarRoombaBounds, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int, bounds_warning::Bool) where {S,A,O,P<:POMDP{S,A,O},B}
    resize!(bd.values, n_particles(b))
    broadcast!((s)->value(bd.mdp, convert_s(Int, s, bd.m)), bd.values, particles(b))
    @inbounds for i in eachindex(W)
        U[i] = dot(bd.values, W[i]) / sum(W[i])
        L[i] = min(bd.time_pen + bd.discount * U[i], U[i])
    end
    return L, U
end

# For AdaOPS

l = 16
w = 10
t = 1
grid = StateGrid(range(-25, stop=15, length=l+1)[2:end-1],
                    range(-20, stop=5, length=w+1)[2:end-1],
                    range(0, stop=2*pi, length=t+1)[2:end-1])

adaops_list = [
                :bounds=>[LidarRoombaBoundsSolver()],
                :delta=>[0.3],
                :grid=>[grid],
                :max_occupied_bins=>[convert(Int, 11*l*w*t/20)],
                :m_min=>[30]
		    ]

adaops_list_labels = [
                    ["(DelayedMDP, MDP)"],
                    [0.3],
                    ["FullGrid"],
                    [convert(Int, 11*l*w*t/20)],
                    [30],
		    ]

adaops_list1 = [
                :bounds=>[LidarRoombaBoundsSolver()],
                :delta=>[0.3],
                :Deff_thres=>[0.0],
                :grid=>[grid],
                :max_occupied_bins=>[convert(Int, 11*l*w*t/20)],
                :m_min=>[10, 30]
		    ]

adaops_list_labels1 = [
                    ["(DelayedMDP, MDP)"],
                    [0.3],
                    [0.0],
                    ["FullGrid"],
                    [convert(Int, 11*l*w*t/20)],
                    [10, 30],
		    ]

# # ARDESPOT
@everywhere function ARDESPOT.bounds(bd::LidarRoombaBounds, pomdp::POMDP, b::ARDESPOT.ScenarioBelief)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0, 0.0
    end
    U = 0.0
    for s in particles(b)
        U += value(bd.mdp, convert_s(Int, s, bd.m))
    end
    U /= n_particles(b)
    L = min(bd.time_pen + bd.discount * U, U)
    return (L, U)
end

ardespot_list = [:default_action=>[RandomSolver(),], 
                :bounds=>[LidarRoombaBoundsSolver()],
                :K=>[30],
                :lambda=>[0.01]
                ]
ardespot_list_labels = [["random",], 
                ["(DelayedMDP, MDP)",],
                [30],
                [0.01]
                ]

# For POMCPOW
mdp_estimator = FOValue(approx_solver)
pomcpow_list = [
                :estimate_value=>[mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :k_observation=>[2.0],
                :alpha_observation=>[0.03],
                :criterion=>[MaxUCB(1000.),]]
pomcpow_list_labels = [
                        ["MDPValue"],
                        [100000,], 
                        [1.0,], 
                        [2.0],
                        [0.03],
                        ["UCB 1000"]]

# Solver list
solver_list = [
                # AdaOPSSolver=>adaops_list, 
                AdaOPSSolver=>adaops_list1, 
                # DESPOTSolver=>ardespot_list,
                # POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    # adaops_list_labels, 
                    adaops_list_labels1,
                    # ardespot_list_labels,
                    # pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                # "ARDESPOT",
                # "POMCPOW",
                ]

                
episodes_per_domain = 334
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=3,
                    domain_queue_length=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=100,
                    belief_updater=belief_updater,
                    experiment_label="LidarRoomba3_334a",
                    full_factorial_design=true)
