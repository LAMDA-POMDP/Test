@everywhere using AA228FinalProject

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
num_particles = 30000 # number of particles in belief
v_noise_coeff = 0.3
om_noise_coeff = 0.1
belief_updater = (m)->RoombaParticleFilter(m, num_particles, v_noise_coeff, om_noise_coeff)

grid = RectangleGrid(range(-25, stop=15, length=201),
                   range(-20, stop=5, length=126),
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


@everywhere struct LidarRoombaBounds
    mdp_policy::LocalApproximationValueIterationPolicy
    values::Vector{Float64}
    time_pen::Float64
    discount::Float64
end

@everywhere function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::LidarRoombaBounds, pomdp::RoombaPOMDP, b::WPFBelief, W::Vector{Vector{Float64}}, obs::Vector{Float64}, max_depth::Int, bounds_warning::Bool)
    resize!(bd.values, n_particles(b))
    broadcast!((s)->value(bd.mdp_policy, s), bd.values, particles(b))
    @inbounds for i in eachindex(W)
        U[i] = dot(bd.values, W[i]) / sum(W[i])
        L[i] = bd.time_pen + bd.discount * U[i]
    end
    return L, U
end

@everywhere struct LidarRoombaBoundsSolver <: Solver
    solver::LocalApproximationValueIterationSolver
end

@everywhere function POMDPs.solve(s::LidarRoombaBoundsSolver, m::RoombaPOMDP)
    policy = solve(s.solver, m)
    LidarRoombaBounds(policy, Float64[], AA228FinalProject.mdp(m).time_pen * (1-discount(m)^19) / (1-discount(m)), discount(m)^20)
end

# For AdaOPS
@everywhere Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)
grid = StateGrid(range(-25, stop=15, length=9)[2:end-1],
                range(-20, stop=5, length=6)[2:end-1],
                range(0, stop=2*pi, length=5)[2:end-1])

adaops_list = [
                :bounds=>[LidarRoombaBoundsSolver(approx_solver)],
                :delta=>[0.3],
                :grid=>[grid],
                :max_occupied_bins=>[(5*8-3*6)*4],
                :m_min=>[10, 20, 30]
		    ]

adaops_list_labels = [
                    ["(DelayedMDP, MDP)"],
                    [0.3],
                    ["FullGrid"],
                    [(5*8-3*6)*6],
                    [10, 20, 30],
		    ]
# # ARDESPOT
@everywhere function ARDESPOT.bounds(bd::LidarRoombaBounds, pomdp::POMDP, b::ARDESPOT.ScenarioBelief)
    U = 0.0
    for s in particles(b)
        U += value(bd.mdp_policy, s)
    end
    U /= weight_sum(b)
    L = bd.time_pen + bd.discount * U
    return L, U
end

ardespot_list = [:default_action=>[RandomSolver(),], 
                :bounds=>[LidarRoombaBoundsSolver(approx_solver)],
                :K=>[100],
                ]
ardespot_list_labels = [["random",], 
                ["(DelayedMDP, MDP)",],
                [100],
                ]

# For POMCPOW
mdp_estimator = FOValue(approx_solver)
pomcpow_list = [:default_action=>[RandomSolver()], 
                :estimate_value=>[mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :k_observation=>[1.0],
                :alpha_observation=>[1/300],
                :criterion=>[MaxUCB(1000.),]]
pomcpow_list_labels = [["random",], 
                        ["MDPValue"],
                        [100000,], 
                        [1.0,], 
                        [1.0],
                        [1/300],
                        ["UCB 1000"]]

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
                    max_queue_length=10,
                    belief_updater=belief_updater,
                    experiment_label="Roomba3_334",
                    full_factorial_design=true)
