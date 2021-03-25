using POMDPs
using ParticleFilters
using POMDPSimulators
using POMDPPolicies
using POMDPModelTools
using BeliefUpdaters
using ParallelExperiment

# Solver
using BSDESPOT
using AdaOPS
using ARDESPOT
# using SARSOP
# using PointBasedValueIteration
# using MCVI
using BasicPOMCP
using POMCPOW
using QMDP
using DiscreteValueIteration
using LocalApproximationValueIteration

using Random
using StaticArrays
using D3Trees
using CSV
using GridInterpolations
using LocalFunctionApproximation
using ProfileView
using Roomba
using Printf

# Belief Updater
struct POMDPResampler{R}
    n::Int
    r::R
end

POMDPResampler(n, r=LowVarianceResampler(n)) = POMDPResampler(n, r)

function ParticleFilters.resample(r::POMDPResampler,
                                  bp::WeightedParticleBelief,
                                  pm::POMDP,
                                  rm::POMDP,
                                  b,
                                  a,
                                  o,
                                  rng)

    if weight_sum(bp) == 0.0
        # no appropriate particles - resample from the initial distribution
        new_ps = [rand(rng, initialstate(pm)) for i in 1:r.n]
        return ParticleCollection(new_ps)
    else
        # normal resample
        return resample(r.r, bp, rng)
    end
end

function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        @fastmath util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

# max_speed = 2.0
# speed_gears = 2
# max_turn_rate = 2.0
# turn_gears = 3
# # action_space = vec([RoombaAct(v, om*max_turn_rate/v) for v in range(1, stop=max_speed, length=speed_gears) for om in [-1, 1]])
# action_space = vec([RoombaAct(v, om) for v in range(0, stop=max_speed, length=speed_gears) for om in range(-max_turn_rate, stop=max_turn_rate, length=turn_gears)])
# m = RoombaPOMDP(sensor=Bumper(), mdp=RoombaMDP(config=1, aspace=action_space, v_max=max_speed, contact_pen=-0.1))
max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
m = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP(config=3, aspace=action_space, v_max=max_speed, sspace=DiscreteRoombaStateSpace(50, 50, 20)))

# Belief updater
num_particles = 50000 # number of particles in belief
pos_noise_coeff = 0.3
ori_noise_coeff = 0.1
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles, BumperResampler(num_particles, m, pos_noise_coeff, ori_noise_coeff)), num_particles)

#grid = RectangleGrid(range(-25, stop=15, length=201),
#                    range(-20, stop=5, length=101),
#                    range(0, stop=2*pi, length=61),
#                    range(0, stop=1, length=2)) # Create the interpolating grid
# grid = RectangleGrid(range(-25, stop=15, length=21),
#                     range(-20, stop=5, length=11),
#                     range(0, stop=2*pi, length=6),
#                     range(0, stop=1, length=2)) # Create the interpolating grid
# interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

# approx_solver = LocalApproximationValueIterationSolver(interp,
#                                                         verbose=true,
#                                                         max_iterations=1000)

# mdp = solve(approx_solver, m)

# For AdaOPS
Base.convert(::SVector{4,Float64}, s::RoombaState) = SVector{4,Float64}(s)
grid = StateGrid(range(-25, stop=15, length=7)[2:end-1],
                range(-20, stop=5, length=5)[2:end-1],
                range(0, stop=2*pi, length=4)[2:end-1],
                [1.])

running = solve(RunningSolver(), m)
random_policy = RandomPolicy(m)

# bounds = AdaOPS.IndependentBounds(FORollout(RandomRush(m, 0.5)), FOValue(mdp), check_terminal=true)
# bounds = AdaOPS.IndependentBounds(SemiPORollout(ModePolicy(mdp)), FOValue(mdp), check_terminal=true)
bounds = AdaOPS.IndependentBounds(SemiPORollout(running), FOValue(mdp), check_terminal=true)
# bounds = AdaOPS.IndependentBounds(FORollout(running), FOValue(mdp), check_terminal=true)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=0.0,
                        zeta=0.1,
                        m_init=10,
                        sigma=8.0,
                        bounds_warnings=false,
                        tree_in_info=true,
                        num_b=30_000
                        )
adaops = solve(solver, m)
@time action(adaops, b0)
a, info = action_info(adaops, b0)
info_analysis(info)

@profview action(adaops, b0)

# pomcpow = POMCPOWSolver(
#                 default_action=running, 
#                 estimate_value=FOValue(mdp),
#                 tree_queries=1000000, 
#                 max_time=1.0, 
#                 k_observation=1.0,
#                 alpha_observation=1.0,
#                 criterion=MaxUCB(1000.)
# )
# pomcpow = solve(pomcpow, m)
# @show r = simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater(m), b0, s0)
# hist = simulate(HistoryRecorder(max_steps=100), m, adaops, belief_updater(m), b0, s0)
# hist_analysis(hist)
# @show undiscounted_reward(hist)