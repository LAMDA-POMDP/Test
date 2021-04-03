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
using AA228FinalProject
using Printf
using LinearAlgebra

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

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
m = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP(config=3, aspace=action_space, v_max=max_speed))

# Belief updater
num_particles = 50000 # number of particles in belief
v_noise_coeff = 0.3
om_noise_coeff = 0.1
belief_updater = (m)->RoombaParticleFilter(m, num_particles, v_noise_coeff, om_noise_coeff)

grid = RectangleGrid(range(-25, stop=15, length=201),
                   range(-20, stop=5, length=126),
                   range(0, stop=2*pi, length=61),
                   range(0, stop=1, length=2)) # Create the interpolating grid
# grid = RectangleGrid(range(-25, stop=15, length=21),
#                     range(-20, stop=5, length=11),
#                     range(0, stop=2*pi, length=6),
#                     range(0, stop=1, length=2)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000)


struct LidarRoombaBounds
    mdp_policy::LocalApproximationValueIterationPolicy
    values::Vector{Float64}
    time_pen::Float64
    discount::Float64
end

function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::LidarRoombaBounds, pomdp::P, b::WPFBelief{S,A,O}, W::Vector{Vector{Float64}}, obs::Vector{O}, max_depth::Int, bounds_warning::Bool) where {S,A,O,P<:POMDP{S,A,O},B}
    resize!(bd.values, n_particles(b))
    broadcast!((s)->value(bd.mdp_policy, s), bd.values, particles(b))
    @inbounds for i in eachindex(W)
        U[i] = dot(bd.values, W[i]) / sum(W[i])
        L[i] = bd.time_pen + bd.discount * U[i]
    end
    return L, U
end

# For AdaOPS
Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)
grid = StateGrid(range(-25, stop=15, length=9)[2:end-1],
                range(-20, stop=5, length=6)[2:end-1],
                range(0, stop=2*pi, length=5)[2:end-1])

random_policy = RandomPolicy(m)

mdp = solve(approx_solver, m)
bounds = LidarRoombaBounds(mdp, Float64[], AA228FinalProject.mdp(m).time_pen * (1-discount(m)^19) / (1-discount(m)), discount(m)^20)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=0.4,
                        m_max=200,
                        max_occupied_bins=(5*8-3*6)*4,
                        tree_in_info=true,
                        num_b=100_000
                        )
adaops = solve(solver, m)
@time action(adaops, b0)
a, info = action_info(adaops, b0)
info_analysis(info)

# @profview action(adaops, b0)

# pomcpow = POMCPOWSolver(
#                 estimate_value=FOValue(mdp),
#                 tree_queries=1000000, 
#                 max_time=1.0, 
#                 k_observation=1.0,
#                 alpha_observation=1.0,
#                 criterion=MaxUCB(1000.)
# )
# pomcpow = solve(pomcpow, m)
# @show r = simulate(RolloutSimulator(max_steps=100), m, pomcpow, belief_updater(m), b0, s0)
hist = simulate(HistoryRecorder(max_steps=100), m, adaops, belief_updater(m), b0, s0)
hist_analysis(hist)
@show undiscounted_reward(hist)