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
using FIB
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
using SparseArrays

function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        @fastmath util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

max_speed = 2.0
speed_gears = 2
max_turn_rate = 2.0
turn_gears = 3
# action_space = vec([RoombaAct(v, om*max_turn_rate/v) for v in range(1, stop=max_speed, length=speed_gears) for om in [-1, 1]])
action_space = vec([RoombaAct(v, om) for v in range(0, stop=max_speed, length=speed_gears) for om in range(-max_turn_rate, stop=max_turn_rate, length=turn_gears)])
m = RoombaPOMDP(sensor=Bumper(), mdp=RoombaMDP(config=1, aspace=action_space, v_max=max_speed))

# Belief updater
num_particles = 50000 # number of particles in belief
v_noise_coeff = 0.3
om_noise_coeff = 0.1
belief_updater = (m)->RoombaParticleFilter(m, num_particles, v_noise_coeff, om_noise_coeff)

struct BumperRoombaBounds{M}
    fib::AlphaVectorPolicy
    blind::AlphaVectorPolicy
    m::M
    states::Vector{Int}
end

function AdaOPS.bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::BumperRoombaBounds, pomdp::RoombaPOMDP, b::WPFBelief, W::Vector{Vector{Float64}}, obs::Vector{Bool}, max_depth::Int, bounds_warning::Bool)
    resize!(bd.states, n_particles(b))
    for (i, s) in enumerate(particles(b))
        bd.states[i] = convert_s(Int, s, bd.m)
    end
    n_states = AA228FinalProject.n_states(bd.m)
    @inbounds for i in eachindex(W)
        # belief_vec = sparsevec(bd.states, W[i]/sum(W[i]), n_states)
        belief = WeightedParticleBelief(bd.states, W[i])
        U[i] = value(bd.fib, belief)
        L[i] = value(bd.blind, belief)
    end
    return L, U
end

struct BumperRoombaBoundsSolver <: Solver end

function POMDPs.solve(s::BumperRoombaBoundsSolver, m::BumperPOMDP)
    mdp = AA228FinalProject.mdp(m)
    discrete_m = RoombaPOMDP(sensor=m.sensor, mdp=RoombaMDP(config=mdp.config, aspace=mdp.aspace, v_max=mdp.v_max, sspace=DiscreteRoombaStateSpace(41, 26, 20)))
    fib = solve(FIBSolver(), discrete_m)
    blind = solve(BlindPolicySolver(), discrete_m)
    BumperRoombaBounds(fib, blind, discrete_m, Int[])
end

# bounds = solve(BumperRoombaBoundsSolver(), m)

# For AdaOPS
Base.convert(::Type{SVector{3,Float64}}, s::RoombaState) = SVector{3,Float64}(s.x, s.y, s.theta)
grid = StateGrid(range(-25, stop=15, length=9)[2:end-1],
                range(-20, stop=5, length=6)[2:end-1],
                range(0, stop=2*pi, length=5)[2:end-1])

b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=0.0,
                        m_max=200,
                        max_occupied_bins=(5*8-3*6)*4,
                        tree_in_info=false,
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