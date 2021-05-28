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
using RoombaPOMDPs
using Printf
using LinearAlgebra
using SparseArrays
using POMDPModels
using LocalApproximationValueIteration
using LocalApproximationRandomStrategy
theme(:mute)
pyplot()

m = LightDark1D()

POMDPs.actionindex(m::LightDark1D, a::Int) = a + 2

interp = LocalGIFunctionApproximator(RectangleGrid(range(-1, stop=1, length=3), range(-100, stop=100, length=401)))
                     
approx_mdp = solve(LocalApproximationValueIterationSolver(
                                interp,
                                verbose=true,
                                max_iterations=1000,
                                is_mdp_generative=true,
                                n_generative_samples=1000),
                            m)

approx_random = solve(LocalApproximationRandomSolver(
                                interp,
                                verbose=true,
                                max_iterations=1000,
                                is_mdp_generative=true,
                                n_generative_samples=1000),
                            m)

# For AdaOPS
Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState) = SVector{1,Float64}(s.y)
grid = StateGrid(range(-10, stop=15, length=26))
bounds = AdaOPS.IndependentBounds(FOValue(approx_random), FOValue(approx_mdp), check_terminal=true)

b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=1.0,
                        m_min=10,
                        tree_in_info=true,
                        num_b=100000,
                        )
adaops = solve(solver, m)
@time action(adaops, b0)
a, info = action_info(adaops, b0)
info_analysis(info)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, LowVarianceResampler(num_particles), num_particles)

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