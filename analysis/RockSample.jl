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
using RockSample
using Statistics
using Combinatorics
using ProfileView
theme(:mute)
pyplot()

function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        @fastmath util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

m = RockSamplePOMDP(11, 11)

# qmdp = solve(RSQMDPSolver(), m)
mdp = solve(RSMDPSolver(), m)
rs_exit = solve(RSExitSolver(), m)
POMDPs.action(p::AlphaVectorPolicy, s::RSState) = action(p, ParticleCollection([s]))

b0 = initialstate(m)
s0 = rand(b0)
    
# bounds = AdaOPS.IndependentBounds(FOValue(rs_exit), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
bounds = AdaOPS.IndependentBounds(FOValue(rs_exit), FOValue(mdp), check_terminal=true, consistency_fix_thresh=1e-5)

adaops_solver = AdaOPSSolver(bounds=bounds,
                        delta=0.1,
                        m_min=100,
                        bounds_warnings=true,
                        tree_in_info=true,
                        default_action=rs_exit 
                        )
lower = (pomdp, b)->value(rs_exit, b)
# upper = (pomdp, b)->value(qmdp, b)
# bounds = ARDESPOT.IndependentBounds(lower, upper, check_terminal=true, consistency_fix_thresh=1e-5)
bounds = ARDESPOT.IndependentBounds(lower, ARDESPOT.FullyObservableValueUB(mdp), check_terminal=true, consistency_fix_thresh=1e-5)

despot_solver = DESPOTSolver(bounds=bounds, K=100, tree_in_info=true, default_action=rs_exit, bounds_warnings=false)

despot = solve(despot_solver, m)
adaops = solve(adaops_solver, m)
# @time p = solve(solver, m)
@time action(despot, b0)
# @time action(adaops, b0)
# show(stdout, MIME("text/plain"), info[:tree])
a, info = action_info(adaops, b0)
info_analysis(info)
a, info = action_info(despot, b0)
D = info[:tree]
@show mean(D.Delta)
@show std(D.Delta)
@show quantile(D.Delta, [0.1,0.9])
# @profview action(despot, b0)
# @profview action(adaops, b0)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, LowVarianceResampler(num_particles), num_particles)

hist = simulate(HistoryRecorder(max_steps=100), m, adaops, belief_updater(m), b0, s0)
hist_analysis(hist)
@show undiscounted_reward(hist)

# num_particles = 30000
# belief_updater = (m)->BasicParticleFilter(m, LowVarianceResampler(num_particles), num_particles)
# @show r = simulate(RolloutSimulator(), m, despot, belief_updater(m), b0, s0)
# @show r = simulate(RolloutSimulator(), m, adaops, belief_updater(m), b0, s0)
# let step = 1
#     for (s, b, a, o) in stepthrough(m, despot, belief_updater(m), b0, s0, "s, b, a, o", max_steps=100)
#         @show step
#         @show s
#         @show a
#         step += 1
#     end
# end
# let step = 1
#     for (s, b, a, o) in stepthrough(m, adaops, belief_updater(m), b0, s0, "s, b, a, o", max_steps=100)
#         @show step
#         @show s
#         @show a
#         step += 1
#     end
# end
