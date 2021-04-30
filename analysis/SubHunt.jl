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
using ProfileView
using SubHunt

rng = Random.GLOBAL_RNG

function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        @fastmath util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

# Low passive std may help agent identify useful information
info_gather_pomdp = SubHuntPOMDP(passive_std=0.5, ownspeed=5, passive_detect_radius=5, p_aware_kill=0.05)
pomdp = SubHuntPOMDP()

# Choose default domain
m = info_gather_pomdp
# qmdp= solve(QMDPSolver(max_iterations=1000, verbose=true), m)
# # mdp = solve(ValueIterationSolver(max_iterations=1000, verbose=true), UnderlyingMDP(m))
# ping_first = PingFirst(qmdp)
# qmdp_policy(p, b::AbstractParticleBelief) = value(qmdp, b)
POMDPs.action(p::PingFirst, b::SubHunt.SubHuntInitDist) = SubHunt.PING
POMDPs.action(p::AlphaVectorPolicy, s::SubState) = action(p, ParticleCollection([s]))

# For AdaOPS
Base.convert(::Type{SVector{3,Float64}}, s::SubState) = SVector{3, Float64}(s.target..., s.goal)
grid = StateGrid(range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=4, length=4)[2:end]
                            )

random_estimator = FORollout(RandomSolver())
# spl_bounds = AdaOPS.IndependentBounds(SemiPORollout(ping_first), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
# bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
bounds = AdaOPS.IndependentBounds(random_estimator, POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
# despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(ping_first), qmdp_policy, check_terminal=true, consistency_fix_thresh=1e-5)
despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), qmdp_policy, check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=100, tree_in_info=true, default_action=ping_first, bounds_warnings=false)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=0.1,
                        zeta=0.05,
                        m_min=20,
                        bounds_warnings=false,
                        default_action=ping_first,
                        tree_in_info=true,
                        num_b=30_000
                        )
adaops = solve(solver, m)
# @time p = solve(solver, m)
# @time action(despot, b0)
# @time action(adaops, b0)
a, info = action_info(adaops, b0)
info_analysis(info)
despot = solve(despot_solver, m)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, LowVarianceResampler(num_particles), num_particles)
@show r = simulate(RolloutSimulator(), m, adaops, belief_updater(m), b0, s0)
@show r = simulate(RolloutSimulator(), m, despot, belief_updater(m), b0, s0)
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
