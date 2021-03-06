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
using LaserTag
using Plots
theme(:mute)
pyplot()

function move_towards(pomdp, b)
    s = typeof(b) <: LTState ? b : rand(b)
    # try to sneak up diagonally
    diff = s.opponent-s.robot
    dx = diff[1]
    dy = diff[2]
    if abs(dx) == 0 && abs(dy) == 0
        LaserTag.DIR_TO_ACTION[[0, 0]]
    elseif abs(dx) < abs(dy)
        LaserTag.DIR_TO_ACTION[[0, sign(dy)]]
    else
        LaserTag.DIR_TO_ACTION[[sign(dx), 0]]
    end
end
move_towards_policy = FunctionPolicy(b->move_towards(nothing, b))

function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        @fastmath util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

m = gen_lasertag()
qmdp = solve(QMDPSolver(max_iterations=1000, verbose=false), m)
POMDPs.action(p::AlphaVectorPolicy, s::LTState) = action(p, ParticleCollection([s]))

# For AdaOPS
Base.convert(::Type{SVector{2,Float64}}, s::LTState) = s.opponent
grid = StateGrid([2:7;], [2:11;])
# bounds = AdaOPS.IndependentBounds(-20.0, POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
blind = BlindPolicySolver(max_iterations=1000)
qmdp = QMDPSolver(max_iterations=1000)
bounds = AdaOPS.IndependentBounds(POValue(blind), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), (p,b)->value(qmdp,b), check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=300, tree_in_info=true, default_action=move_towards_policy, bounds_warnings=false)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        delta=0.1,
                        m_min=10,
                        grid=grid,
                        bounds_warnings=true,
                        tree_in_info=true,
                        # default_action=move_towards_policy,
                        num_b=50000
                        )
despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
@time action_info(despot, b0)
@time action_info(adaops, b0)

a, info = action_info(adaops, b0)
info_analysis(info)

# @profview action(adaops, b0)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, LowVarianceResampler(num_particles), num_particles)
# @show simulate(HistoryRecorder(max_steps=100), m, despot, belief_updater(m), b0, s0)
hist = simulate(HistoryRecorder(max_steps=100), m, adaops, belief_updater(m), b0, s0)
@show discounted_reward(hist)
hist_analysis(hist)

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
