using Pkg
Pkg.activate(".")

using BSDESPOT
using POMDPs
using POMDPSimulators
using RockSample
using ParticleFilters
using Statistics
using ProgressMeter

m = RockSamplePOMDP(11, 11)
move_east = solve(RSExitSolver(), m)
qmdp = solve(RSQMDPSolver(), m)
qmdp_bounds = BSDESPOT.IndependentBounds(BSDESPOT.DefaultPolicyLB(move_east), (p, b)->(value(qmdp, b)), check_terminal=true, consistency_fix_thresh=1e-5)
function lower_bounded_zeta(d, k, zeta=0.8)
    max(zeta, 1 - (0.2*k + 0.2*(1-d)))
end

despot = solve(BS_DESPOTSolver(default_action=move_east, bounds=qmdp_bounds, K=100, beta=0, adjust_zeta=(d,k)->1.0, C=1.0, tree_in_info=true), m)
bs_despot = solve(BS_DESPOTSolver(default_action=move_east, bounds=qmdp_bounds, K=100, beta=0, adjust_zeta=(d,k)->lower_bounded_zeta(d, k, 0.9), C=4.0, tree_in_info=true), m)

despot_gap = Float64[]
bs_despot_gap = Float64[]

up = SIRParticleFilter(m, 30000)
b0 = initialstate(m)
@showprogress for i in 1:30
    s0 = rand(b0)
    for info in stepthrough(m, despot, up, b0, s0, "action_info")
        D = info[:tree]
        push!(despot_gap, D.mu[1] - D.l[1])
    end
    for info in stepthrough(m, bs_despot, up, b0, s0, "action_info")
        D = info[:tree]
        push!(bs_despot_gap, D.mu[1] - D.l[1])
    end
end
@show mean(despot_gap), std(despot_gap)/sqrt(length(despot_gap))
@show mean(bs_despot_gap), std(bs_despot_gap)/sqrt(length(bs_despot_gap))