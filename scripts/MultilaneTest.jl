using Pkg
Pkg.activate("..")

using Multilane
using POMCPOW
using POMDPs
using POMDPSimulators
using MCTS
using ARDESPOT
using DataFrames
using CSV
using Random
using Printf

@show cor = 0.0
@show lambda = 2.0

@show N = 1
@show n_iters = 1_000_000
@show max_time = 1.0
@show max_depth = 40
@show val = SimpleSolver()

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=8.0,
                 k_state=4.5,
                 alpha_state=1/10.0,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(val)
                 # estimate_value=val
                )

solvers = Dict{String, Solver}(
    "despot" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(val), 1.02, check_terminal=true)
        sol = DESPOTSolver(lambda=0.01,
                     K=100,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     random_source=MemorizingSource(100, max_depth, rng, min_reserve=25),
                     default_action=ReportWhenUsed(MLAction(0.0, 0.0)),
                     rng=rng)
    end, 
    "pomcpow" => begin
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05) 
        POMCPOWSolver(tree_queries=n_iters,
                               criterion=MaxUCB(8.0),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=4.5,
                               alpha_observation=1/10.0,
                               estimate_value=FORollout(val),
                               # estimate_value=val,
                               check_repeat_obs=false,
                               node_sr_belief_updater=BehaviorPOWFilter(wup)
                              )
    end,
    "qmdp" => QBSolver(dpws),
    "pftdpw" => begin
        m = 15
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
        rng = MersenneTwister(123)
        up = BehaviorParticleUpdater(nothing, m, 0.0, 0.0, wup, rng)
        BBMDPSolver(dpws, up)
    end,
    "pftdpw_5" => begin
        m = 5
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
        rng = MersenneTwister(123)
        up = BehaviorParticleUpdater(nothing, m, 0.0, 0.0, wup, rng)
        BBMDPSolver(dpws, up)
    end,
    "pftdpw_10" => begin
        m = 10
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
        rng = MersenneTwister(123)
        up = BehaviorParticleUpdater(nothing, m, 0.0, 0.0, wup, rng)
        BBMDPSolver(dpws, up)
    end,
    "pftdpw_20" => begin
        m = 20
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
        rng = MersenneTwister(123)
        up = BehaviorParticleUpdater(nothing, m, 0.0, 0.0, wup, rng)
        BBMDPSolver(dpws, up)
    end,
    "simple" => SimpleSolver()
)

behaviors = standard_uniform(correlation=cor)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=1000.0
                             )
rmodel = SuccessReward(lambda=lambda)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

for (k, solver) in solvers
    @show k

    sims = []
    for i in 1:N
        if k == "qmdp"
            planner = deepcopy(solve(solver, mdp))
        else
            planner = deepcopy(solve(solver, pomdp))
        end
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05) 
        filter = BehaviorParticleUpdater(pomdp, 5000, 0.0, 0.0, wup, MersenneTwister(i+50_000))
        # filter = AggressivenessUpdater(pomdp, 5000, 0.0, 0.0, wup, MersenneTwister(i+50_000))

        rng = MersenneTwister(i+70_000)
        is = rand(rng, initialstate(pomdp))
        ips = MLPhysicalState(is)

        md = Dict(:solver=>k, :i=>i)
        simulate(HistoryRecorder(max_steps=5, rng=rng), pomdp, planner, filter, ips, is)
    end
end
