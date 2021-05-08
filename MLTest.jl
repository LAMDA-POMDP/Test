@everywhere using Multilane

cor = 0.0
l = 200
behaviors = standard_uniform(correlation=cor)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=1000.0
                             )
rmodel = SuccessReward(lambda=l)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

@everywhere function adjust1(l, d, k)
    1
end

@everywhere function adjust2(l, d, k)
    max(l, 1 - (0.1*k + 0.1*(1-d)))
end
#=
# For ARDESPOT
flfu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_towards_policy), ARDESPOT.FullyObservableValueUB(mdp), check_terminal=true, consistency_fix_thresh=1e-5)
plfu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), ARDESPOT.FullyObservableValueUB(mdp), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [:default_action=>[move_towards_policy,], 
                    :bounds=>[flfu_bounds, plfu_bounds],
                    :K=>[100],
                    :bounds_warnings=>[false],
                ]
ardespot_list_labels = [["MoveTowards",], 
                        ["(MoveTowards, MDP)", "(QMDPRollout, MDP)"],
                        [100],
                        [false],
                        ]

# For BS-DESPOT
qmdp_bounds = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(move_east), QMDPSolver(max_iterations=1000, verbose=true), check_terminal=true, consistency_fix_thresh=1e-5)

pl_despot_list = [
        :default_action=>[move_east],
        :bounds=>[qmdp_bounds],
        :K=>[100],
        :beta=>[0, 0.3],
        :adjust_zeta=>[adjust1, adjust2],
        :C=>[4]
]

pl_despot_list_labels = [
        ["move_east"],
        ["(MoveEast, QMDP)"],
        [100,],
        [0, 0.3],
        ["adjust_zeta1", "adjust_zeta2"],
        [4,]
]
=#
# For POMCPOW
val = SimpleSolver()
wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
pomcpow_list = [
                    :estimate_value=>[FORollout(val)],
                    :tree_queries=>[1000000,],
                    :max_time=>[1.0,],
                    :max_depth=>[40],
                    :criterion=>[MaxUCB(8.0),],
                    :final_criterion=>[MaxTries()],
                    :enable_action_pw=>[false],
                    :check_repeat_obs=>[false,],
                    :k_observation=>[4.5,],
                    :alpha_observation=>[1/10.0,],
                    :node_sr_belief_updater=>[BehaviorPOWFilter(wup)],
                    ]
pomcpow_list_labels = [
                    ["SimpleSolver"],
                    [1000000,],
                    [1.0,],
                    [40],
                    [8.0,],
                    ["MaxTries"],
                    [false],
                    [false],
                    [4.5,],
                    [1/10,],
                    ["wup"],
                    ]

# Solver list
solver_list = [
                #PL_DESPOTSolver=>pl_despot_list,
                #DESPOTSolver=>ardespot_list,
                #AdaOPSSolver=>adaops_list,
                POMCPOWSolver=>pomcpow_list,
                # QMDPSolver=>[:max_iterations=>[1000,]],
                # FuncSolver=>[:func=>[move_towards,]],
                ]

solver_list_labels = [
                    #pl_despot_list_labels,
                    #ardespot_list_labels,
                    #adaops_list_labels,
                    pomcpow_list_labels,
                    # [[1000,]],
                    # [["MoveTowards",]],
                    ]

solver_labels = [
                #"BSDESPOT",
                #"ARDESPOT",
                #"AdaOPS",
                "POMCPOW",
                # "QMDP",
                # "MoveTowards",
                ]

episodes_per_domain = 1000
max_steps = 50

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=1,
                    experiment_label="ML1000",
                    full_factorial_design=true) 