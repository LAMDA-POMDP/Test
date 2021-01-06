@everywhere using LaserTag

@everywhere function move_towards(pomdp, b)
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
qmdp = QMDPSolver(max_iterations=1000)
mdp = ValueIterationSolver(max_iterations=1000, include_Q=false)
@everywhere POMDPs.action(p::AlphaVectorPolicy, s::LTState) = action(p, ParticleCollection([s]))

# For AdaOPS
@everywhere convert(s::LTState, pomdp::LaserTagPOMDP) = s.opponent
grid = StateGrid(convert, [2:7;], [2:11;])
pu_bounds = AdaOPS.IndependentBounds(FORollout(RandomSolver()), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
splpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

adaops_list1 = [
            #:default_action=>[move_towards_policy,],
            :bounds=>[splpu_bounds],
            :delta=>[0.3, 0.5, 0.7, 1.0],
            :grid=>[grid],
            :m_init=>[30],
            :sigma=>[2.0, 3.0],
            :zeta=>[0.03, 0.05, 0.07],
            :bounds_warnings=>[false],
            ]
adaops_list_labels1 = [
                    #["MoveTowards",],
                    ["(SemiPO_QMDP, QMDP)"],
                    [0.3, 0.5, 0.7, 1.0],
                    ["FullGrid"],
                    [30],
                    [2.0, 3.0],
                    [0.03, 0.05, 0.07],
                    [false],
                    ]
adaops_list2 = [
            #:default_action=>[move_towards_policy,],
            :bounds=>[splpu_bounds],
            :delta=>[0.03, 0.1],
            :grid=>[grid],
            :m_init=>[50],
            :sigma=>[2.0],
            :zeta=>[0.1, 0.15, 0.2],
            :bounds_warnings=>[false],
            ]
adaops_list_labels2 = [
                    #["MoveTowards",],
                    ["(SemiPO_QMDP, QMDP)"],
                    [0.03, 0.1],
                    ["FullGrid"],
                    [50],
                    [2.0],
                    [0.1, 0.15, 0.2],
                    [false],
                    ]

#= For ARDESPOT
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

# For POMCPOW
value_estimator = FOValue(mdp)
pomcpow_list = [:estimate_value=>[value_estimator],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :max_depth=>[90],
                    :criterion=>[MaxUCB(26.0),],
                    :final_criterion=[MaxTries()],
                    :enable_action_pw=>[false],
                    :check_repeat_obs=[false,],
                    :k_observation=>[4.,],
                    :alpha_observation=>[1/35,]
                    :tree_in_info=[false],
                    :default_action=[move_towards_policy],
                    ]
pomcpow_list_labels = [["MDP"],
                    [150000,],
                    [1.0,],
                    [90],
                    [26.0,],
                    ["MaxTries"],
                    [false],
                    [false],
                    [4.,],
                    [1/35,],
                    [false],
                    ["MoveTowards"],
                    ]

=#
# Solver list
solver_list = [
                #DESPOTSolver=>ardespot_list,
                AdaOPSSolver=>adaops_list1,
                AdaOPSSolver=>adaops_list2,
                #POMCPOWSolver=>pomcpow_list,
                # QMDPSolver=>[:max_iterations=>[1000,]],
                # FuncSolver=>[:func=>[move_towards,]],
                ]

solver_list_labels = [
                    #ardespot_list_labels,
                    adaops_list_labels1,
                    adaops_list_labels2,
                    #pomcpow_list_labels,
                    # [[1000,]],
                    # [["MoveTowards",]],
                    ]

solver_labels = [
                #"ARDESPOT",
                "AdaOPS",
                #"POMCPOW",
                # "QMDP",
                # "MoveTowards",
                ]

episodes_per_domain = 10
max_steps = 100

parallel_experiment(gen_lasertag,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=100,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=960,
                    experiment_label="LT100*10",
                    full_factorial_design=true) 
