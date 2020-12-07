@everywhere using RockSample

function rsgen(map)
    possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
    selected = unique(rand(possible_ps, map[2]))
    while length(selected) != map[2]
        push!(selected, rand(possible_ps))
        selected = unique!(selected)
    end
    return RockSamplePOMDP(map_size=(map[1],map[1]), rocks_positions=selected)
end

# default policy
move_east = FunctionPolicy() do b
    return 2
end

maps = [(7, 8), (11, 11), (15, 15)]
# maps = [(7, 8),]
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    fl_bounds = AdaOPS.IndependentBounds(FORollout(move_east), maps[k][2]*10+10, check_terminal=true, consistency_fix_thresh=1e-5)
    flfu_bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    flpu_bounds = AdaOPS.IndependentBounds(FORollout(move_east), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
    # plpu_bounds = AdaOPS.IndependentBounds(POValue(SARSOPSolver(fast=true, timeout=100.0)), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)

    # pomdp = rsgen(maps[k])
    # b0 = initialstate(pomdp)
    # solver = AdaOPSSolver(epsilon_0=0.1,
    #                       bounds=plpu_bounds,
    #                       k_min=maps[k][2],
    #                       zeta=0.5
    #                      )
    # @time p = solve(solver, pomdp)
    # D, extra_info = build_tree_test(p, b0)
    # extra_info_analysis(extra_info)
    # inchrome(D3Tree(D))

    adaops_list = [:default_action=>[move_east,],
                :bounds=>(k==1 ? [fl_bounds, flpu_bounds, flfu_bounds, ] : (k==2 ? [fl_bounds, flfu_bounds] : [fl_bounds])),
                :delta=>[0.1, 0.3, 1.0],
                :grid=>[nothing],
                :k_min=>[maps[k][2]],
                :zeta=>[0.1, 0.3, 0.5],
                :xi=>[0.95, 0.3, 0.1],
                ]
    adaops_list_labels = [["MoveEast",],
                        (k==1 ? ["(FO_MoveEast, $(maps[k][2]*10+10))", "(FO_MoveEast, QMDP)", "(FO_MoveEast, MDP)"] : (k==2 ? ["(FO_MoveEast, $(maps[k][2]*10+10))", "(FO_MoveEast, MDP)"] : ["(FO_MoveEast, $(maps[k][2]*10+10))"])),
                        [0.1, 0.3, 1.0],
                        ["NullGrid"],
                        [maps[k][2]],
                        [0.1, 0.3, 0.5],
                        [0.95, 0.3, 0.1],
                        ]

    # For PL-DESPOT
    bounds = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(move_east), maps[k][2]*10+10.0, check_terminal=true)
    bounds_ub = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(move_east), PL_DESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    pldespot_list = [:default_action=>[move_east,], 
                        :bounds=>(k==3 ? [bounds] : [bounds, bounds_ub]),
                        :K=>[100, 300],
                        :lambda=>[0.0, 0.01, 0.1],
                        :C=>[Inf, 10., 20., 30.],
                        :beta=>[0.0, 0.1, 0.3]]
    pldespot_list_labels = [["MoveEast",], 
                        k==3 ? ["(MoveEast, $(maps[k][2]*10+10))"] : ["(MoveEast, $(maps[k][2]*10+10))", "(MoveEast, MDP)"],
                        [100, 300],
                        [0.0, 0.01, 0.1],
                        [Inf, 10., 20., 30.],
                        [0.0, 0.1, 0.3]]

    # For ARDESPOT
    bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), maps[k][2]*10+10.0, check_terminal=true)
    bounds_ub = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    ardespot_list = [:default_action=>[move_east,], 
                        :bounds=>(k==3 ? [bounds] : [bounds, bounds_ub]),
                        :K=>[100, 300],
                        :lambda=>[0.0, 0.01, 0.1],
                    ]
    ardespot_list_labels = [["MoveEast",], 
                            k==3 ? ["(MoveEast, $(maps[k][2]*10+10))"] : ["(MoveEast, $(maps[k][2]*10+10))", "(MoveEast, MDP)"],
                            [100, 300],
                            [0.0, 0.01, 0.1],
                            ]

    # For POMCPOW
    random_value_estimator = FORollout(RandomSolver())
    value_estimator = FORollout(move_east)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator, random_value_estimator],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :enable_action_pw=>[false, true],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEastRollout", "RandomRollout"],
                        [200000,], 
                        [1.0,],
                        [false, true],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        PL_DESPOTSolver=>pldespot_list, 
        DESPOTSolver=>ardespot_list, 
        POMCPOWSolver=>pomcpow_list,
        AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        pldespot_list_labels, 
        ardespot_list_labels,
        pomcpow_list_labels,
        adaops_list_labels,
    ]

    solver_labels = [
        "PL-DESPOT",
        "ARDESPOT",
        "POMCPOW",
        "AdaOPS",
    ]

    number_of_episodes = 1000
    max_steps = 100
    parallel_experiment(number_of_episodes,
                        max_steps,
                        solver_list,
                        solver_labels=solver_labels,
                        solver_list_labels=solver_list_labels,
                        belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                        max_queue_length=300,
                        experiment_label="RS1000$(maps[k])",
                        full_factorial_design=true)do 
                            rsgen(maps[k])
                        end
end
