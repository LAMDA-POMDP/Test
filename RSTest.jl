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

# maps = [(7, 8), (11, 11), (15, 15)]
maps = [(7, 8),]
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    fl_bounds = AdaOPS.IndependentBounds(FORollout(move_east), 10, check_terminal=true, consistency_fix_thresh=1e-5)
    flpu_bounds = AdaOPS.IndependentBounds(FORollout(move_east), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
    flfu_bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)

    adaops_list = [:default_action=>[move_east,],
                :bounds=>[fl_bounds, flpu_bounds, flfu_bounds],
                :delta=>[0.1, 0.3, 1.0],
                :grid=>[nothing],
                :k_min=>[maps[k][2]],
                :zeta=>[0.3, 0.4, 0.5]
                ]
    adaops_list_labels = [["MoveEast",],
                        ["(FO_MoveEast, 10)", "(FO_MoveEast, QMDP)", "(FO_MoveEast, MDP)"],
                        [0.1, 0.3, 1.0],
                        ["NullGrid"],
                        [maps[k][2]],
                        [0.3, 0.4, 0.5]
                        ]

    # For PL-DESPOT
    bounds = PL_DESPOT.IndependentBounds(DefaultPolicyLB(move_east), 40.0, check_terminal=true)
    bounds_ub = PL_DESPOT.IndependentBounds(DefaultPolicyLB(move_east), FullyObservableValueUB(ValueIterationSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
    pldespot_list = [:default_action=>[move_east,], 
                        :bounds=>[bounds,],
                        :K=>[100, 300],
                        :C=>[Inf, 10., 20., 30.],
                        :beta=>[0.0, 0.1, 0.3]]
    pldespot_list_labels = [["MoveEast",], 
                        ["MoveEastLB_FixedUB",],
                        [100, 300],
                        [Inf, 10., 20., 30.],
                        [0.0, 0.1, 0.3]]
    # end
    # For POMCPOW
    # random_value_estimator = FORollout(RandomPolicy(pomdp))
    value_estimator = FORollout(move_east)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEastRollout"],
                        [200000,], 
                        [1.0,],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        # PL_DESPOTSolver=>pldespot_list, 
        # POMCPOWSolver=>pomcpow_list,
        AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        # pldespot_list_labels, 
        # pomcpow_list_labels,
        adaops_list_labels,
    ]

    solver_labels = [
        # "PL-DESPOT",
        # "POMCPOW",
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
                        experiment_label="RS1000$(maps[k])",
                        full_factorial_design=true)do 
                            rsgen(maps[k])
                        end
end
