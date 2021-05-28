@everywhere using RockSample

# maps = [(7, 8), (11, 11), (15, 15)]
maps = [(15, 15)]
# maps = [(15, 15),]
# maps = [(11, 11),]
move_east = RSExitSolver()
qmdp = RSQMDPSolver()
@everywhere POMDPs.action(p::AlphaVectorPolicy, s::RSState) = action(p, ParticleCollection([s]))
@everywhere POMDPs.value(p::AlphaVectorPolicy, s::RSState) = action(p, ParticleCollection([s]))


for k in 1:length(maps)
    println(k)

    # For AdaOPS
    # bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    ada_blpu_bounds = AdaOPS.IndependentBounds(FOValue(move_east), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

    adaops_list = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.1, 0.3, 1.0],
            :m_min=>[10, 30, 100],
            :bounds_warnings=>[true],
            ]
    adaops_list_labels = [
                    ["(MoveEast, QMDP)"],
                    [0.1, 0.3, 1.0],
                    [10, 30, 100],
                    [true],
                    ]

    # For ARDESPOT
    blpu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), ARDESPOT.FullyObservableValueUB(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

    ardespot_list = [
                    :default_action=>[move_east,], 
                    :bounds=>[blpu_bounds],
                    :K=>[30, 100, 300],
                    :lambda=>[0.0, 0.001, 0.01, 0.1],
                    :bounds_warnings=>[false],
                ]
    ardespot_list_labels = [
                        ["MoveEast",], 
                        ["(MoveEast, QMDP)"],
                        [30, 100, 300],
                        [0.0, 0.001, 0.01, 0.1],
                        [false],
                        ]

    # For POMCPOW
    value_estimator = FOValue(move_east)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator,],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :enable_action_pw=>[false,],
	        :k_observation=>[1.,2.,4.,8.,],
                        :alpha_observation=>[0.01,0.03,0.1,0.3,1.0],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEast",],
                        [200000,], 
                        [1.0,],
                        [false,],
	        [1.,2.,4.,8.,],
	        [0.01,0.03,0.1,0.3,1.0],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        #DESPOTSolver=>ardespot_list,
        #AdaOPSSolver=>adaops_list,
        POMCPOWSolver=>pomcpow_list,
        # QMDPSolver=>[:max_iterations=>[1000,]],
        # FuncSolver=>[:func=>[move_towards,]],
    ]

    solver_list_labels = [
        #ardespot_list_labels,
        #adaops_list_labels,
        pomcpow_list_labels,
    ]

    solver_labels = [
        #"ARDESPOT",
        #"AdaOPS",
        "POMCPOW",
    ]

    episodes_per_domain = 10
    max_steps = 100
    parallel_experiment(episodes_per_domain,
                        max_steps,
                        solver_list,
                        num_of_domains=100,
                        solver_labels=solver_labels,
                        solver_list_labels=solver_list_labels,
                        belief_updater=(m)->BasicParticleFilter(m, LowVarianceResampler(30000), 30000),
                        max_queue_length=24,
                        domain_queue_length=1,
                        experiment_label="RS100_10$(maps[k])",
                        full_factorial_design=true)do 
                            RockSamplePOMDP(maps[k]...)
                        end
end