@everywhere using RockSample

# maps = [(7, 8), (11, 11), (15, 15)]
maps = [(11, 11), (15, 15)]
# maps = [(15, 15),]
# maps = [(11, 11),]
move_east = RSExitSolver()
qmdp = RSQMDPSolver()
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    # bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    bounds = AdaOPS.IndependentBounds(FOValue(move_east), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

    adaops_list = [
                :bounds=>[bounds,],
                :delta=>[0.1],
                :m_min=>[100],
                :m_max=>[200, 300],
                :num_b=>[250_000],
                ]
    adaops_list_labels = [
                        ["(MoveEast, QMDP)",],
                        [0.1],
                        [100],
                        [200, 300],
                        [250_000],
                        ]

    # For BSDESPOT
    qmdp_bounds = BSDESPOT.IndependentBounds(BSDESPOT.DefaultPolicyLB(move_east), qmdp, check_terminal=true, consistency_fix_thresh=1e-5)

    bsdespot_list = [
        :default_action=>[move_east],
        :bounds=>[qmdp_bounds],
        :K=>[100],
        :beta=>[0],
        :adjust_zeta=>[(d,k)->1.0, (d,k)->lower_bounded_zeta(d,k,0.9)],
        :impl=>[:gap, :val],
        :C=>[4]
        ]

    bsdespot_list_labels = [
        ["move_east"],
        ["(MoveEast, QMDP)"],
        [100],
        [0],
        ["fixed zeta", "lower bounded zeta"],
        [:gap, :value],
        [4,]
        ]

    # For POMCPOW
    value_estimator = FOValue(move_east)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator,],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :enable_action_pw=>[false,],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEast",],
                        [200000,], 
                        [1.0,],
                        [false,],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        BS_DESPOTSolver=>bsdespot_list, 
        # POMCPOWSolver=>pomcpow_list,
        # AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        bsdespot_list_labels,
        # pomcpow_list_labels,
        # adaops_list_labels,
    ]

    solver_labels = [
        "BSDESPOT",
        # "POMCPOW",
        # "AdaOPS",
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
                        max_queue_length=4,
                        domain_queue_length=5,
                        experiment_label="RS100_10$(maps[k])",
                        full_factorial_design=true)do 
                            RockSamplePOMDP(maps[k]...)
                        end
end