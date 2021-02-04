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

@everywhere function adjust1(l, d, k)
    1
end

@everywhere function adjust2(l, d, k)
    max(l, 1 - (0.2*k + 0.2*(1-d)))
end

# default policy
@everywhere struct MoveEast<:Policy end
@everywhere POMDPs.action(p::MoveEast, b) = 2
move_east = MoveEast()

#maps = [(7, 8), (11, 11), (15, 15)]
# maps = [(15, 15),]
maps = [(11, 11),]
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    # bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    bounds = AdaOPS.IndependentBounds(FORollout(move_east), POValue(QMDPSolver(max_iterations=1000, verbose=true)), check_terminal=true, consistency_fix_thresh=1e-5)

    adaops_list = [
                :default_action=>[move_east,],
                :bounds=>[bounds,],
                :delta=>[0.01, 0.015, 0.02],
                :m_init=>[30],
                :sigma=>[2.0],
                :zeta=>[0.02, 0.03, 0.04, 0.05],
                :bounds_warnings=>[false],
                ]
    adaops_list_labels = [
                        ["MoveEast",],
                        ["(MoveEast, QMDP)",],
                        [0.01, 0.015, 0.02],
                        [30],
                        [2.0],
                        [0.02, 0.03, 0.04, 0.05],
                        [false],
                        ]
    # For BSDESPOT
    qmdp_bounds = BS_DESPOT.IndependentBounds(BS_DESPOT.DefaultPolicyLB(move_east), QMDPSolver(max_iterations=1000, verbose=true), check_terminal=true, consistency_fix_thresh=1e-5)

    bsdespot_list = [
        :default_action=>[move_east],
        :bounds=>[qmdp_bounds],
        :K=>[100],
        :beta=>[0, 0.3],
        :adjust_zeta=>[adjust1, adjust2],
        :C=>[4]
        ]

    bsdespot_list_labels = [
        ["move_east"],
        ["(MoveEast, QMDP)"],
        [100,],
        [0, 0.3],
        ["adjust_zeta1", "adjust_zeta2"],
        [4,]
        ]

    # For POMCPOW
    value_estimator = FORollout(move_east)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator,],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :enable_action_pw=>[false,],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEastRollout",],
                        [200000,], 
                        [1.0,],
                        [false,],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        BS_DESPOTSolver=>bsdespot_list, 
        POMCPOWSolver=>pomcpow_list,
        AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        bsdespot_list_labels,
        pomcpow_list_labels,
        adaops_list_labels,
    ]

    solver_labels = [
        "BSDESPOT",
        "POMCPOW",
        "AdaOPS",
    ]

    episodes_per_domain = 10
    max_steps = 100
    parallel_experiment(episodes_per_domain,
                        max_steps,
                        solver_list,
                        num_of_domains=100,
                        solver_labels=solver_labels,
                        solver_list_labels=solver_list_labels,
                        belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                        max_queue_length=320,
                        domain_queue_length=5,
                        experiment_label="RS100*10$(maps[k])",
                        full_factorial_design=true)do 
                            rsgen(maps[k])
                        end
end