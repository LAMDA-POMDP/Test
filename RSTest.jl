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
@everywhere struct MoveEast<:Policy end
@everywhere POMDPs.action(p::MoveEast, b) = 2
move_east = MoveEast()

maps = [(7, 8), (11, 11), (15, 15)]
# maps = [(7, 8),]
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    @everywhere convert(s::RSState, pomdp::RockSamplePOMDP) = SVector(sum(s.rocks))
    grid = StateGrid(convert, range(1, stop=maps[k][2], length=maps[k][2])[2:end])

    bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    adaops_list = [:default_action=>[move_east,],
                :bounds=>[bounds,],
                :delta=>[0.1, 0.3],
                :grid=>[grid],
                :m_init=>[30],
                :zeta=>[0.2, 0.3],
                :bounds_warnings=>[false]
                ]
    adaops_list_labels = [["MoveEast",],
                        ["(MoveEast, MDP)",],
                        [0.1, 0.3],
                        ["FullGrid"],
                        [30],
                        [0.2, 0.3],
                        [false]
                        ]

    # For ARDESPOT
    bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    ardespot_list = [:default_action=>[move_east,], 
                        :bounds=>[bounds,],
                        :K=>[100],
                        :bounds_warnings=>[false]
                    ]
    ardespot_list_labels = [["MoveEast",], 
                            ["(MoveEast, MDP)",],
                            [100],
                            [false]
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
        DESPOTSolver=>ardespot_list, 
        POMCPOWSolver=>pomcpow_list,
        AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        ardespot_list_labels,
        pomcpow_list_labels,
        adaops_list_labels,
    ]

    solver_labels = [
        "ARDESPOT",
        "POMCPOW",
        "AdaOPS",
    ]

    episodes_per_domain = 100
    max_steps = 100
    parallel_experiment(episodes_per_domain,
                        max_steps,
                        solver_list,
                        num_of_domains=10,
                        solver_labels=solver_labels,
                        solver_list_labels=solver_list_labels,
                        belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                        max_queue_length=300,
                        experiment_label="RS10*100$(maps[k])",
                        full_factorial_design=true)do 
                            rsgen(maps[k])
                        end
end