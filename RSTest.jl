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
struct MoveEast<:Policy end
@everywhere POMDPs.action(p::MoveEast, b) = 2
move_east = MoveEast()
struct SampleNearRock <: Policy
    m::RockSamplePOMDP
end
struct SampleNearRockSolver <: Solver end
@everywhere POMDPs.solve(solver::SampleNearRockSolver, pomdp::RockSamplePOMDP) = SampleNearRock(pomdp)
@everywhere function POMDPs.action(p::SampleNearRock, s::RSState)
    is_good = s.rocks
    for i in 1:length(s.rocks)
        rock_dist = norm(p.m.rocks_positions[i] - s.pos, 2)
        # Suppose that when distance is less than 4, the sensing is effective
        if is_good[i] == 1
            if rock_dist == 0
                # Sample the good rock
                return RockSample.BASIC_ACTIONS_DICT[:sample]
            else
                # Move to the good rock
                diff = p.m.rocks_positions[currently_best_rock] - s.pos 
                if diff[1] != 0 && diff[2] != 0
                    diff[rand([1,2])] = 0
                end
                diff = RSPos(sign.(diff))
                return findfirst(x->(x==diff), RockSample.ACTION_DIRS)
            end
        end
    end
    return RockSample.BASIC_ACTIONS_DICT[:east]
end
@everywhere function POMDPs.action(p::SampleNearRock, b::AbstractParticleBelief)
    s = rand(b) 
    good_probs = zeros(Float64, length(s.rocks)) 
    for (i, s) in enumerate(particles(b))
        good_probs += s.rocks .* weight(b, i)
    end 
    good_probs = good_probs ./ weight_sum(b)
    # Calculate the distance between the robot and rocks
    for i in 1:length(s.rocks)
        rock_dist = norm(p.m.rocks_positions[i] - s.pos, 2)
        # Suppose that when distance is less than 4, the sensing is effective
        if rock_dist < 4 && good_probs[i] > 0.3
            if good_probs > 0.9
                if rock_dist == 0
                    # Sample the good rock
                    return RockSample.BASIC_ACTIONS_DICT[:sample]
                else
                    # Move to the good rock
                    diff = p.m.rocks_positions[currently_best_rock] - s.pos 
                    if diff[1] != 0 && diff[2] != 0
                        diff[rand([1,2])] = 0
                    end
                    diff = RSPos(sign.(diff))
                    return findfirst(x->(x==diff), RockSample.ACTION_DIRS)
                end
            else
                # Probe the potentially good rock.
                return 5 + i
            end
        end
    end
    return RockSample.BASIC_ACTIONS_DICT[:east]
end
sample_near_rock = SampleNearRockSolver()

maps = [(7, 8), (11, 11), (15, 15)]
# maps = [(7, 8),]
for k in 1:length(maps)
    println(k)

    # For AdaOPS
    convert(s::RSState, pomdp::RockSamplePOMDP) = SVector(sum(s.rocks))
    grid = StateGrid(convert, range(1, stop=maps[k][2], length=maps[k][2])[2:end])

    fu_bounds = AdaOPS.IndependentBounds(FORollout(move_east), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    splfu_bounds = AdaOPS.IndependentBounds(SemiPORollout(sample_near_rock), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

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

    adaops_list = [:default_action=>[sample_near_rock,],
                :bounds=>[fu_bounds, splfu_bounds],
                :delta=>[0.1, 0.3],
                :grid=>[nothing, grid],
                :m_init=>[30, 50],
                :zeta=>[0.1, 0.3],
                :xi=>[0.1, 0.3, 0.95]
                ]
    adaops_list_labels = [["MoveEast",],
                        ["(MoveEast, MDP)", "(SampleNearRock, MDP)"],
                        [0.1, 0.3],
                        ["NullGrid", "FullGrid"],
                        [30, 50],
                        [0.1, 0.3],
                        [0.1, 0.3, 0.95],
                        ]

    # For PL-DESPOT
    bounds = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(move_east), PL_DESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    plbounds = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(sample_near_rock), PL_DESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    pldespot_list = [:default_action=>[sample_near_rock,], 
                        :bounds=>([bounds, plbounds]),
                        :K=>[100, 300],
                        :lambda=>[0.0, 0.01, 0.1],
                        :C=>[Inf, 10., 20., 30.],
                        :beta=>[0.0, 0.1, 0.3]]
    pldespot_list_labels = [["SampleNearRock",], 
                        ["(MoveEast, MDP)", "(SampleNearRock, MDP)"],
                        [100, 300],
                        [0.0, 0.01, 0.1],
                        [Inf, 10., 20., 30.],
                        [0.0, 0.1, 0.3]]

    # For ARDESPOT
    bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
    plbounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(sample_near_rock), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

    ardespot_list = [:default_action=>[sample_near_rock,], 
                        :bounds=>[bounds, plbounds],
                        :K=>(k == 1 ? [100] : [100, 300]),
                        :lambda=>(k== 1 ? [0.0] : [0.0, 0.01, 0.1]),
                    ]
    ardespot_list_labels = [["SampleNearRock",], 
                            ["(MoveEast, MDP)", "(SampleNearRock, MDP)"],
                            k == 1 ? [100] : [100, 300],
                            k== 1 ? [0.0] : [0.0, 0.01, 0.1],
                            ]

    # For POMCPOW
    random_value_estimator = FORollout(RandomSolver())
    value_estimator = FORollout(move_east)
    sample_value_estimator = FORollout(sample_near_rock)
    pomcpow_list = [:default_action=>[move_east,],
                        :estimate_value=>[value_estimator, random_value_estimator, sample_value_estimator],
                        :tree_queries=>[200000,], 
                        :max_time=>[1.0,],
                        :enable_action_pw=>[false, true],
                        :criterion=>[MaxUCB(10.),]]
    pomcpow_list_labels = [["MoveEast",],
                        ["MoveEastRollout", "RandomRollout", "SampleNearRockRollout"],
                        [200000,], 
                        [1.0,],
                        [false, true],
                        [MaxUCB(10.),]]

    # Solver list
    solver_list = [
        # PL_DESPOTSolver=>pldespot_list, 
        DESPOTSolver=>ardespot_list, 
        # POMCPOWSolver=>pomcpow_list,
        AdaOPSSolver=>adaops_list,
    ]
    solver_list_labels = [
        # pldespot_list_labels, 
        ardespot_list_labels,
        # pomcpow_list_labels,
        adaops_list_labels,
    ]

    solver_labels = [
        # "PL-DESPOT",
        "ARDESPOT",
        # "POMCPOW",
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
