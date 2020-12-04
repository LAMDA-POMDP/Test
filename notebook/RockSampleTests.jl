# load the packages
import Pkg
Pkg.cd("..")
Pkg.activate(".")

# global variables
max_workers = 10
rock_nums = [15, 11, 8]

# create n workers
using Distributed
addprocs(max_workers, exeflags="--project")

# the algorithms tested in experiments
@everywhere[
    # POMDP related pkgs
    using POMDPs
    using QMDP
    # POMCP
    using POMCPOW
    using BasicPOMCP
    # DESPOT
    push!(LOAD_PATH, "../LB-DESPOT")
    using LBDESPOT # LBDESPOT pkg
    # UCT-DESPOT
    push!(LOAD_PATH, "../UCT-DESPOT")
    using UCTDESPOT # UCT-DESPOT pkg
    # Rocksample pkg
    using RockSample

    using ParallelExp
    using BasicPOMCP
    using POMDPPolicies # For function policy and random policy
    using ParticleFilters
    using BeliefUpdaters # For belief updater 
    using Random
]

Random.seed!(0)
# set up the environment
maps = [(7, 7), (11, 11), (15, 15)]
pomdps = []
for map in maps
    current_rocks = []
    possible_ps = [(i, j) for i in range(1, length=map[1]), j in range(1, length=map[2])]
    selected = rand(1:map[1]*map[2], pop!(rock_nums))
    for pos in selected
        rock = possible_ps[pos]
        push!(current_rocks, rock)
    end
    pomdp = RockSamplePOMDP(map_size=map, rocks_positions=current_rocks)
    push!(pomdps, pomdp)
end

# test algorithms on environments
let 
    k = 1
    for pomdp in pomdps
        println(k)
        # QMDP upper bound
        if k == 1
            qmdp_policy = solve(QMDPSolver(), pomdp)
            @everywhere function qmdp_upper_bound(pomdp, b)
                return value($qmdp_policy, b)
            end
        end

        # default policy
        move_east = FunctionPolicy() do b
            return 2
        end

        # better default policy
        function ind_rank(arr::Vector{Float64}, inds::Vector{Int})
            # Input: arr: an array storing all elements needed
            #        inds: an array storing the indexes of elements need to be ranked
            # Output: an array of ranking for each element indexed by inds
            sort_ind = Vector{Int}(undef, length(inds))
            ind_ranking = Vector{Int}(undef, length(inds))
            sort_ind[1] = 1
            for i in 2:length(inds)
                ind = 1
                for j in i-1:-1:1
                    if arr[inds[i]] < arr[inds[sort_ind[j]]]
                        sort_ind[j+1] = sort_ind[j]
                    else
                        ind = j + 1
                        break
                    end
                end
                sort_ind[ind] = i
            end

            ind_ranking[sort_ind[1]] = 1
            rank = 1
            for i in 2:length(sort_ind)
                if arr[inds[sort_ind[i]]] == arr[inds[sort_ind[i-1]]]
                    ind_ranking[sort_ind[i]] = rank
                else
                    rank += 1
                    ind_ranking[sort_ind[i]] = rank
                end
            end
            return ind_ranking
        end
        to_best = FunctionPolicy() do b 
            if typeof(b) <: RSState 
                s = b 
                good_probs = s.rocks
            else 
                s = rand(b) 
                good_count = zeros(Int, length(s.rocks)) 
                for state in particles(b) 
                    good_count += state.rocks 
                end 
                good_probs = good_count./length(s.rocks)
            end 

            # Calculate the distance between the robot and rocks
            rock_dist = Vector{Float64}(undef, length(s.rocks))
            for i in 1:length(s.rocks)
                pos_diff = pomdp.rocks_positions[i] - s.pos
                rock_dist[i] = sqrt(pos_diff[1]*pos_diff[1] + pos_diff[2]*pos_diff[2])
            end
            # rank the distance
            dist_ranking = ind_rank(rock_dist, [i for i in 1:length(s.rocks)])
            
            # select the rock with the smallest distance s.t. the probability of being a good rock greater than 0.1
            currently_best_rock = 0
            for i in 1:length(s.rocks)
                if good_probs[i] > 0.1
                    currently_best_rock = i
                    break
                end
            end
            # if all rocks are bad with a high probability, go east
            if currently_best_rock == 0
                return 2
            end

            # calculate the distance between the best rock and the current position
            rock_pos = pomdp.rocks_positions[currently_best_rock]
            diff = rock_pos - s.pos 
            dist = sqrt(diff[1]*diff[1] + diff[2]*diff[2])
            # sense the currently best rock when coming close and being not sure about wheather it is a good rock
            if dist < 2 && good_probs[currently_best_rock] < 0.9
                return 5 + currently_best_rock
            end

            # sample
            if dist == 0
                return 5
            end

            # randomly choose an action movint to the best rock
            px = exp(diff[1])/(exp(diff[1])+exp(diff[2]))
            if rand() < px
                if diff[1] == 0
                    return rand([2,4]) # east or west
                elseif sign(diff[1]) == 1
                    return 2 # to est
                else
                    return 4 # to west
                end
            else
                if diff[2] == 0
                    return rand([1,3]) # north or south
                elseif sign(diff[2]) == 1
                    return 1 # to north
                else
                    return 3 # to south
                end
            end
        end

        # For LB-DESPOT
            random_bounds = IndependentBounds(DefaultPolicyLB(RandomPolicy(pomdp)), 40.0, check_terminal=true)
            bounds = IndependentBounds(DefaultPolicyLB(to_best), 40.0, check_terminal=true)
            if k == 1
                bounds_hub = IndependentBounds(DefaultPolicyLB(to_best), qmdp_upper_bound, check_terminal=true)
                lbdespot_list = [:default_action=>[to_best,], 
                                    :bounds=>[bounds, bounds_hub],
                                    :K=>[100, 300],
                                    :beta=>[0.0, 0.3]]
            else
                lbdespot_list = [:default_action=>[to_best,], 
                                    :bounds=>[bounds,],
                                    :K=>[100, 300],
                                    :beta=>[0.0, 0.3]]
            end
            # lbdespot_list2 = [:default_action=>[to_best,], 
            #                     :bounds=>[random_bounds],
            #                     :K=>[100],
            #                     :beta=>[0.3]]
            # lbdespot_list3 = [:default_action=>[to_best,], 
            #                     :bounds=>[bounds_hub],
            #                     :K=>[100],
            #                     :beta=>[0.5]]
        # For UCT-DESPOT
            random_rollout_policy = RandomPolicy(pomdp)
            rollout_policy = to_best
            uctdespot_list = [:default_action=>[to_best,],
                                :rollout_policy=>[rollout_policy],
                                :max_trials=>[100000,],
                                :K=>[300, 1000, 3000],
                                :m=>[30, 100],
                                :c=>[10., 30., 3.]]
            # uctdespot_list = [:default_action=>[RandomPolicy(pomdp),],
            #                     :rollout_policy=>[random_rollout_policy],
            #                     :max_trials=>[100000,],
            #                     :K=>[300, 100, 500],
            #                     :m=>[50, 30],
            #                     :c=>[1.,10,]]
        # For POMCPOW
            # random_value_estimator = FORollout(RandomPolicy(pomdp))
            # value_estimator = FORollout(to_best)
            # pomcpow_list = [:default_action=>[RandomPolicy(pomdp),],
            #                     :estimate_value=>[random_value_estimator],
            #                     :tree_queries=>[200000,], 
            #                     :max_time=>[1.0,],
            #                     :criterion=>[MaxUCB(10.),]]

        # Solver list
            solver_list = [
                LB_DESPOTSolver=>lbdespot_list, 
                # LB_DESPOTSolver=>lbdespot_list2, 
                # LB_DESPOTSolver=>lbdespot_list3, 
                UCT_DESPOTSolver=>uctdespot_list, 
                # POMCPOWSolver=>pomcpow_list
            ]

        number_of_episodes = 2
        max_steps = 300
        # Pkg.cd("notebook")
        dfs = parallel_experiment(pomdp,
                                number_of_episodes,
                                max_steps, solver_list,
                                full_factorial_design=false)
        CSV.write("RockSample_DESPOT_$k.csv", dfs[1])
        # CSV.write("RockSample_DESPOT2_$k.csv", dfs[2])
        # CSV.write("RockSample_DESPOT3_$k.csv", dfs[3])
        CSV.write("RockSample_UCT_DESPOT_$k.csv", dfs[2])
        # CSV.write("RockSample_POMCP_$k.csv", dfs[3])
        k += 1
    end
end