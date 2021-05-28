@everywhere using POMDPModels


# Belief updater

pomdp = LightDark1D()

POMDPs.actionindex(m::LightDark1D, a::Int) = a + 2

interp = LocalGIFunctionApproximator(RectangleGrid(range(-1, stop=1, length=3), range(-100, stop=100, length=401)))
                     
approx_mdp = solve(LocalApproximationValueIterationSolver(
                                interp,
                                verbose=true,
                                max_iterations=1000,
                                is_mdp_generative=true,
                                n_generative_samples=1000),
                            pomdp)

approx_random = solve(LocalApproximationRandomSolver(
                                interp,
                                verbose=true,
                                max_iterations=1000,
                                is_mdp_generative=true,
                                n_generative_samples=1000),
                            pomdp)


random = solve(RandomSolver(), pomdp)

# For AdaOPS
@everywhere Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState) = SVector{1,Float64}(s.y)
grid = StateGrid(range(-10, stop=15, length=26))
bounds = AdaOPS.IndependentBounds(FOValue(approx_random), FOValue(approx_mdp), check_terminal=true)
fixed_bounds = AdaOPS.IndependentBounds(FORollout(random), pomdp.correct_r, check_terminal=true)
#
adaops_list = [
                :bounds=>[bounds],
                :delta=>[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                :m_min=>[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                :grid=>[grid],
		    ]

adaops_list_labels = [
                ["Random, MDP",],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                ["FullGrid"],
		    ]

# ARDESPOT
bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(random), pomdp.correct_r, check_terminal=true)
ardespot_list = [
                :default_action=>[random],
                :bounds=>[bounds],
                :lambda=>[0.1],
                :K=>[30],
                :bounds_warnings=>[false,],
                ]
ardespot_list_labels = [
                ["Random"],
                ["(Random, $(pomdp.correct_r))",],
                [0.1],
                [30],
                [false],
                ]

# For POMCPOW
pomcpow_list = [
                :estimate_value=>[FOValue(approx_mdp)],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(10.0),],
                :max_depth=>[20],
                :k_observation=>[4.0],
                :alpha_observation=>[0.03],
                ]

pomcpow_list_labels = [
                        ["MDP"],
                        [100000,], 
                        [1.0,], 
                        ["UCB 10",],
                        [20],
                        [4.0],
                        [0.03]
                        ]

# Solver list
solver_list = [
                AdaOPSSolver=>adaops_list, 
                #DESPOTSolver=>ardespot_list,
                #POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    adaops_list_labels,
                    #ardespot_list_labels,
                    #pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                #"ARDESPOT",
                #"POMCPOW",
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
                    max_queue_length=100,
                    belief_updater=(m)->BasicParticleFilter(m, LowVarianceResampler(30000), 30000),
                    experiment_label="LightDark_4",
                    full_factorial_design=true)
