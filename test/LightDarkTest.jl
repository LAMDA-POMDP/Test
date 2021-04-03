@everywhere using POMDPModels


# Belief updater

pomdp = LightDark1D()

random = solve(RandomSolver(), pomdp)

# For AdaOPS
@everywhere Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState) = SVector{1,Float64}(s.y)
grid = StateGrid(range(-10, stop=15, length=26))
bounds = AdaOPS.IndependentBounds(FORollout(random), pomdp.correct_r, check_terminal=true)
adaops_list = [
                :bounds=>[bounds],
                :delta=>[1.0],
                :m_min=>[10],
                :grid=>[grid],
		    ]

adaops_list_labels = [
                ["Random, $(pomdp.correct_r)"],
                [1.0],
                [10],
                ["FullGrid"],
		    ]
# ARDESPOT
bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(random), pomdp.correct_r, check_terminal=true)
ardespot_list = [
                :default_action=>[random],
                :bounds=>[bounds],
                :lambda=>[0.0],
                :K=>[100],
                :bounds_warnings=>[false,],
                ]
ardespot_list_labels = [
                ["Random"],
                ["(Random, $(pomdp.correct_r))",],
                [0.0],
                [100],
                [false],
                ]

# For POMCPOW
pomcpow_list = [
                :estimate_value=>[FORollout(random)],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(30.0), MaxUCB(15.0), MaxUCB(45.0)],
                :max_depth=>[20],
                :k_observation=>[1.0],
                :alpha_observation=>[1/15.0, 1/45]
                ]

pomcpow_list_labels = [
                        ["Random"],
                        [100000,], 
                        [1.0,], 
                        ["UCB 30", "UCB 15", "UCB 45"],
                        [20],
                        [1.0],
                        [1/15.0, 1/45]
                        ]

# Solver list
solver_list = [
                AdaOPSSolver=>adaops_list, 
                # DESPOTSolver=>ardespot_list,
                # POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    adaops_list_labels, 
                    # ardespot_list_labels,
                    # pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                # "ARDESPOT",
                # "POMCPOW",
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
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    experiment_label="LightDark",
                    full_factorial_design=true)
