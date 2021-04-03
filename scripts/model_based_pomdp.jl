using Pkg
Pkg.activate(".")
using POMDPs
using SARSOP
using POMDPModels
using POMDPSimulators
using DeepQLearning
using Flux
using POMDPPolicies
using BeliefUpdaters
using POMDPModelTools
using ProgressMeter

m = TigerPOMDP()
# p = solve(SARSOPSolver(), m)

POMDPs.convert_o(::Type{Array{Float32, N} where N}, b::BoolDistribution, ::TigerPOMDP) = [0f0, 0f0]
input_dims = reduce(*, size(convert_o(Vector{Float32}, first(observations(m)), m))) + reduce(*, size(convert_a(Vector{Float32}, first(actions(m)), m)))
model = Chain(x->flattenbatch(x), LSTM(input_dims, 32), Dense(32, 32), Dense(32, length(actions(m))))
max_steps = 1e8
exploration = EpsGreedyPolicy(m, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2))
solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=max_steps,
                            learning_rate=0.0001, exploration_policy=exploration,
                            log_freq=500, target_update_freq = 1000,
                            recurrence=true,trace_length=100, double_q=true, dueling=true, max_episode_length=100)

policy = solve(solver, m)

N = 100000
rsum = 0.0
@showprogress for i in 1:N
    sim = RolloutSimulator(max_steps=T)
    global rsum += simulate(sim, m, policy, PreviousObservationUpdater())
end
@show rsum/N
