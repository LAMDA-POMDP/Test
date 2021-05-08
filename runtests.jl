using LBDESPOT
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters

include("memorizing_rng.jl")
include("independent_bounds.jl")

pomdp = BabyPOMDP()
pomdp.discount = 1.0

K = 10
rng = MersenneTwister(14)
rs = MemorizingSource(K, 50, rng)
Random.seed!(rs, 10)
b_0 = initialstate_distribution(pomdp)
scenarios = [i=>rand(rng, b_0) for i in 1:K]
o = false
b = ScenarioBelief(scenarios, rs, 0, o)

@testset "Branching Simulation" begin
    pol = FeedWhenCrying()
    r1 = LBDESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->0.0)
    r2 = LBDESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->0.0)
    @test r1 == r2
    tval = 7.0
    r3 = LBDESPOT.branching_sim(pomdp, pol, b, 10, (m,x)->tval)
    @test r3 == r2 + tval*length(b.scenarios)
end

scenarios = [1=>rand(rng, b_0)]
b = ScenarioBelief(scenarios, rs, 0, false)

@testset "Rollout" begin
    pol = FeedWhenCrying()
    r1 = LBDESPOT.rollout(pomdp, pol, b, 10, (m,x)->0.0)
    r2 = LBDESPOT.rollout(pomdp, pol, b, 10, (m,x)->0.0)
    @test r1 == r2
    tval = 7.0
    r3 = LBDESPOT.rollout(pomdp, pol, b, 10, (m,x)->tval)
    @test r3 == r2 + tval
end

# AbstractParticleBelief interface
# @testset "Abstract Particle Belief Interface" begin
#     @test n_particles(b) == 1
#     s = particle(b,1)
#     @test rand(rng, b) == s
#     @test pdf(b, rand(rng, b_0)) == 1
#     sup = support(b)
#     @test length(sup) == 1
#     @test first(sup) == s
#     @test mode(b) == s
#     @test mean(b) == s
#     @test first(particles(b)) == s
#     @test first(weights(b)) == 1.0
#     @test first(weighted_particles(b)) == (s => 1.0)
#     @test weight_sum(b) == 1.0
#     @test weight(b, 1) == 1.0
#     @test currentobs(b) == o
#     @test_deprecated previous_obs(b)
#     @test history(b)[end].o == o
# end

pomdp = BabyPOMDP()

# constant bounds
bds = (reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = LB_DESPOTSolver(bounds=bds, beta=0.01)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=5)
println("\nConstant bound:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# constant bounds value implementation
bds = (reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = LB_DESPOTSolver(bounds=bds, beta=0.01, impl=:val)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=5)
println("\nConstant bound with value implementation:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# constant bounds value implementation
bds = (reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = LB_DESPOTSolver(bounds=bds, beta=0.01, impl=:prob)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=5)
println("\nConstant bound with probability implementation:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# policy lower bound
bds = IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0)
solver = LB_DESPOTSolver(bounds=bds)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=5)
println("\nPolicy lower bound:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# policy lower bound with final value
fv(m::BabyPOMDP, x) = reward(m, true, false)/(1-discount(m))
bds = IndependentBounds(DefaultPolicyLB(FeedWhenCrying(), final_value=fv), 0.0)
solver = LB_DESPOTSolver(bounds=bds)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=5)
println("\nPolicy lower bound with final value:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))\n")

# Type stability
pomdp = BabyPOMDP()
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = LB_DESPOTSolver(epsilon_0=0.1,
                      bounds=bds,
                      rng=MersenneTwister(4)
                     )
p = solve(solver, pomdp)

b0 = initialstate_distribution(pomdp)
D,_ = @inferred LBDESPOT.build_despot(p, b0)
@inferred LBDESPOT.explore!(D, 1, p)
@inferred LBDESPOT.expand!(D, length(D.children), p)
@inferred LBDESPOT.prune!(D, 1, p)
@inferred LBDESPOT.find_blocker(D, length(D.children), p)
@inferred LBDESPOT.make_default!(D, length(D.children))
@inferred LBDESPOT.backup!(D, 1, p)
@inferred LBDESPOT.next_best(D, 1, p)
@inferred LBDESPOT.excess_uncertainty(D, 1, p)
@inferred action(p, b0)

include("random_2.jl")

bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
rng = MersenneTwister(4)
solver = LB_DESPOTSolver(epsilon_0=0.1,
                      bounds=bds,
                      rng=rng,
                      random_source=MemorizingSource(500, 90, rng),
                      tree_in_info=true
                     )
p = solve(solver, pomdp)
a = action(p, initialstate_distribution(pomdp))

# visualization
println("\n Constant Bound Tree:\n")
show(stdout, MIME("text/plain"), D)
a, info = action_info(p, initialstate_distribution(pomdp))
show(stdout, MIME("text/plain"), info[:tree])

# from README:
println("\nTigerPOMDP in README:\n")
using POMDPs, POMDPModels, POMDPSimulators, LBDESPOT

pomdp = TigerPOMDP()

solver = LB_DESPOTSolver(bounds=(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
