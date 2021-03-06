using Pkg
Pkg.activate(".")

using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using Random
using POMCPOW
using Multilane
using AutoViz
using Reel
using Cairo
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images

if !(@isdefined hist) || hist === nothing
    behaviors = standard_uniform(correlation=0.75)
    pp = PhysicalParam(4, lane_length=120.0)
    dmodel = NoCrashIDMMOBILModel(10, pp,
                                behaviors=behaviors,
                                p_appear=1.0,
                                lane_terminate=true,
                                max_dist=1000.0
                                )
    rmodel = SuccessReward(lambda=1.0, speed_thresh=20.0, lane_change_cost=0.05)
    pomdp = NoCrashPOMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, 0.95, true);

    rng = MersenneTwister(15)

    s = rand(rng, initialstate(pomdp))

    @show n_iters = 1000
    @show max_time = Inf
    @show max_depth = 40
    @show val = SimpleSolver()

    wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
    solver = POMCPOWSolver(tree_queries=n_iters,
                            criterion=MaxUCB(2.0),
                            max_depth=max_depth,
                            max_time=max_time,
                            enable_action_pw=false,
                            k_observation=2.0,
                            alpha_observation=1/20,
                            estimate_value=FORollout(val),
                            # estimate_value=val,
                            check_repeat_obs=false,
                            node_sr_belief_updater=BehaviorPOWFilter(wup),
                            rng=MersenneTwister(7),
                            tree_in_info=true,
                            default_action=MLAction(0.0, 0.0) # maintain at the end
                            )
    planner = solve(solver, pomdp)

    sim = HistoryRecorder(rng=rng, max_steps=100, show_progress=true) # initialize a random number generator

    up = BehaviorParticleUpdater(pomdp, 1000, 0.1, 0.1, wup, MersenneTwister(50000))

    hist = POMDPs.simulate(sim, pomdp, planner, up, MLPhysicalState(s), s)

    @show discounted_reward(hist)
end

fpstep = 6
sh = state_hist(hist)
surfaces = []
surfdir = "img/"
frames = Frames(MIME("image/png"), fps=fpstep/pp.dt)
@showprogress for (s, ai, r, sp) in eachstep(hist, "s, action_info, r, sp")
    tree = get(ai, :tree, nothing)
    rollouts = make_rollouts(planner, tree)
    nwr = NodeWithRollouts(POWTreeObsNode(tree, 1), rollouts)
    for t in range(0.0, stop=1.0, length=fpstep)
        if s.t ≈ 0.0 || t > 0.0
            is = interp_state(s, sp, t)
            # fname = joinpath(surfdir, string(length(surfaces)+1)*".svg")
            # surf = CairoSVGSurface(fname, AutoViz.DEFAULT_CANVAS_WIDTH, AutoViz.DEFAULT_CANVAS_HEIGHT)
            # push!(surfaces, visualize(pomdp, is, r, tree=nwr, surface=surf))
            push!(surfaces, visualize(pomdp, is, r, tree=nwr))
            push!(frames, last(surfaces))
        end
    end
end

# simulate a little farther with another policy
sdm = deepcopy(dmodel)
sdm.lane_terminate = false
sdm.max_dist = 1200 # give the car some extra room at the end
spomdp = NoCrashPOMDP{typeof(rmodel), typeof(dmodel.behaviors)}(sdm, rmodel, 0.95, true);
s = last(state_hist(hist))
s.terminal = nothing
policy = Multilane.BehaviorPolicy(spomdp, Multilane.NORMAL, true, rng)
bonus_hist = POMDPs.simulate(sim, spomdp, policy, up, MLPhysicalState(s), s)
@showprogress for (s, r, sp) in eachstep(bonus_hist, "s,r,sp")
    for t in range(0.0, stop=1.0, length=fpstep)
        if t > 0.0
            is = interp_state(s, sp, t)
            push!(frames, visualize(spomdp, is, r))
        end
    end
end

gifname = "img/Multilane.gif"
write(gifname, frames)
# run(`xdg-open $gifname`)
# gif = load(gifname)
# imshow(gif)

# @show fname = "img/selected_frame.svg"
# selected = nothing
# for (s, r, sp, info) in eachstep(hist, "s, r, sp, action_info")
#     tree = info === nothing ? nothing : get(info, :tree, nothing)
#     rollouts = make_rollouts(planner, tree)
#     nwr = NodeWithRollouts(POWTreeObsNode(tree, 1), rollouts)
#     # if s.t <= 19.5 && sp.t > 19.5
#     if s.t <= 14.5 && sp.t > 14.5
#         is = interp_state(s, sp, (19.5-s.t)/(sp.t-s.t))
#         surf = CairoSVGSurface(fname, AutoViz.DEFAULT_CANVAS_WIDTH, AutoViz.DEFAULT_CANVAS_HEIGHT)
#         global selected = visualize(pomdp, is, r, tree=nwr, surface=surf)
#     end
# end
# finish(selected)