# Training SVMs by Measuring the Distance Between Two Shapes

*Notes on a geometric SVM solver: what the Triangle Algorithm does, the three changes that made it fast and provable, and the benchmark run that told us where it stops being useful.*

---

I first came across the geometric view of support vector machines through Bahman Kalantari's work on convex hulls, and it took me a while to believe it was the same problem I'd learned in the standard way. The textbook version is an optimization: minimize the norm of the weight vector subject to every training point sitting on the correct side of a margin. It's fine. It's also weirdly hard to picture.

Here is the version that's easy to picture. Take your two classes of points and wrap each one in its convex hull, the tightest convex shape that contains it. Find the pair of points, one on each hull, that are closest to each other. Draw the perpendicular bisector of the segment between them. That line is the maximum-margin separator, and the margin is half the length of the segment. This is an exact equivalence, not an approximation (Bennett and Bredensteiner wrote it up properly in 2000).

![An SVM is a distance problem](figs/01_svm_is_distance.png)

Once you see it this way, a different kind of training algorithm suggests itself. You don't have to solve a quadratic program. You can just walk toward the closest pair.

This post is about one algorithm that does that, the Triangle Algorithm, and about a paper we've just finished on making it faster and putting a proper convergence proof under it. It's also about a benchmark run this week that went badly in an instructive way, and a second run that mapped out exactly where the badness starts. I'll try to keep the mathematics to what's needed for the pictures to make sense. Everything below is backed by code that's public, and I say something at the end about how the work was actually done, because that part is a little unusual.

---

## The Triangle Algorithm and its certificate

The algorithm keeps two points, p somewhere inside the first hull and q somewhere inside the second, and tries to pull them together. Lots of methods do something like this. What makes this one worth caring about is that every iterate comes with a certificate.

The upper bound part is trivial: p and q are on their hulls, so the true distance can't be larger than ‖p − q‖. The lower bound is the nice part. Take the direction from q to p and slide a line perpendicular to it until it just touches the first hull, then do the same on the other side for the second hull. The two lines are parallel, and the gap between them is a distance the hulls cannot possibly be closer than, because each hull lies entirely on its own side of its line.

![The certificate](figs/02_certificate.png)

So at every step you hold an interval that contains the answer. You stop when the interval is narrow enough, and at that point you know, not just hope, how close you are. LIBSVM and LIBLINEAR don't give you this; they stop when their internal progress measure gets small, which is usually fine and occasionally isn't. I didn't appreciate how much this certificate would matter until later, when it turned out to be the thing that made the screening result provable and, less comfortably, the thing that made the algorithm's bad days visible in the logs.

How does p actually move? Any point inside a hull is a weighted average of the hull's vertices, so moving p means changing weights. The classical move, from a 1974 paper by Mitchell, Demyanov and Malozemov, is to find the vertex that would help most if it carried more weight, find the vertex currently carrying weight that is hurting most, and shift some weight from the second to the first.

![The MDM weight transfer](figs/03_weight_transfer.png)

If you know the Frank–Wolfe literature you'll recognize this as the pairwise Frank–Wolfe step. I didn't, at first. Realizing it later was what let us borrow a convergence theory instead of building one.

---

## The zig-zag problem

There's a simpler move than the transfer, and it's what the original algorithm used: pick the most promising vertex and move p straight toward it, stopping at the point on that segment closest to q. It has a failure mode that everyone who has implemented one of these methods has met. When the true closest point is in the interior of an edge rather than at a vertex, the toward-step aims at one end of the edge, lands somewhere short of the target, then aims at the other end, and so on. It converges, but slower and slower.

![Zig-zag versus pairwise steps](figs/04_zigzag.png)

The figure is from the actual solver on a five-point toy problem. The toward-step version was still going after 399 steps; the transfer version needed six. In our earlier paper we had suggested a patch for this, jumping to the midpoint of the two vertices the algorithm was bouncing between. When I finally tested that patch properly, at a tight tolerance, it did almost nothing. On one instance plain toward-steps needed 159,000 iterations, the midpoint patch 149,000, and pairwise transfers 181. On a bigger instance the first two never finished within the 300,000-iteration budget and the transfers finished in under 4,000.

In retrospect the reason is simple. A toward-step can only add weight to a vertex. It has no way to take weight away from a vertex that's pulling in the wrong direction, except by diluting it, and dilution is slow. A transfer removes weight directly. It's a little embarrassing that the fix had been sitting in a 1974 paper the whole time, but that's usually how it goes.

---

## Block transfers

Every transfer needs a pass over all n points to find the best receiver and the worst donor. That's O(n) work for one move, and the question I kept coming back to was how much of that pass could be reused.

What we ended up with is a block transfer. Take the top k receivers and the worst k donors instead of one of each, pair them up best-with-worst, work out how far each pair would like to move, and then, rather than applying the k moves one after another, add the k directions together into one direction and do a single exact line search along it.

![Block transfer](figs/05_block.png)

The reason not to apply them sequentially is that you'd have to refresh all the scores between moves, and that refresh is the O(n) pass we were trying to avoid. The aggregate uses one snapshot of the scores, one line search, and one batched update of the caches for all k moves together.

You should be suspicious whenever someone claims to have combined k good moves into one, because moves can interfere; two directions that individually help can partly cancel. So we proved a bound: the block step makes at least 1/(2k) of the progress the single best transfer would have made, even when moves are clipped by how much weight a donor actually has. The proof is short, the triangle inequality and Cauchy–Schwarz and not much else.

A factor of 2k is not a great guarantee on its own. What makes the block step safe in practice is that the progress of the block and the progress of the single best transfer can both be computed in constant time from numbers the algorithm already has. So the algorithm computes both and takes whichever is larger. We call this the guard, and with it the block step provably never does worse than pairwise Frank–Wolfe at any iteration. In the situation that actually occurs in high dimensions, where the k directions are close to orthogonal, the block gets nearly the sum of all k gains. In the benchmarks that came out as somewhere between 4 and 30 times fewer iterations, with the bigger factors at higher dimension.

---

## The convergence theorem

Lacoste-Julien and Jaggi showed in 2015 that pairwise and away-step Frank–Wolfe converge linearly on strongly convex problems over polytopes, meaning the error shrinks by a fixed factor each iteration, at a rate set by a geometric quantity they called the pyramidal width. Loosely, it measures how fat the polytope is in every direction, relative to its diameter.

Two things had to happen for us to use that result. The first is that our objective needed to be strongly convex, and ½‖p − q‖² as a function of the pair (p, q) isn't; it doesn't care if you move p and q together. The fix is to work with the difference z = p − q, which lives in the Minkowski difference of the two hulls, the set of all differences of a point in one and a point in the other. That set is itself a polytope, the objective ½‖z‖² is as strongly convex as a function gets, and every move our algorithm makes on either hull is a legitimate move on this difference polytope. One pyramidal width then controls everything.

![Pyramidal width intuition](figs/07_width.png)

The second thing was getting around the worst part of pairwise Frank–Wolfe's theory, the so-called swap steps. These are iterations where a donor runs out of weight before the line search would like to stop, and the number of them is bounded in the original analysis by a constant involving a factorial. Our way around it is to never take a clipped transfer in that situation. Instead, take an away step on the donor: shrink its weight directly. An away step either makes guaranteed progress or drives the donor's weight to exactly zero, and the second case, a drop step, is easy to count, because you cannot drop more vertices than you have added. The factorial disappears.

What comes out is a linear-convergence theorem with every constant written down. The contraction per iteration is the square of (pyramidal width over diameter), divided by 16, and at least a 1/(k+1) fraction of the iterations contract. I should say that the first version of this theorem I wrote down had a constant with the wrong units, a length to the fourth power where there should have been a dimensionless ratio, and it survived for a couple of days until a re-derivation caught it. Every inequality in the final proof is also checked numerically in the released code, on thousands of random configurations including deliberately nasty clipped and boundary cases, and across more than five thousand instrumented steps of the real solver none of them increased the objective. I've come to think of those numerical checks as part of the proof rather than as decoration.

---

## Safe screening

In a problem with a sparse solution, most training points are irrelevant. They will never be support vectors and every pass over them is wasted. If you could identify them early you could stop scanning them. The obvious risk is throwing away something that mattered.

The certificate is what makes this safe. Given the current bounds, strong convexity says the true optimum sits within a radius r = √(UB² − LB²) of where you are now. A point with zero weight whose score margin, the amount by which it fails to be the extreme point in the current direction, exceeds r times its distance to that extreme point can never be a support vector for any optimum compatible with the certificate. So you delete it. The true support survives every deletion, the reduced problem has the same optimum, and every later certificate is still valid.

![Safe screening](figs/06_screening.png)

We also proved the rule actually fires, which isn't automatic for safe rules. There is an identity I found genuinely pleasing: the certified gap is exactly the Frank–Wolfe gap divided by the current distance. Chaining that with the linear rate gives an explicit iteration count after which every non-support point has been removed and each pass costs only the size of the support. On the test instances the surviving set was exactly the optimal support, and 10,000 points shrank to somewhere between 36 and about a thousand, for wall-clock gains of 1.4 to 2.4 times at tight tolerance.

At loose tolerance it does nothing, because the run finishes before the radius is small enough to exclude anyone. The theorem says that should happen and it does. I found that more reassuring than a rule that always helped would have been.

---

## Where it works and where it doesn't

On the problems the theory says should suit it, hard margins, linear and RBF, in high dimension, the optimized solver ties or slightly beats LIBSVM, within about 15 percent on all four classes we tried. On the three soft-margin classes, synthetic L2, MNIST odd-vs-even, and a ν-SVM, the standard solvers are 1.3 to 2.4 times faster. (An earlier run on a shared development container had shown us winning six of seven by a comfortable margin; the dedicated machine did not agree, and the dedicated numbers are the ones in the paper.) Where the new step rule wins outright is not speed but reach: at a tight tolerance the paper-style configuration doesn't converge at all and the new one takes half a second, and on MNIST single transfers never reach the tolerance while the block does.

![Consolidated benchmark](figs/08_regimes.png)

Then we rented a 32-core machine and ran the standard real-world benchmarks (a9a, w8a, the usual LIBSVM sets) with an L2 soft margin, and I watched the log for an hour with a sinking feeling. Every a9a cell was taking around fifty minutes. LIBLINEAR was taking twelve seconds. Same solution, same test accuracy to four digits, certified to the same tolerance, at about 250 times the cost.

![Real-data convergence traces](figs/09_realdata.png)

My first assumption was that it was stuck; the times were too uniform. It wasn't. The traces show it converging linearly, exactly as the theorem says, just with a terrible constant. The reason is in the support: on these datasets 60 to 65 percent of the points end up as support vectors with nearly equal weights. In the difference-polytope picture that means the optimum is in the middle of an almost regular simplex with thousands of vertices, and the pyramidal width of a simplex shrinks like 1/n. Each iteration costs O(n) and the number of iterations grows with n, so the whole thing is quadratic. Coordinate descent on the primal, which is what LIBLINEAR does, never looks at support density and simply doesn't care.

I tried the two obvious rescues. Starting from the centroid instead of a vertex made it slightly worse. Bigger blocks traded iterations for per-iteration cost and came out even. The theory was telling me the same thing both times: the geometry sets the rate, not the starting point.

So we gave every solver a ten-minute budget and ran the rest of the L2 battery. On the four overlapping datasets (a9a, w8a, ijcnn1, and covtype at a hundred thousand points), at three values of C, our solver hit the cap in every single cell and LIBLINEAR finished in between a third of a second and twenty-five seconds. One detail from those logs stuck with me. At C = 0.1 the iterate had LIBLINEAR's test accuracy, to three digits, long before the cap, while the certificate was still reporting a gap of 0.1 or worse. The point p is close to right; the proof that it's right is what takes an hour. That is what a small pyramidal width looks like from the inside.

The fifth dataset went the other way, and it's the cleanest illustration I have of the whole argument. Gisette has five thousand features and is linearly separable, so even its soft-margin solution has a sparse support, about a fifth of the points. Same solver, same code, same tolerance: 30 to 45 seconds to a certified solution at every value of C, against four to five minutes for our SMO and between eight minutes and an hour and a half for LIBLINEAR run to the same primal accuracy, all landing on the same answer and the same test accuracy. To be fair to LIBLINEAR, that tight tolerance is our choice, made so its answer would match ours; at its default tolerance it takes five seconds to a minute on gisette, about the same as we do. But at the default tolerance it stops 5 to 24 percent above the optimum, with lower test accuracy, and nothing in its output tells you so. At the largest C it spent its hour and a half even at the tight tolerance and still stopped five percent short of the optimum the certificate had pinned down; on separable data the primal objective goes flat at large C and a progress-based stopping rule fires too early, which is the kind of thing a certificate exists to catch. Nothing about the algorithm changed between covtype and gisette. The polytope did.

That run showed the boundary from the losing side. The theory also says where the winning side should be: sparse support, well-separated hulls, expensive inner products. So we ran a second battery on the same five datasets in the regime the algorithm is actually built for, hard margins and kernels, always against a baseline solving the same objective.

![Regime battery on real data](figs/10_regime_battery.png)

Some of it went the way the theory said. The first thing the algorithm does on each dataset is decide whether the two hulls intersect at all, which is a certificate that a linear hard margin does or doesn't exist; it settled that in under a minute per dataset, and none of the standard solvers can answer the question. Gisette, with five thousand features and a fifth of its points in the support, is the geometry the method likes: it tied LIBSVM on the linear hard margin, beat it by 20 percent on the RBF hard margin, and beat kernel SMO by a factor of three on the kernel soft margin, with identical solutions to five digits. Across all five datasets the kernel soft-margin cells needed four to six times fewer kernel-column evaluations than SMO; whether that turned into wall-clock depended on how expensive a column was, a win of three to four times on the wide datasets and a loss of about 1.5 times on the narrow ones, where our O(n) NumPy scan per iteration is the bottleneck and not the kernel.

Some of it didn't. On ijcnn1's RBF hard margin the two solvers returned the same answer to the last digit, 819 support vectors, identical accuracy, and ours took 420 seconds where LIBSVM took 1.3. The hulls in feature space are 0.004 apart in a space of diameter about 1.4. That ratio goes into the rate constant squared, and no step rule fixes it; LIBSVM's second-order working-set selection just handles that conditioning better. On a9a and covtype the feature-space hulls are within a ten-thousandth of touching and neither solver certified anything, ours timing out honestly and LIBSVM's large-C surrogate stopping at its tolerance a long way out. And on w8a the feature-space hulls intersect, because the dataset contains identical points with opposite labels, so no hard margin exists; the algorithm proved that and moved on.

So the picture that emerged is not "the method is slow on real data." It's that the method's speed is set by one geometric ratio, width over diameter, and the real datasets sample that ratio from both ends. The paper reports both. What I like about the certificate here is that the boundary is visible while the algorithm runs. Watch the support density and the gap and you know which side of it you're on.

I'd rather publish that than a paper that only showed the wins.

---

## What I'd want someone to take away

Geometry gives you certificates. Framing the SVM as a distance between hulls hands you upper and lower bounds at every iteration, and that turned out to be the foundation for both the screening result and the honest evaluation.

The step rule is not a detail. The gap between a toward-step and a weight transfer was the gap between 159,000 iterations and 181. The gap between an unguarded block and a guarded one is the gap between "usually faster" and "never slower."

And the pyramidal width tells you in advance whether a method of this kind will fly or crawl. Sparse support, fat polytope, fast. Dense support or nearly touching hulls, thin polytope, slow. If you have heavily overlapping data and want an L2 soft margin, use LIBLINEAR and don't look back. If you want a hard margin in high dimension, a kernel with expensive columns, a ν-SVM, or a certificate you can trust, this family is worth trying.

The solvers, the proofs, the numerical checks, and every experiment script including the ones that made the figures in this post are in the repository.

---