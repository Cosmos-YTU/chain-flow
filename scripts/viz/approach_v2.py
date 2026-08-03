"""Chained-Flow approach — detailed, concrete explainer.
Shows a real running example ("The capital of France is ...") with hidden-state
heatmaps, an animated flow trajectory, a Markov-bias bar chart, a growing draft
tree with real tokens, and a concrete verify/accept pass.
"""
import numpy as np
from manim import *

rng = np.random.default_rng(7)

C_BG    = "#0d1117"
C_BASE  = "#4f8bff"
C_DRAFT = "#ffd166"
C_FLOW  = "#2dd4bf"
C_VAE   = "#c792ea"
C_GOOD  = "#3ddc84"
C_BAD   = "#f4593b"
C_MUTE  = "#8b98a5"
C_HOT   = "#ff6b9d"

config.background_color = C_BG


# ---------- reusable visual primitives ----------
def heat(n_rows, seed=None, cell=0.16, base=C_FLOW, w=None):
    """A vertical hidden-state vector as a column of colored cells."""
    r = np.random.default_rng(seed) if seed is not None else rng
    vals = r.random(n_rows)
    cells = VGroup()
    for i, v in enumerate(vals):
        c = Square(side_length=cell, stroke_width=0.5, stroke_color="#00000055",
                   fill_color=base, fill_opacity=0.15 + 0.8 * v)
        cells.add(c)
    cells.arrange(DOWN, buff=0.02)
    return cells


def heat_grid(rows, cols, seed=0, cell=0.16, base=C_BASE):
    r = np.random.default_rng(seed)
    g = VGroup()
    for _ in range(cols):
        col = VGroup()
        for _ in range(rows):
            v = r.random()
            col.add(Square(side_length=cell, stroke_width=0.5, stroke_color="#00000055",
                           fill_color=base, fill_opacity=0.15 + 0.8 * v))
        col.arrange(DOWN, buff=0.02)
        g.add(col)
    g.arrange(RIGHT, buff=0.06)
    return g


def tok(label, color=C_DRAFT, w=1.05, h=0.62, fs=22, fill=0.16):
    b = RoundedRectangle(corner_radius=0.09, width=w, height=h, stroke_color=color,
                         stroke_width=2.5, fill_color=color, fill_opacity=fill)
    t = Text(label, font_size=fs, color=WHITE).move_to(b)
    return VGroup(b, t)


def bars(labels, values, color=C_DRAFT, bw=0.5, maxh=1.8, fs=18):
    """A simple vertical bar chart returning (group, {label:bar})."""
    g = VGroup()
    ref = {}
    for lab, v in zip(labels, values):
        bar = Rectangle(width=bw, height=max(0.03, v) * maxh, stroke_width=0,
                        fill_color=color, fill_opacity=0.9)
        lb = Text(lab, font_size=fs, color=C_MUTE)
        col = VGroup(bar, lb)
        bar.move_to(ORIGIN)
        lb.next_to(bar, DOWN, buff=0.12)
        # anchor bars to a common baseline
        col_anchor = VGroup(bar, lb)
        g.add(col_anchor)
        ref[lab] = bar
    g.arrange(RIGHT, buff=0.28, aligned_edge=DOWN)
    return g, ref


def header(n, txt, color):
    return Text(f"{n}  ·  {txt}", weight=BOLD, font_size=32, color=color).to_edge(UP, buff=0.45)


class ApproachV2(Scene):
    def construct(self):
        self.opening()
        self.s1_chain()
        self.s2_learnings()
        self.s3_markov()
        self.s4_flow()
        self.s5_vae()
        self.s6_verify()
        self.closing()

    # ---------------------------------------------------------------- #
    def opening(self):
        t = Text("Chained-Flow", weight=BOLD, font_size=48, color=C_FLOW)
        s = Text("drafting a whole tree of tokens in a single pass — then verifying it losslessly",
                 font_size=22, color=C_MUTE).next_to(t, DOWN, buff=0.3)
        self.play(Write(t))
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(1.0)
        # running example
        prompt = VGroup(*[tok(w, C_BASE, w=1.3) for w in ["The", "capital", "of", "France", "is"]])
        prompt.arrange(RIGHT, buff=0.14).shift(DOWN * 1.4)
        lab = Text("our running example", font_size=18, color=C_MUTE).next_to(prompt, UP, buff=0.3)
        self.play(FadeOut(VGroup(t, s), shift=UP * 0.3))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.1) for x in prompt], lag_ratio=0.2), FadeIn(lab))
        self.wait(0.8)
        q = tok("?", C_DRAFT, w=0.8).next_to(prompt, RIGHT, buff=0.2)
        self.play(FadeIn(q), Flash(q, color=C_DRAFT))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in [prompt, lab, q]])

    # ---------------------------------------------------------------- #
    def s1_chain(self):
        h = header("1", "The old way: a sequential chain", C_BASE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        clock = Text("passes: 0", font_size=22, color=C_MUTE).to_corner(UR, buff=0.6)
        self.play(FadeIn(clock))
        words = ["Paris", ".", "The", "city", "is", "home"]
        chain = VGroup().shift(LEFT * 5.2 + UP * 0.6)
        prev = None
        for i, w in enumerate(words):
            b = tok(w, C_DRAFT, w=1.15).next_to(prev, RIGHT, buff=0.85) if prev else tok(w, C_DRAFT, w=1.15).move_to(LEFT * 5.0 + UP * 0.6)
            newclock = Text(f"passes: {i+1}", font_size=22, color=C_MUTE).to_corner(UR, buff=0.6)
            anims = [GrowFromCenter(b), Transform(clock, newclock)]
            if prev:
                ar = Arrow(prev.get_right(), b.get_left(), buff=0.08, color=C_MUTE, stroke_width=3)
                anims.append(GrowArrow(ar)); chain.add(ar)
            self.play(*anims, run_time=0.55)
            chain.add(b); prev = b
        cap = Text("each token needs its own backbone pass — K passes, strictly one after another",
                   font_size=21, color=C_MUTE).to_edge(DOWN, buff=1.4)
        self.play(FadeIn(cap))
        self.wait(1.0)
        # fragility
        toks_only = [m for m in chain if isinstance(m, VGroup)]
        x = Cross(toks_only[2], color=C_BAD, stroke_width=6).scale(0.45)
        self.play(Create(x))
        self.play(*[toks_only[j].animate.set_opacity(0.2) for j in range(3, len(toks_only))])
        frag = Text("one wrong token → everything after it is thrown away", font_size=24, color=C_BAD).move_to(cap)
        self.play(FadeOut(cap), FadeIn(frag))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in [h, clock, chain, x, frag]])

    # ---------------------------------------------------------------- #
    def s2_learnings(self):
        h = header("2", "Two learnings: EAGLE & DSpark", C_DRAFT)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        lead = Text("what actually raises how many tokens get accepted?", font_size=23, color=C_MUTE).next_to(h, DOWN, buff=0.3)
        self.play(FadeIn(lead))

        # accept-length bar chart that grows as we add each lever
        chart, ref = bars(["base", "+DSpark", "+EAGLE"], [0.30, 0.30, 0.30], color=C_FLOW, maxh=2.3)
        chart.shift(DOWN * 0.4 + LEFT * 3.2)
        axis = Line(chart.get_corner(DL) + LEFT * 0.2, chart.get_corner(DR) + RIGHT * 0.2, color=C_MUTE)
        ylab = Text("accepted tokens", font_size=18, color=C_MUTE).rotate(PI / 2).next_to(chart, LEFT, buff=0.2)
        self.play(FadeIn(chart), Create(axis), FadeIn(ylab))

        # EAGLE explanation on the right
        e = VGroup(
            Text("EAGLE", weight=BOLD, font_size=24, color=C_BASE),
            Text("feed each chosen token back in,", font_size=20, color=C_MUTE),
            Text("re-run the backbone → +0.53", font_size=20, color=WHITE),
            Text("but: K sequential passes", font_size=20, color=C_BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).shift(RIGHT * 3.0 + UP * 0.6)
        self.play(FadeIn(e))
        self.play(ref["+EAGLE"].animate.stretch_to_fit_height(2.3 * 0.83).align_to(ref["base"], DOWN),
                  run_time=0.9)
        self.wait(1.2)

        d = VGroup(
            Text("DSpark", weight=BOLD, font_size=24, color=C_VAE),
            Text("a cheap low-rank token→bias", font_size=20, color=C_MUTE),
            Text("almost free → +0.25", font_size=20, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).shift(RIGHT * 3.0 + DOWN * 1.4)
        self.play(FadeIn(d))
        self.play(ref["+DSpark"].animate.stretch_to_fit_height(2.3 * 0.55).align_to(ref["base"], DOWN),
                  run_time=0.9)
        self.wait(1.0)
        take = Text("both are the SAME lever — token feedback. We want it without K sequential passes.",
                    font_size=22, color=C_GOOD).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(take, shift=UP * 0.2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, lead, chart, axis, ylab, e, d, take]])

    # ---------------------------------------------------------------- #
    def s3_markov(self):
        h = header("3", "Token feedback, made cheap: the Markov head", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # logits over a tiny vocab for our example, before feedback
        vocab = ["Paris", "France", "the", "a", "London", "home"]
        base_p = [0.34, 0.10, 0.16, 0.12, 0.05, 0.07]
        chart, ref = bars(vocab, base_p, color=C_DRAFT, maxh=2.2, fs=16)
        chart.shift(RIGHT * 2.6 + DOWN * 0.3)
        ttl = Text("next-token logits", font_size=18, color=C_MUTE).next_to(chart, UP, buff=0.2)
        self.play(FadeIn(chart), FadeIn(ttl))

        # the mechanism on the left, feeding 'France'
        prev = tok("France", C_BASE, w=1.5).shift(LEFT * 4.6 + UP * 1.4)
        w1 = Text("W1  (lookup)", font_size=18, color=C_VAE)
        rankv = heat(6, seed=3, base=C_VAE, cell=0.14).scale(1.0)
        w2 = Text("W2", font_size=18, color=C_VAE)
        step = VGroup(prev.copy()).shift(LEFT*4.6)
        rankv.next_to(prev, DOWN, buff=0.7)
        r_lab = Text("rank r ≪ vocab", font_size=16, color=C_MUTE).next_to(rankv, DOWN, buff=0.15)
        eq = Text("bias = W2( W1[ prev ] )", font_size=20, color=WHITE).next_to(prev, RIGHT, buff=0.4)
        self.play(FadeIn(prev), FadeIn(eq))
        a1 = Arrow(prev.get_bottom(), rankv.get_top(), buff=0.12, color=C_VAE, stroke_width=3)
        self.play(GrowArrow(a1), FadeIn(rankv), FadeIn(r_lab))
        self.wait(0.5)

        # the bias flows into the chart and reshapes it: Paris & France grow, others shrink
        bias_arrow = Arrow(rankv.get_right(), chart.get_left(), buff=0.2, color=C_VAE, stroke_width=3)
        blab = Text("+ bias", font_size=18, color=C_GOOD).next_to(bias_arrow, UP, buff=0.05)
        self.play(GrowArrow(bias_arrow), FadeIn(blab))
        new_p = [0.55, 0.16, 0.10, 0.07, 0.03, 0.04]
        self.play(*[ref[v].animate.stretch_to_fit_height(max(0.03, p) * 2.2).align_to(ref[vocab[0]], DOWN)
                    for v, p in zip(vocab, new_p)], run_time=1.2)
        self.play(ref["Paris"].animate.set_fill(C_GOOD), Flash(ref["Paris"], color=C_GOOD))
        note = Text("learns \"after France, boost Paris\" — starts at zero, so it only adds a correction",
                    font_size=20, color=C_MUTE).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, chart, ttl, prev, eq, rankv, r_lab, a1, bias_arrow, blab, note]])

    # ---------------------------------------------------------------- #
    def s4_flow(self):
        h = header("4", "The flow drafter: one pass → all K hiddens", C_FLOW)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # context hidden as a heatmap column
        ctxh = heat(14, seed=1, base=C_BASE, cell=0.15).shift(LEFT * 5.3 + UP * 0.4)
        ctxl = Text("context\nhidden", font_size=17, color=C_MUTE).next_to(ctxh, DOWN, buff=0.15)
        self.play(FadeIn(ctxh), FadeIn(ctxl))

        # flow trajectory: z0 morphs through 2 steps into target (a moving dot w/ traced path in a mini plane)
        plane = Rectangle(width=3.2, height=2.4, stroke_color=C_MUTE, stroke_width=1.2,
                          fill_opacity=0.04, fill_color=C_FLOW).shift(LEFT * 1.6 + UP * 0.5)
        pl_lab = Text("flow field (latent space)", font_size=16, color=C_MUTE).next_to(plane, UP, buff=0.12)
        z0 = Dot(plane.get_center() + LEFT * 1.0 + DOWN * 0.7, color=C_DRAFT, radius=0.08)
        z0l = Text("z0", font_size=16, color=C_DRAFT).next_to(z0, DOWN, buff=0.05)
        target = Dot(plane.get_center() + RIGHT * 1.1 + UP * 0.6, color=C_FLOW, radius=0.08)
        tl = Text("ĥ", font_size=18, color=C_FLOW).next_to(target, UP, buff=0.05)
        path = TracedPath(z0.get_center, stroke_color=C_FLOW, stroke_width=4)
        self.play(FadeIn(plane), FadeIn(pl_lab), FadeIn(z0), FadeIn(z0l), FadeIn(target), FadeIn(tl))
        self.add(path)
        mid = plane.get_center() + LEFT * 0.0 + DOWN * 0.05
        step_lab = Text("2 Euler steps", font_size=16, color=C_FLOW).next_to(plane, DOWN, buff=0.15)
        self.play(FadeIn(step_lab))
        self.play(z0.animate.move_to(mid), run_time=0.7)
        self.play(z0.animate.move_to(target.get_center()), run_time=0.7)
        self.wait(0.3)

        # output: all K hiddens at once as heatmap columns
        outs = VGroup(*[heat(14, seed=10 + i, base=C_FLOW, cell=0.13) for i in range(8)])
        outs.arrange(RIGHT, buff=0.12).shift(RIGHT * 3.4 + UP * 0.5)
        ol = Text("ĥ1 … ĥ8   (all at once)", font_size=17, color=C_FLOW).next_to(outs, DOWN, buff=0.15)
        big_ar = Arrow(plane.get_right(), outs.get_left(), buff=0.2, color=C_FLOW, stroke_width=4)
        self.play(GrowArrow(big_ar))
        self.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.1) for o in outs], lag_ratio=0.08), FadeIn(ol), run_time=1.4)
        punch = Text("one parallel pass predicts every future hidden — no sequential loop",
                     font_size=21, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in [h, ctxh, ctxl, plane, pl_lab, z0, z0l, target, tl, path,
                                         step_lab, outs, ol, big_ar, punch]])

        # each hidden -> top-b tokens -> a TREE with real words + probs
        h2 = header("4", "… then branch into a draft TREE", C_FLOW)
        self.play(FadeIn(h2))
        root = tok("is", C_BASE, w=0.9, h=0.55).shift(LEFT * 5.5)
        # depth 1
        p1 = tok("Paris", C_DRAFT, w=1.1, h=0.55).shift(LEFT * 3.3 + UP * 1.3)
        p2 = tok("the", C_DRAFT, w=1.0, h=0.55).shift(LEFT * 3.3 + DOWN * 1.3)
        # depth 2
        c1 = tok(".", C_DRAFT, w=0.7, h=0.5).shift(LEFT * 1.2 + UP * 1.9)
        c2 = tok(",", C_DRAFT, w=0.7, h=0.5).shift(LEFT * 1.2 + UP * 0.7)
        c3 = tok("capital", C_DRAFT, w=1.3, h=0.5).shift(LEFT * 1.0 + DOWN * 1.3)
        # depth 3
        d1 = tok("The", C_DRAFT, w=0.9, h=0.5).shift(RIGHT * 1.0 + UP * 1.9)
        d2 = tok("of", C_DRAFT, w=0.7, h=0.5).shift(RIGHT * 1.0 + DOWN * 1.3)

        def edge(a, b, p):
            l = Line(a.get_right(), b.get_left(), color=C_MUTE, stroke_width=2)
            t = Text(p, font_size=14, color=C_MUTE).move_to(l.get_center() + UP * 0.18)
            return VGroup(l, t)
        e1 = edge(root, p1, "0.55"); e2 = edge(root, p2, "0.16")
        e3 = edge(p1, c1, "0.6"); e4 = edge(p1, c2, "0.3"); e5 = edge(p2, c3, "0.7")
        e6 = edge(c1, d1, "0.8"); e7 = edge(c3, d2, "0.65")
        self.play(FadeIn(root))
        self.play(FadeIn(e1), FadeIn(e2), FadeIn(p1), FadeIn(p2))
        self.play(FadeIn(e3), FadeIn(e4), FadeIn(e5), FadeIn(c1), FadeIn(c2), FadeIn(c3))
        self.play(FadeIn(e6), FadeIn(e7), FadeIn(d1), FadeIn(d2))
        cap = Text("keep the top-b tokens at every step · edges = probabilities · path feedback added cheaply",
                   font_size=20, color=C_MUTE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cap))
        self.wait(1.8)
        self.tree = VGroup(root, p1, p2, c1, c2, c3, d1, d2, e1, e2, e3, e4, e5, e6, e7)
        self.play(*[FadeOut(m) for m in [h2, self.tree, cap]])

    # ---------------------------------------------------------------- #
    def s5_vae(self):
        h = header("5", "The joint VAE: flow in a tiny latent", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        big = heat(22, seed=2, base=C_BASE, cell=0.11).shift(LEFT * 5.4 + UP * 0.2)
        bl = Text("hidden 4096", font_size=16, color=C_MUTE).next_to(big, DOWN, buff=0.12)
        small = heat(8, seed=5, base=C_VAE, cell=0.14).shift(LEFT * 2.4 + UP * 0.2)
        sl = Text("latent 1024", font_size=16, color=C_MUTE).next_to(small, DOWN, buff=0.12)
        small2 = heat(8, seed=6, base=C_VAE, cell=0.14).shift(RIGHT * 0.6 + UP * 0.2)
        sl2 = Text("latent (flowed)", font_size=16, color=C_MUTE).next_to(small2, DOWN, buff=0.12)
        outk = VGroup(*[heat(22, seed=30 + i, base=C_FLOW, cell=0.09) for i in range(4)]).arrange(RIGHT, buff=0.1).shift(RIGHT * 3.6 + UP * 0.2)
        okl = Text("K hiddens", font_size=16, color=C_MUTE).next_to(outk, DOWN, buff=0.12)
        a1 = Arrow(big.get_right(), small.get_left(), buff=0.15, color=C_VAE, stroke_width=3)
        a1l = Text("encode", font_size=15, color=C_VAE).next_to(a1, UP, buff=0.05)
        a2 = Arrow(small.get_right(), small2.get_left(), buff=0.15, color=C_FLOW, stroke_width=3)
        a2l = Text("flow", font_size=15, color=C_FLOW).next_to(a2, UP, buff=0.05)
        a3 = Arrow(small2.get_right(), outk.get_left(), buff=0.15, color=C_VAE, stroke_width=3)
        a3l = Text("decode", font_size=15, color=C_VAE).next_to(a3, UP, buff=0.05)
        self.play(FadeIn(big), FadeIn(bl))
        self.play(GrowArrow(a1), FadeIn(a1l), FadeIn(small), FadeIn(sl))
        self.play(GrowArrow(a2), FadeIn(a2l), FadeIn(small2), FadeIn(sl2))
        self.play(GrowArrow(a3), FadeIn(a3l), FadeIn(outk), FadeIn(okl))
        cheap = Text("the expensive flow runs in the small latent → cheap, and portable across 4B / 9B / 27B",
                     font_size=20, color=C_MUTE).next_to(big, UP, buff=0.4).shift(RIGHT * 2)
        self.play(FadeIn(cheap))
        self.wait(1.2)

        pipeline = VGroup(big, bl, small, sl, small2, sl2, outk, okl, a1, a1l, a2, a2l, a3, a3l, cheap)
        self.play(pipeline.animate.scale(0.8).to_edge(UP, buff=1.1))

        # frozen vs joint — concrete outcome
        frozen = VGroup(
            Text("Frozen VAE — trained to RECONSTRUCT", font_size=21, color=C_BAD),
            Text("9B: reconstruction 0.53  →  accept 0   ✗", font_size=21, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).shift(DOWN * 1.4 + LEFT * 3.0)
        joint = VGroup(
            Text("Joint VAE — UNFROZEN, trained for ACCEPT", font_size=21, color=C_GOOD),
            Text("(+ a reconstruction anchor to stay decodable)", font_size=18, color=C_MUTE),
            Text("9B: accept 4.9  →  1.27× lossless   ✓", font_size=21, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).shift(DOWN * 1.7 + RIGHT * 2.6)
        self.play(FadeIn(frozen))
        self.wait(0.8)
        self.play(FadeIn(joint))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in [h, pipeline, frozen, joint]])

    # ---------------------------------------------------------------- #
    def s6_verify(self):
        h = header("6", "Verify the whole tree in ONE pass", C_GOOD)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # rebuild a compact tree (nodes) with indices
        root = tok("is", C_BASE, w=0.8, h=0.5).shift(LEFT * 5.4 + UP * 0.3)
        p1 = tok("Paris", C_DRAFT, w=1.0, h=0.5).shift(LEFT * 3.5 + UP * 1.4)
        p2 = tok("the", C_DRAFT, w=0.9, h=0.5).shift(LEFT * 3.5 + DOWN * 0.9)
        c1 = tok(".", C_DRAFT, w=0.6, h=0.45).shift(LEFT * 1.6 + UP * 2.0)
        c2 = tok(",", C_DRAFT, w=0.6, h=0.45).shift(LEFT * 1.6 + UP * 0.8)
        c3 = tok("capital", C_DRAFT, w=1.2, h=0.45).shift(LEFT * 1.6 + DOWN * 0.9)
        e1 = Line(root.get_right(), p1.get_left(), color=C_MUTE, stroke_width=2)
        e2 = Line(root.get_right(), p2.get_left(), color=C_MUTE, stroke_width=2)
        e3 = Line(p1.get_right(), c1.get_left(), color=C_MUTE, stroke_width=2)
        e4 = Line(p1.get_right(), c2.get_left(), color=C_MUTE, stroke_width=2)
        e5 = Line(p2.get_right(), c3.get_left(), color=C_MUTE, stroke_width=2)
        tree = VGroup(e1, e2, e3, e4, e5, root, p1, p2, c1, c2, c3)
        self.play(FadeIn(tree))

        # tree-attention mask: a grid showing each node attends only to its ancestors
        grid_lab = Text("tree attention: each node sees only its own ancestors", font_size=19, color=C_MUTE)
        grid_lab.shift(RIGHT * 3.2 + UP * 2.1)
        names = ["is", "Paris", "the", ".", ",", "cap"]
        anc = [[1,0,0,0,0,0],[1,1,0,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[1,1,0,0,1,0],[1,0,1,0,0,1]]
        cells = VGroup()
        cs = 0.34
        for i in range(6):
            for j in range(6):
                on = anc[i][j]
                sq = Square(side_length=cs, stroke_width=1, stroke_color="#33415577",
                            fill_color=C_GOOD if on else C_BG, fill_opacity=0.75 if on else 0.15)
                sq.move_to(RIGHT * (2.2 + j * cs) + UP * (1.4 - i * cs))
                cells.add(sq)
        self.play(FadeIn(grid_lab), FadeIn(cells, lag_ratio=0.01, run_time=1.2))
        one = Text("→ the backbone scores all 6 nodes in a single forward pass",
                   font_size=19, color=WHITE).next_to(cells, DOWN, buff=0.4)
        self.play(FadeIn(one))
        self.wait(1.4)

        # accept the longest correct path: is -> Paris -> .
        acc_cap = Text("accept the longest path the backbone agrees with:", font_size=20, color=C_GOOD).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(acc_cap))
        self.play(root[0].animate.set_stroke(C_GOOD, 5),
                  e1.animate.set_color(C_GOOD).set_stroke(width=4), p1[0].animate.set_stroke(C_GOOD, 5))
        self.play(e3.animate.set_color(C_GOOD).set_stroke(width=4), c1[0].animate.set_stroke(C_GOOD, 5))
        # reject the rest
        self.play(*[m.animate.set_opacity(0.2) for m in [p2, c2, c3, e2, e4, e5]])
        res = Text("3 tokens accepted from 1 verify pass — guaranteed identical to normal decoding",
                   font_size=21, color=C_GOOD).move_to(acc_cap)
        self.play(FadeOut(acc_cap), FadeIn(res))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in [h, tree, grid_lab, cells, one, res]])

    # ---------------------------------------------------------------- #
    def closing(self):
        title = Text("Chained-Flow", weight=BOLD, font_size=40, color=C_FLOW).to_edge(UP, buff=1.0)
        rows = VGroup(
            Text("one flow pass  →  a whole draft tree", font_size=26, color=C_FLOW),
            Text("token feedback (Markov + path)  →  at zero sequential cost", font_size=24, color=C_VAE),
            Text("verified in one backbone pass  →  lossless", font_size=24, color=C_GOOD),
        ).arrange(DOWN, buff=0.35).shift(UP * 0.2)
        results = VGroup(
            Text("measured — 4B: accept 6.7   ·   9B: accept 4.9 → 1.27× lossless",
                 font_size=22, color=WHITE),
        ).next_to(rows, DOWN, buff=0.7)
        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in rows], lag_ratio=0.4), run_time=2.0)
        self.play(FadeIn(results, shift=UP * 0.15))
        self.wait(2.2)
        self.play(FadeOut(VGroup(title, rows, results)))
