"""Chained-Flow approach v3 — detailed, concrete, corrected.
Fixes: chain example shows a real wrong (swapped) token cascading; learnings bar
chart grows left-to-right, baseline-aligned; frozen-vs-joint VAE spelled out by
what each optimizes; tree attention explained in depth incl. one-pass /
one-greedy-path acceptance (no per-candidate re-runs); no overlapping layout.
"""
import numpy as np
from manim import *

C_BG    = "#0d1117"
C_BASE  = "#4f8bff"
C_DRAFT = "#ffd166"
C_FLOW  = "#2dd4bf"
C_VAE   = "#c792ea"
C_GOOD  = "#3ddc84"
C_BAD   = "#f4593b"
C_MUTE  = "#8b98a5"

config.background_color = C_BG


def heat(n, seed, base=C_FLOW, cell=0.15):
    r = np.random.default_rng(seed)
    g = VGroup()
    for _ in range(n):
        v = r.random()
        g.add(Square(side_length=cell, stroke_width=0.5, stroke_color="#00000055",
                     fill_color=base, fill_opacity=0.2 + 0.75 * v))
    g.arrange(DOWN, buff=0.02)
    return g


def tok(label, color=C_DRAFT, w=1.05, h=0.62, fs=22, fill=0.16):
    b = RoundedRectangle(corner_radius=0.09, width=w, height=h, stroke_color=color,
                         stroke_width=2.5, fill_color=color, fill_opacity=fill)
    return VGroup(b, Text(label, font_size=fs, color=WHITE).move_to(b))


def header(n, txt, color):
    return Text(f"{n}  ·  {txt}", weight=BOLD, font_size=31, color=color).to_edge(UP, buff=0.4)


class BarChart2:
    """Bottom-anchored bars that grow in place, aligned to a common baseline."""
    def __init__(self, labels, baseline_y, x0, dx, bw=0.55, unit=2.2, color=C_FLOW):
        self.bars, self.labs, self.unit, self.baseline_y = {}, {}, unit, baseline_y
        self.group = VGroup()
        for i, lab in enumerate(labels):
            x = x0 + i * dx
            bar = Rectangle(width=bw, height=0.02, stroke_width=0, fill_color=color, fill_opacity=0.9)
            bar.move_to([x, baseline_y + 0.01, 0])
            t = Text(lab, font_size=17, color=C_MUTE).move_to([x, baseline_y - 0.28, 0])
            self.bars[lab] = (bar, x)
            self.labs[lab] = t
            self.group.add(bar, t)

    def set(self, lab, val):
        bar, x = self.bars[lab]
        h = max(0.02, val * self.unit)
        newbar = Rectangle(width=bar.width, height=h, stroke_width=0,
                           fill_color=bar.get_fill_color(), fill_opacity=0.9)
        newbar.move_to([x, self.baseline_y + h / 2, 0])
        return bar.animate.become(newbar)


class ApproachV3(Scene):
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
        s = Text("draft a whole tree of tokens in one pass — then verify it losslessly",
                 font_size=22, color=C_MUTE).next_to(t, DOWN, buff=0.3)
        self.play(Write(t)); self.play(FadeIn(s, shift=UP * 0.2)); self.wait(0.9)
        prompt = VGroup(*[tok(w, C_BASE, w=1.35) for w in ["The", "capital", "of", "France", "is"]])
        prompt.arrange(RIGHT, buff=0.14).shift(DOWN * 1.4)
        lab = Text("running example — the model should continue with \"Paris\"",
                   font_size=19, color=C_MUTE).next_to(prompt, UP, buff=0.35)
        self.play(FadeOut(VGroup(t, s), shift=UP * 0.3))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.1) for x in prompt], lag_ratio=0.18), FadeIn(lab))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [prompt, lab]])

    # ---------------------------------------------------------------- #
    def s1_chain(self):
        h = header("1", "The old way: a sequential chain", C_BASE)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        sub = Text("the drafter emits one token, feeds it back, emits the next …",
                   font_size=21, color=C_MUTE).next_to(h, DOWN, buff=0.25)
        self.play(FadeIn(sub))

        counter = Text("backbone passes: 0", font_size=20, color=C_MUTE).move_to([4.3, 1.9, 0])
        self.play(FadeIn(counter))
        # draft: correct would be "Paris . The city ..." but the chain guesses "France" wrongly at step 1
        words = ["France", "is", "the", "capital", "city"]
        correct = [False, True, True, True, True]  # step-1 "France" is the wrong token (should be "Paris")
        chain = VGroup()
        prev = None
        boxes = []
        for i, w in enumerate(words):
            b = tok(w, C_DRAFT, w=1.25)
            if prev is None:
                b.move_to(LEFT * 5.1 + UP * 0.5)
            else:
                b.next_to(prev, RIGHT, buff=0.8)
            cnt = Text(f"backbone passes: {i+1}", font_size=20, color=C_MUTE).move_to([4.3, 1.9, 0])
            anims = [GrowFromCenter(b), Transform(counter, cnt)]
            if prev is not None:
                ar = Arrow(prev.get_right(), b.get_left(), buff=0.06, color=C_MUTE, stroke_width=3)
                anims.append(GrowArrow(ar)); chain.add(ar)
            self.play(*anims, run_time=0.5)
            chain.add(b); boxes.append(b); prev = b

        self.wait(0.5)
        # the model's TRUE next token after "is" was "Paris" — the chain guessed "France"
        truth = tok("Paris", C_GOOD, w=1.25).move_to(boxes[0]).shift(UP * 1.5)
        tlab = Text("backbone's true next token", font_size=16, color=C_GOOD).next_to(truth, UP, buff=0.12)
        self.play(FadeIn(truth, shift=DOWN * 0.2), FadeIn(tlab))
        x = Cross(boxes[0], color=C_BAD, stroke_width=6).scale(0.5)
        self.play(Create(x))
        self.wait(0.4)
        # cascade: everything after the wrong token is built on it → wasted
        self.play(*[boxes[j].animate.set_opacity(0.18) for j in range(1, len(boxes))])
        frag = Text("one wrong token → every later token was built on it → all wasted",
                    font_size=23, color=C_BAD).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(frag))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, sub, counter, chain, truth, tlab, x, frag]])

    # ---------------------------------------------------------------- #
    def s2_learnings(self):
        h = header("2", "What raises acceptance: EAGLE & DSpark", C_DRAFT)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # chart: base -> +DSpark -> +EAGLE, left to right, baseline aligned
        chart = BarChart2(["base", "+DSpark", "+EAGLE"], baseline_y=-2.0, x0=-4.6, dx=1.15, unit=1.9, color=C_FLOW)
        axis = Line([-5.4, -2.0, 0], [-1.7, -2.0, 0], color=C_MUTE)
        ylab = Text("accepted tokens", font_size=17, color=C_MUTE).rotate(PI / 2).move_to([-5.7, -0.9, 0])
        self.play(FadeIn(chart.group), Create(axis), FadeIn(ylab))
        self.play(chart.set("base", 1.0), run_time=0.7)

        # DSpark first (matches left-to-right order)
        d = VGroup(
            Text("DSpark", weight=BOLD, font_size=25, color=C_VAE),
            Text("a cheap low-rank token → bias", font_size=20, color=C_MUTE),
            Text("nearly free  →  +0.25", font_size=20, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).shift(RIGHT * 2.7 + UP * 1.5)
        self.play(FadeIn(d))
        self.play(chart.set("+DSpark", 1.25), run_time=0.8)
        self.wait(0.9)

        e = VGroup(
            Text("EAGLE", weight=BOLD, font_size=25, color=C_BASE),
            Text("feed each token back, re-run backbone", font_size=20, color=C_MUTE),
            Text("stronger  →  +0.53", font_size=20, color=WHITE),
            Text("but pays K sequential passes", font_size=20, color=C_BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).shift(RIGHT * 2.7 + DOWN * 1.3)
        self.play(FadeIn(e))
        self.play(chart.set("+EAGLE", 1.53), run_time=0.8)
        self.wait(1.0)
        take = Text("both are the same lever — token feedback. Our goal: keep it, drop the K passes.",
                    font_size=21, color=C_GOOD).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(take, shift=UP * 0.2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, chart.group, axis, ylab, d, e, take]])

    # ---------------------------------------------------------------- #
    def s3_markov(self):
        h = header("3", "Token feedback, made cheap: the Markov head", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        vocab = ["Paris", "France", "the", "a", "London", "home"]
        base_p = [0.34, 0.10, 0.16, 0.12, 0.05, 0.07]
        chart = BarChart2(vocab, baseline_y=-1.6, x0=0.7, dx=0.85, bw=0.5, unit=2.0, color=C_DRAFT)
        # rename labels smaller
        for lab in vocab:
            chart.labs[lab].set(font_size=15)
        ttl = Text("next-token logits", font_size=18, color=C_MUTE).move_to([2.8, 0.9, 0])
        self.play(FadeIn(chart.group), FadeIn(ttl))
        self.play(*[chart.set(v, p) for v, p in zip(vocab, base_p)], run_time=0.8)

        prev = tok("France", C_BASE, w=1.5).shift(LEFT * 4.6 + UP * 1.2)
        eq = Text("bias = W2( W1[ prev ] )", font_size=20, color=WHITE).next_to(prev, DOWN, buff=0.35)
        rankv = heat(6, seed=3, base=C_VAE, cell=0.16).next_to(eq, DOWN, buff=0.35)
        r_lab = Text("rank r ≪ vocab · starts at zero", font_size=15, color=C_MUTE).next_to(rankv, DOWN, buff=0.12)
        self.play(FadeIn(prev), FadeIn(eq))
        self.play(FadeIn(rankv), FadeIn(r_lab))
        bias_arrow = Arrow(rankv.get_right(), [0.2, -0.6, 0], buff=0.2, color=C_VAE, stroke_width=3)
        blab = Text("+ bias", font_size=18, color=C_GOOD).next_to(bias_arrow, UP, buff=0.05)
        self.play(GrowArrow(bias_arrow), FadeIn(blab))
        new_p = [0.55, 0.16, 0.10, 0.07, 0.03, 0.04]
        self.play(*[chart.set(v, p) for v, p in zip(vocab, new_p)], run_time=1.1)
        self.play(chart.bars["Paris"][0].animate.set_fill(C_GOOD), Flash(chart.bars["Paris"][0], color=C_GOOD))
        note = Text("learns \"after France, boost Paris\" — the same lever, at zero sequential cost",
                    font_size=20, color=C_MUTE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, chart.group, ttl, prev, eq, rankv, r_lab, bias_arrow, blab, note]])

    # ---------------------------------------------------------------- #
    def s4_flow(self):
        h = header("4", "The flow drafter: one pass → all K hiddens", C_FLOW)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        ctxh = heat(14, seed=1, base=C_BASE, cell=0.15).shift(LEFT * 5.3 + UP * 0.3)
        ctxl = Text("context\nhidden", font_size=16, color=C_MUTE).next_to(ctxh, DOWN, buff=0.15)
        self.play(FadeIn(ctxh), FadeIn(ctxl))
        plane = Rectangle(width=3.0, height=2.2, stroke_color=C_MUTE, stroke_width=1.2,
                          fill_opacity=0.04, fill_color=C_FLOW).shift(LEFT * 1.7 + UP * 0.4)
        pl_lab = Text("flow in latent space (2 Euler steps)", font_size=16, color=C_MUTE).next_to(plane, UP, buff=0.12)
        z0 = Dot(plane.get_center() + LEFT * 1.0 + DOWN * 0.6, color=C_DRAFT, radius=0.08)
        z0l = Text("z0", font_size=15, color=C_DRAFT).next_to(z0, LEFT, buff=0.08)
        target = Dot(plane.get_center() + RIGHT * 1.0 + UP * 0.55, color=C_FLOW, radius=0.08)
        path = TracedPath(z0.get_center, stroke_color=C_FLOW, stroke_width=4)
        self.play(FadeIn(plane), FadeIn(pl_lab), FadeIn(z0), FadeIn(z0l), FadeIn(target))
        self.add(path)
        self.play(z0.animate.move_to(plane.get_center() + DOWN * 0.05), run_time=0.6)
        self.play(z0.animate.move_to(target.get_center()), run_time=0.6)
        outs = VGroup(*[heat(14, seed=10 + i, base=C_FLOW, cell=0.12) for i in range(8)])
        outs.arrange(RIGHT, buff=0.12).shift(RIGHT * 3.5 + UP * 0.4)
        ol = Text("ĥ1 … ĥ8  — all at once", font_size=16, color=C_FLOW).next_to(outs, DOWN, buff=0.15)
        big_ar = Arrow(plane.get_right(), outs.get_left(), buff=0.2, color=C_FLOW, stroke_width=4)
        self.play(GrowArrow(big_ar))
        self.play(LaggedStart(*[FadeIn(o) for o in outs], lag_ratio=0.08), FadeIn(ol), run_time=1.2)
        punch = Text("one parallel pass predicts every future hidden — no sequential loop",
                     font_size=21, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch)); self.wait(1.3)
        self.play(*[FadeOut(m) for m in [h, ctxh, ctxl, plane, pl_lab, z0, z0l, target, path, outs, ol, big_ar, punch]])

        # tree
        h2 = header("4", "… keep top-b at each step → a draft TREE", C_FLOW)
        self.play(FadeIn(h2))
        root = tok("is", C_BASE, w=0.85, h=0.55).shift(LEFT * 5.4 + UP * 0.2)
        p1 = tok("Paris", C_DRAFT, w=1.05, h=0.55).shift(LEFT * 3.2 + UP * 1.5)
        p2 = tok("the", C_DRAFT, w=0.95, h=0.55).shift(LEFT * 3.2 + DOWN * 1.4)
        c1 = tok(".", C_DRAFT, w=0.6, h=0.5).shift(LEFT * 0.9 + UP * 2.1)
        c2 = tok(",", C_DRAFT, w=0.6, h=0.5).shift(LEFT * 0.9 + UP * 0.9)
        c3 = tok("capital", C_DRAFT, w=1.25, h=0.5).shift(LEFT * 0.7 + DOWN * 1.4)
        d1 = tok("The", C_DRAFT, w=0.85, h=0.5).shift(RIGHT * 1.4 + UP * 2.1)
        d2 = tok("of", C_DRAFT, w=0.6, h=0.5).shift(RIGHT * 1.4 + DOWN * 1.4)

        def edge(a, b, p, up=True):
            l = Line(a.get_right(), b.get_left(), color=C_MUTE, stroke_width=2)
            off = UP * 0.2 if up else DOWN * 0.2
            t = Text(p, font_size=13, color=C_MUTE).move_to(l.point_from_proportion(0.5) + off)
            return VGroup(l, t)
        edges = VGroup(edge(root, p1, "0.55"), edge(root, p2, "0.16", up=False),
                       edge(p1, c1, "0.6"), edge(p1, c2, "0.3", up=False), edge(p2, c3, "0.7"),
                       edge(c1, d1, "0.8"), edge(c3, d2, "0.65"))
        nodes = VGroup(root, p1, p2, c1, c2, c3, d1, d2)
        self.play(FadeIn(root))
        self.play(FadeIn(edges[0]), FadeIn(edges[1]), FadeIn(p1), FadeIn(p2))
        self.play(FadeIn(edges[2]), FadeIn(edges[3]), FadeIn(edges[4]), FadeIn(c1), FadeIn(c2), FadeIn(c3))
        self.play(FadeIn(edges[5]), FadeIn(edges[6]), FadeIn(d1), FadeIn(d2))
        cap = Text("one flow pass → many candidate continuations at once · edges = probabilities",
                   font_size=20, color=C_MUTE).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(cap)); self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h2, edges, nodes, cap]])

    # ---------------------------------------------------------------- #
    def s5_vae(self):
        h = header("5", "The joint VAE — and why frozen fails", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # compact pipeline at top
        big = heat(18, seed=2, base=C_BASE, cell=0.10).shift(LEFT * 5.3 + UP * 1.6)
        small = heat(7, seed=5, base=C_VAE, cell=0.13).shift(LEFT * 3.0 + UP * 1.6)
        outk = VGroup(*[heat(18, seed=30 + i, base=C_FLOW, cell=0.08) for i in range(3)]).arrange(RIGHT, buff=0.1).shift(LEFT * 0.7 + UP * 1.6)
        a1 = Arrow(big.get_right(), small.get_left(), buff=0.12, color=C_VAE, stroke_width=3)
        a2 = Arrow(small.get_right(), outk.get_left(), buff=0.12, color=C_VAE, stroke_width=3)
        labs = VGroup(Text("hidden", font_size=13, color=C_MUTE).next_to(big, DOWN, buff=0.08),
                      Text("latent", font_size=13, color=C_MUTE).next_to(small, DOWN, buff=0.08),
                      Text("encode", font_size=12, color=C_VAE).next_to(a1, UP, buff=0.03),
                      Text("flow+decode", font_size=12, color=C_FLOW).next_to(a2, UP, buff=0.03),
                      Text("K hiddens", font_size=13, color=C_MUTE).next_to(outk, DOWN, buff=0.08))
        pipe = VGroup(big, small, outk, a1, a2, labs)
        self.play(FadeIn(pipe))
        self.wait(0.4)

        # two rows: FROZEN vs JOINT, each showing what it's optimized for + the token outcome
        # frozen row
        fr_t = Text("FROZEN VAE", weight=BOLD, font_size=22, color=C_BAD).shift(LEFT * 5.0 + UP * 0.1)
        fr1 = Text("trained ONLY to reconstruct (h → z → h), then frozen", font_size=19, color=C_MUTE)
        fr2 = Text("reconstruction-perfect ≠ token-perfect: at 9B the decoded hidden", font_size=19, color=C_MUTE)
        fr3 = Text("keeps the right top-token only 53% of the time", font_size=19, color=WHITE)
        fr_txt = VGroup(fr1, fr2, fr3).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(fr_t, DOWN, aligned_edge=LEFT, buff=0.15)
        fr_tokbad = tok("London", C_BAD, w=1.3, h=0.5).shift(RIGHT * 4.8 + UP * 0.0)
        fr_x = Text("wrong token → accept 0", font_size=17, color=C_BAD).next_to(fr_tokbad, DOWN, buff=0.1)
        self.play(FadeIn(fr_t), FadeIn(fr_txt))
        self.play(FadeIn(fr_tokbad), FadeIn(fr_x))
        self.wait(1.4)

        # joint row
        jt_t = Text("JOINT VAE", weight=BOLD, font_size=22, color=C_GOOD).shift(LEFT * 5.0 + DOWN * 1.8)
        jt1 = Text("UNFROZEN · trained jointly with the flow, for ACCEPT", font_size=19, color=C_MUTE)
        jt2 = Text("objective = does the decoded hidden give the RIGHT token?", font_size=19, color=C_MUTE)
        jt3 = Text("(+ a small reconstruction anchor to stay decodable)", font_size=18, color=C_MUTE)
        jt_txt = VGroup(jt1, jt2, jt3).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(jt_t, DOWN, aligned_edge=LEFT, buff=0.15)
        jt_tokgood = tok("Paris", C_GOOD, w=1.3, h=0.5).shift(RIGHT * 4.8 + DOWN * 1.9)
        jt_ok = Text("right token → accept 4.9 · 1.27×", font_size=17, color=C_GOOD).next_to(jt_tokgood, DOWN, buff=0.1)
        self.play(FadeIn(jt_t), FadeIn(jt_txt))
        self.play(FadeIn(jt_tokgood), FadeIn(jt_ok))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in [h, pipe, fr_t, fr_txt, fr_tokbad, fr_x, jt_t, jt_txt, jt_tokgood, jt_ok]])

    # ---------------------------------------------------------------- #
    def s6_verify(self):
        h = header("6", "Verify: one pass, one greedy path", C_GOOD)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # small tree left
        root = tok("is", C_BASE, w=0.8, h=0.5).shift(LEFT * 5.5 + UP * 0.2)
        p1 = tok("Paris", C_DRAFT, w=1.0, h=0.5).shift(LEFT * 3.7 + UP * 1.4)
        p2 = tok("the", C_DRAFT, w=0.9, h=0.5).shift(LEFT * 3.7 + DOWN * 1.0)
        c1 = tok(".", C_DRAFT, w=0.55, h=0.45).shift(LEFT * 1.9 + UP * 2.0)
        c2 = tok(",", C_DRAFT, w=0.55, h=0.45).shift(LEFT * 1.9 + UP * 0.7)
        c3 = tok("big", C_DRAFT, w=0.8, h=0.45).shift(LEFT * 1.9 + DOWN * 1.0)
        e = [Line(root.get_right(), p1.get_left(), color=C_MUTE, stroke_width=2),
             Line(root.get_right(), p2.get_left(), color=C_MUTE, stroke_width=2),
             Line(p1.get_right(), c1.get_left(), color=C_MUTE, stroke_width=2),
             Line(p1.get_right(), c2.get_left(), color=C_MUTE, stroke_width=2),
             Line(p2.get_right(), c3.get_left(), color=C_MUTE, stroke_width=2)]
        tree = VGroup(*e, root, p1, p2, c1, c2, c3)
        self.play(FadeIn(tree))

        # tree-attention mask depth (compact, upper-right)
        gl = Text("tree attention", font_size=18, color=C_MUTE).move_to([2.9, 2.3, 0])
        gl2 = Text("each node sees only its\nroot→node path", font_size=15, color=C_MUTE, line_spacing=0.8).move_to([4.9, 1.9, 0])
        names = ["is", "Paris", "the", ".", ",", "big"]
        anc = [[1,0,0,0,0,0],[1,1,0,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[1,1,0,0,1,0],[1,0,1,0,0,1]]
        cells = VGroup(); cs = 0.30; ox, oy = 2.5, 1.5
        for i in range(6):
            for j in range(6):
                on = anc[i][j]
                sq = Square(side_length=cs, stroke_width=1, stroke_color="#33415577",
                            fill_color=C_GOOD if on else C_BG, fill_opacity=0.7 if on else 0.12)
                sq.move_to([ox + j * cs, oy - i * cs, 0]); cells.add(sq)
        rlabs = VGroup(*[Text(n, font_size=12, color=C_MUTE).move_to([ox - 0.42, oy - i * cs, 0]) for i, n in enumerate(names)])
        self.play(FadeIn(gl), FadeIn(gl2), FadeIn(cells, lag_ratio=0.01), FadeIn(rlabs))
        one = Text("the backbone scores every node in ONE forward pass",
                   font_size=20, color=WHITE).move_to([0, -1.5, 0])
        self.play(FadeIn(one)); self.wait(1.2)

        # acceptance: greedy walk following backbone argmax — ONE path, no re-runs
        acc = Text("accept = walk from the root, taking the child that matches the backbone's argmax",
                   font_size=20, color=C_GOOD).move_to([0, -2.2, 0])
        self.play(FadeIn(acc))
        self.play(root[0].animate.set_stroke(C_GOOD, 5), e[0].animate.set_color(C_GOOD).set_stroke(width=4),
                  p1[0].animate.set_stroke(C_GOOD, 5))
        self.play(e[2].animate.set_color(C_GOOD).set_stroke(width=4), c1[0].animate.set_stroke(C_GOOD, 5))
        self.play(*[m.animate.set_opacity(0.18) for m in [p2, c2, c3, e[1], e[3], e[4]]])
        keyq = Text("no per-candidate re-runs, no ties — one pass, one greedy path, always.",
                    font_size=20, color=C_GOOD).move_to([0, -2.95, 0])
        self.play(FadeIn(keyq))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in [h, tree, gl, gl2, cells, rlabs, one, acc, keyq]])

    # ---------------------------------------------------------------- #
    def closing(self):
        title = Text("Chained-Flow", weight=BOLD, font_size=40, color=C_FLOW).to_edge(UP, buff=1.0)
        rows = VGroup(
            Text("one flow pass  →  a whole draft tree", font_size=26, color=C_FLOW),
            Text("token feedback (Markov + path)  →  at zero sequential cost", font_size=24, color=C_VAE),
            Text("one backbone pass, one greedy path  →  lossless", font_size=24, color=C_GOOD),
        ).arrange(DOWN, buff=0.35).shift(UP * 0.1)
        res = Text("measured — 4B: accept 6.7   ·   9B: accept 4.9 → 1.27× lossless",
                   font_size=22, color=WHITE).next_to(rows, DOWN, buff=0.7)
        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in rows], lag_ratio=0.4), run_time=2.0)
        self.play(FadeIn(res, shift=UP * 0.15)); self.wait(2.2)
        self.play(FadeOut(VGroup(title, rows, res)))
