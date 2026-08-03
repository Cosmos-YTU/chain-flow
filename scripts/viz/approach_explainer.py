from manim import *

# ---- palette ----
C_BG    = "#0d1117"
C_BASE  = "#4f8bff"   # backbone
C_DRAFT = "#ffd166"   # drafter / tokens
C_FLOW  = "#2dd4bf"   # flow
C_VAE   = "#c792ea"   # vae / latent
C_GOOD  = "#6ee7a8"   # accept / good
C_BAD   = "#f78c6c"   # reject / bad
C_MUTE  = "#8b98a5"   # muted text

config.background_color = C_BG


def title_card(txt, sub=None, color=C_FLOW):
    t = Text(txt, weight=BOLD, font_size=42, color=color)
    g = VGroup(t)
    if sub:
        s = Text(sub, font_size=24, color=C_MUTE).next_to(t, DOWN, buff=0.25)
        g.add(s)
    return g.move_to(ORIGIN)


def sec_header(n, txt, color):
    h = Text(f"{n}  ·  {txt}", weight=BOLD, font_size=34, color=color)
    return h.to_edge(UP, buff=0.5)


def token(label, color=C_DRAFT, w=1.0, h=0.7, fill=0.18):
    b = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                         stroke_color=color, stroke_width=2.5,
                         fill_color=color, fill_opacity=fill)
    t = Text(label, font_size=22, color=WHITE).move_to(b)
    return VGroup(b, t)


def box(label, color, w=3.2, h=1.0, fs=24):
    b = RoundedRectangle(corner_radius=0.12, width=w, height=h,
                         stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=0.12)
    t = Text(label, font_size=fs, color=WHITE).move_to(b)
    return VGroup(b, t)


class Approach(Scene):
    def construct(self):
        self.opening()
        self.s1_chain()
        self.s2_learnings()
        self.s3_feedback()
        self.s4_flow()
        self.s5_vae()
        self.s6_overall()
        self.closing()

    # ------------------------------------------------------------------ #
    def opening(self):
        t = title_card("Chained-Flow", "speculative decoding that drafts a whole tree in one shot", C_FLOW)
        self.play(Write(t[0]), run_time=1.0)
        self.play(FadeIn(t[1], shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(t))

    # ------------------------------------------------------------------ #
    def s1_chain(self):
        h = sec_header("1", "Where we started: the chain", C_BASE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # speculative decoding idea
        back = box("Qwen backbone · frozen", C_BASE, w=4.6).shift(UP * 1.6)
        idea = Text("a small drafter proposes K tokens · the backbone verifies",
                    font_size=22, color=C_MUTE).next_to(back, DOWN, buff=0.35)
        self.play(FadeIn(back), FadeIn(idea))
        self.wait(0.8)
        self.play(FadeOut(idea))

        # the chain: sequential, one feeds the next
        toks = VGroup(*[token(t) for t in ["The", "cat", "sat", "on"]]).arrange(RIGHT, buff=0.9)
        toks.shift(DOWN * 0.4)
        arrows = VGroup()
        for a, b in zip(toks, toks[1:]):
            arrows.add(Arrow(a.get_right(), b.get_left(), buff=0.1, color=C_DRAFT, stroke_width=3))
        cap = Text("old repo: predict one, feed it back, predict the next … K sequential passes",
                   font_size=22, color=C_MUTE).next_to(toks, DOWN, buff=0.7)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in toks], lag_ratio=0.5), run_time=2.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.5), run_time=1.5)
        self.play(FadeIn(cap))
        self.wait(0.8)

        # fragility: one miss kills the rest
        x = Cross(toks[2], color=C_BAD, stroke_width=6).scale(0.5)
        frag = Text("one wrong token → the whole rest is wasted.  A chain is fragile.",
                    font_size=24, color=C_BAD).move_to(cap)
        self.play(Create(x))
        self.play(toks[3].animate.set_opacity(0.25), FadeOut(cap), FadeIn(frag))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in [h, back, toks, arrows, x, frag]])

    # ------------------------------------------------------------------ #
    def s2_learnings(self):
        h = sec_header("2", "Learnings: EAGLE & DSpark", C_DRAFT)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        lever = Text("Our measurements found ONE lever: token feedback",
                     font_size=26, color=WHITE).shift(UP * 1.9)
        self.play(FadeIn(lever, shift=UP * 0.2))
        self.wait(0.5)

        # EAGLE row
        eagle = box("EAGLE", C_BASE, w=2.4, h=0.9, fs=24).shift(LEFT * 3.4 + UP * 0.5)
        e_desc = VGroup(
            Text("condition each token on the previous one", font_size=21, color=C_MUTE),
            Text("+0.53 accept  ✓   but K sequential passes  ✗", font_size=21, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(eagle, RIGHT, buff=0.5)
        self.play(FadeIn(eagle), FadeIn(e_desc))
        self.wait(1.0)

        # DSpark row
        dspark = box("DSpark", C_VAE, w=2.4, h=0.9, fs=24).shift(LEFT * 3.4 + DOWN * 1.1)
        d_desc = VGroup(
            Text("a cheap low-rank token → logit bias", font_size=21, color=C_MUTE),
            Text("+0.25 accept  ✓   almost free  ✓", font_size=21, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(dspark, RIGHT, buff=0.5)
        self.play(FadeIn(dspark), FadeIn(d_desc))
        self.wait(1.0)

        take = Text("Goal: keep token feedback — without paying K sequential passes.",
                    font_size=24, color=C_GOOD).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(take, shift=UP * 0.2))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in [h, lever, eagle, e_desc, dspark, d_desc, take]])

    # ------------------------------------------------------------------ #
    def s3_feedback(self):
        h = sec_header("3", "Token feedback + the Markov head", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # markov head mechanism
        mh = Text("Markov head:   bias = W2( W1[ prev_token ] )", font_size=26, color=WHITE).shift(UP * 1.9)
        self.play(Write(mh))
        prev = token("prev", C_DRAFT).shift(LEFT * 4.2 + UP * 0.3)
        w1 = box("W1", C_VAE, w=1.1, h=0.8, fs=22).next_to(prev, RIGHT, buff=0.6)
        rank = token("rank r", C_VAE, w=1.3).next_to(w1, RIGHT, buff=0.6)
        w2 = box("W2", C_VAE, w=1.1, h=0.8, fs=22).next_to(rank, RIGHT, buff=0.6)
        bias = token("+bias", C_GOOD, w=1.4).next_to(w2, RIGHT, buff=0.6)
        row = VGroup(prev, w1, rank, w2, bias)
        ars = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, color=C_MUTE, stroke_width=3)
                       for a, b in zip(row[:-1], row[1:])])
        self.play(FadeIn(prev))
        self.play(LaggedStart(GrowArrow(ars[0]), FadeIn(w1), GrowArrow(ars[1]), FadeIn(rank),
                              GrowArrow(ars[2]), FadeIn(w2), GrowArrow(ars[3]), FadeIn(bias), lag_ratio=0.5), run_time=2.6)
        note = Text("low rank r ≪ vocab · starts at zero · learns \"after X, boost these\"",
                    font_size=21, color=C_MUTE).next_to(row, DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.2)

        fb = Text("Path conditioning: each branch feeds its chosen token back as a cheap correction —",
                  font_size=22, color=WHITE).shift(DOWN * 1.5)
        fb2 = Text("the same token-feedback lever, at zero sequential cost.",
                   font_size=22, color=C_GOOD).next_to(fb, DOWN, buff=0.15)
        self.play(FadeIn(fb), FadeIn(fb2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, mh, row, ars, note, fb, fb2]])

    # ------------------------------------------------------------------ #
    def s4_flow(self):
        h = sec_header("4", "Our flow drafter", C_FLOW)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # one flow pass -> all K hiddens at once
        ctx = box("context hidden", C_BASE, w=3.0, h=0.9, fs=22).shift(LEFT * 4.3 + UP * 1.5)
        # flow arrow with steps
        flow = Arrow(ctx.get_right(), ctx.get_right() + RIGHT * 2.4, buff=0.1, color=C_FLOW, stroke_width=5)
        flab = Text("flow\n(2 steps)", font_size=18, color=C_FLOW).next_to(flow, UP, buff=0.1)
        hiddens = VGroup(*[token(f"ĥ{i}", C_FLOW, w=0.75, h=0.6) for i in range(1, 6)])
        hiddens.arrange(RIGHT, buff=0.2).next_to(flow, RIGHT, buff=0.3)
        self.play(FadeIn(ctx))
        self.play(GrowArrow(flow), FadeIn(flab))
        self.play(LaggedStart(*[FadeIn(x, shift=RIGHT * 0.1) for x in hiddens], lag_ratio=0.15), run_time=1.4)
        one = Text("ONE parallel pass predicts all K future hiddens — not K sequential steps",
                   font_size=22, color=WHITE).shift(UP * 0.25)
        self.play(FadeIn(one))
        self.wait(1.2)

        # hiddens -> logits -> top-b -> tree
        # build a small tree
        root = token("The", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 4.5 + DOWN * 1.6)
        n1 = token("cat", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 2.4 + DOWN * 0.9)
        n2 = token("dog", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 2.4 + DOWN * 2.3)
        n3 = token("sat", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 0.3 + DOWN * 0.5)
        n4 = token("ran", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 0.3 + DOWN * 1.3)
        n5 = token("ate", C_DRAFT, w=0.9, h=0.6).shift(LEFT * 0.3 + DOWN * 2.3)
        edges = VGroup(
            Line(root.get_right(), n1.get_left(), color=C_MUTE),
            Line(root.get_right(), n2.get_left(), color=C_MUTE),
            Line(n1.get_right(), n3.get_left(), color=C_MUTE),
            Line(n1.get_right(), n4.get_left(), color=C_MUTE),
            Line(n2.get_right(), n5.get_left(), color=C_MUTE),
        )
        tree = VGroup(edges, root, n1, n2, n3, n4, n5)
        tcap = Text("keep top-b at each hidden → branch into a draft TREE (one pass = a whole tree)",
                    font_size=21, color=C_FLOW).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(root))
        self.play(Create(edges), LaggedStart(*[FadeIn(n) for n in [n1, n2, n3, n4, n5]], lag_ratio=0.2), run_time=2.0)
        self.play(FadeIn(tcap))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, ctx, flow, flab, hiddens, one, tree, tcap]])

    # ------------------------------------------------------------------ #
    def s5_vae(self):
        h = sec_header("5", "The joint VAE", C_VAE)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        prob = Text("Flow in full hidden space (2560–5120 dim) is expensive and base-specific.",
                    font_size=23, color=C_MUTE).shift(UP * 2.0)
        self.play(FadeIn(prob))

        # encode -> small latent -> flow -> decode
        big = box("hidden 4096", C_BASE, w=2.6, h=1.0, fs=22).shift(LEFT * 4.6 + UP * 0.3)
        enc = Text("encode", font_size=18, color=C_VAE)
        lat = box("latent 1024", C_VAE, w=2.0, h=0.8, fs=20).shift(LEFT * 1.2 + UP * 0.3)
        a1 = Arrow(big.get_right(), lat.get_left(), buff=0.15, color=C_VAE, stroke_width=4)
        enc.next_to(a1, UP, buff=0.08)
        flow = Text("flow", font_size=18, color=C_FLOW)
        lat2 = box("latent 1024", C_VAE, w=2.0, h=0.8, fs=20).shift(RIGHT * 1.7 + UP * 0.3)
        a2 = Arrow(lat.get_right(), lat2.get_left(), buff=0.15, color=C_FLOW, stroke_width=4)
        flow.next_to(a2, UP, buff=0.08)
        dec = Text("decode", font_size=18, color=C_VAE)
        big2 = box("K hiddens", C_FLOW, w=2.2, h=1.0, fs=20).shift(RIGHT * 4.7 + UP * 0.3)
        a3 = Arrow(lat2.get_right(), big2.get_left(), buff=0.15, color=C_VAE, stroke_width=4)
        dec.next_to(a3, UP, buff=0.08)
        row = VGroup(big, a1, enc, lat, a2, flow, lat2, a3, dec, big2)
        self.play(FadeIn(big))
        self.play(GrowArrow(a1), FadeIn(enc), FadeIn(lat))
        self.play(GrowArrow(a2), FadeIn(flow), FadeIn(lat2))
        self.play(GrowArrow(a3), FadeIn(dec), FadeIn(big2))
        cheap = Text("flow runs in the tiny latent → cheap and portable across model sizes",
                     font_size=21, color=C_MUTE).next_to(row, DOWN, buff=0.5)
        self.play(FadeIn(cheap))
        self.wait(1.0)

        # frozen failed -> joint won
        fail = Text("Frozen VAE (trained for reconstruction) failed:  9B recon 0.53 → accept 0",
                    font_size=22, color=C_BAD).shift(DOWN * 1.6)
        self.play(FadeIn(fail))
        self.wait(1.0)
        win = Text("Fix: train the VAE UNFROZEN, jointly, for ACCEPT — with a reconstruction anchor.",
                   font_size=22, color=WHITE).move_to(fail)
        win2 = Text("9B → accept 4.9 · 1.27× lossless",
                    font_size=24, color=C_GOOD).next_to(win, DOWN, buff=0.2)
        self.play(FadeOut(fail), FadeIn(win))
        self.play(FadeIn(win2, shift=UP * 0.2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in [h, prob, row, cheap, win, win2]])

    # ------------------------------------------------------------------ #
    def s6_overall(self):
        h = sec_header("6", "Draft → Verify", C_GOOD)
        self.play(FadeIn(h, shift=DOWN * 0.2))

        # DRAFT side
        draft_t = Text("DRAFT (cheap)", font_size=22, color=C_FLOW).shift(LEFT * 3.6 + UP * 2.1)
        steps = VGroup(
            Text("encode context → latent", font_size=20, color=C_MUTE),
            Text("one flow pass → K hiddens", font_size=20, color=C_MUTE),
            Text("top-b + token feedback → tree", font_size=20, color=C_MUTE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(draft_t, DOWN, buff=0.3).shift(RIGHT*0.2)
        self.play(FadeIn(draft_t))
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.1) for s in steps], lag_ratio=0.4), run_time=1.8)

        # small tree
        root = token("x", C_DRAFT, w=0.7, h=0.55).shift(RIGHT * 1.3 + UP * 1.4)
        a = token("a", C_DRAFT, w=0.7, h=0.55).shift(RIGHT * 3.0 + UP * 2.0)
        b = token("b", C_DRAFT, w=0.7, h=0.55).shift(RIGHT * 3.0 + UP * 0.8)
        c = token("c", C_DRAFT, w=0.7, h=0.55).shift(RIGHT * 4.7 + UP * 2.0)
        edges = VGroup(Line(root.get_right(), a.get_left(), color=C_MUTE),
                       Line(root.get_right(), b.get_left(), color=C_MUTE),
                       Line(a.get_right(), c.get_left(), color=C_MUTE))
        tree = VGroup(edges, root, a, b, c)
        self.play(Create(edges), *[FadeIn(n) for n in [root, a, b, c]])
        self.wait(0.6)

        # VERIFY side: backbone one forward, accept longest path
        verify = box("backbone · ONE forward (tree attention)", C_BASE, w=6.8, h=0.9, fs=22).shift(DOWN * 0.7)
        self.play(FadeIn(verify))
        # highlight accepted path root->a->c
        acc = VGroup(root[0].copy(), a[0].copy(), c[0].copy(), edges[0].copy(), edges[2].copy())
        self.play(root[0].animate.set_stroke(C_GOOD, 5), a[0].animate.set_stroke(C_GOOD, 5),
                  c[0].animate.set_stroke(C_GOOD, 5), edges[0].animate.set_color(C_GOOD),
                  edges[2].animate.set_color(C_GOOD),
                  b[0].animate.set_opacity(0.2), b[1].animate.set_opacity(0.2))
        res = Text("accept the longest correct root→leaf path · lossless (frozen model has the final say)",
                   font_size=21, color=C_GOOD).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(res))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in [h, draft_t, steps, tree, verify, res]])

    # ------------------------------------------------------------------ #
    def closing(self):
        l1 = Text("one flow pass  →  a whole draft tree", font_size=30, color=C_FLOW)
        l2 = Text("token feedback, at zero sequential cost", font_size=24, color=C_VAE)
        l3 = Text("verified losslessly by the frozen model", font_size=24, color=C_GOOD)
        g = VGroup(l1, l2, l3).arrange(DOWN, buff=0.35).move_to(ORIGIN)
        self.play(Write(l1))
        self.play(FadeIn(l2, shift=UP * 0.15))
        self.play(FadeIn(l3, shift=UP * 0.15))
        self.wait(2.0)
        self.play(FadeOut(g))
