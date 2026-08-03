"""Manim explainer: how the V9 flow-tree drafter builds a whole draft tree from ONE flow pass.

Render:  manim -qm scripts/viz/flow_tree_manim.py FlowTree
"""
from manim import *

BG   = "#0f1117"
INK  = "#e8eaed"
MUT  = "#9aa0aa"
CTX  = "#34a0c4"
FLOW = "#e8913e"
NODE = "#46c07a"
ALT  = "#c98394"
ACC  = "#a07fe0"
MARK = "#e06a82"


def cell(label, color, w=0.9, h=0.9, fs=20):
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h,
                           color=color, fill_color=color, fill_opacity=0.22, stroke_width=2.5)
    txt = Text(label, font_size=fs, color=INK)
    return VGroup(box, txt)


def chip(label, color=NODE, w=1.15, h=0.42, fs=17):
    box = RoundedRectangle(corner_radius=0.09, width=w, height=h,
                           color=color, fill_color=color, fill_opacity=0.28, stroke_width=2)
    txt = Text(label, font_size=fs, color=INK)
    return VGroup(box, txt)


class FlowTree(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---------- title ----------
        title = Text("Flow Draft Tree", font_size=46, color=INK, weight=BOLD)
        sub = Text("one flow pass  →  a whole draft tree", font_size=27, color=FLOW)
        grp = VGroup(title, sub).arrange(DOWN, buff=0.28)
        self.play(Write(title), run_time=1.1)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(grp.animate.scale(0.42).to_corner(UL), run_time=0.9)

        # ---------- persistent counters ----------
        passes = VGroup(Text("flow passes:", font_size=22, color=MUT),
                        Text("1", font_size=30, color=FLOW, weight=BOLD)).arrange(RIGHT, buff=0.15).to_corner(UR).shift(DOWN*0.1)
        nodes_n = [1]
        nodes_lbl = VGroup(Text("tree nodes:", font_size=22, color=MUT),
                           Text("1", font_size=30, color=NODE, weight=BOLD)).arrange(RIGHT, buff=0.15)
        nodes_lbl.next_to(passes, DOWN, aligned_edge=RIGHT, buff=0.2)
        self.play(FadeIn(passes), FadeIn(nodes_lbl))

        def set_nodes(n):
            new = Text(str(n), font_size=30, color=NODE, weight=BOLD).move_to(nodes_lbl[1]).align_to(nodes_lbl[1], LEFT)
            self.play(Transform(nodes_lbl[1], new), run_time=0.3)

        # ---------- context -> flow -> K hiddens (ONE pass) ----------
        ctx = VGroup(*[cell(f"c{i}", CTX, w=0.7, h=0.7, fs=16) for i in range(1, 4)])
        ctx.arrange(DOWN, buff=0.2).move_to(LEFT * 5.4 + DOWN * 0.3)
        ctx_lbl = Text("context", font_size=18, color=MUT).next_to(ctx, DOWN, buff=0.2)

        flowbox = RoundedRectangle(corner_radius=0.18, width=1.7, height=2.4,
                                   color=FLOW, fill_color=FLOW, fill_opacity=0.18, stroke_width=3).move_to(LEFT * 3.1 + DOWN*0.3)
        flowtxt = Text("FLOW\n(1 pass)", font_size=20, color=FLOW, weight=BOLD, line_spacing=0.6).move_to(flowbox)

        self.play(FadeIn(ctx), FadeIn(ctx_lbl))
        self.play(Create(flowbox), Write(flowtxt))
        self.play(*[GrowArrow(Arrow(c.get_right(), flowbox.get_left(), buff=0.12,
                                    stroke_width=3, color=MUT, max_tip_length_to_length_ratio=0.15)) for c in ctx],
                  run_time=0.7)

        xs = [-0.9, 1.1, 3.1, 5.1]           # depth positions, reused by the tree below
        y_h = 2.35
        hids = VGroup(*[cell(f"h{i+1}", FLOW, w=0.95, h=0.95, fs=22).move_to([xs[i], y_h, 0]) for i in range(4)])
        # emphasize: all K appear together from the single pass
        beam = [Arrow(flowbox.get_right(), h.get_left(), buff=0.1, stroke_width=3,
                      color=FLOW, max_tip_length_to_length_ratio=0.1) for h in hids]
        self.play(LaggedStart(*[GrowArrow(a) for a in beam], lag_ratio=0.05), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(h, scale=0.6) for h in hids], lag_ratio=0.08), run_time=0.8)
        note1 = Text("one pass → all K positions' hidden states at once", font_size=24, color=INK).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note1, shift=UP*0.15))
        self.play(Indicate(passes[1], color=FLOW, scale_factor=1.25))
        self.wait(1.1)
        self.play(FadeOut(note1))

        # keep hiddens + a faint depth guide; fade the flow machinery to declutter
        self.play(FadeOut(ctx), FadeOut(ctx_lbl), FadeOut(flowbox), FadeOut(flowtxt),
                  *[FadeOut(a) for a in beam],
                  *[FadeOut(a) for a in self.mobjects if isinstance(a, Arrow) and a not in beam])
        guides = VGroup(*[DashedLine([xs[i], y_h-0.55, 0], [xs[i], -2.6, 0], color=MUT, stroke_width=1.2, dash_length=0.12)
                          for i in range(4)])
        dlabels = VGroup(*[Text(f"depth {i+1}", font_size=16, color=MUT).move_to([xs[i], -2.85, 0]) for i in range(4)])
        self.play(*[Create(g) for g in guides], *[FadeIn(d) for d in dlabels], run_time=0.7)

        note2 = Text("every tree node at depth d is scored from the SAME hidden  h_d  — no new flow pass",
                     font_size=23, color=NODE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note2, shift=UP*0.15))

        # ---------- build the tree, depth by depth, reusing h_d ----------
        # node positions: (x index, y)
        A = chip("42", NODE).move_to([xs[0], 0.9, 0])
        self.play(FadeIn(A, scale=0.7))
        set_nodes(1)

        # depth 2: two children of A (branch!)
        B = chip("is", NODE).move_to([xs[1], 1.5, 0])
        C = chip("was", ALT).move_to([xs[1], 0.3, 0])
        e_ab = Line(A.get_right(), B.get_left(), color=MUT, stroke_width=2.5)
        e_ac = Line(A.get_right(), C.get_left(), color=MUT, stroke_width=2.5)
        # a copy of h2 glides down to show reuse
        h2copy = hids[1].copy()
        self.play(h2copy.animate.scale(0.5).move_to([xs[1], 0.9, 0]).set_opacity(0.5), run_time=0.7)
        tag2 = Text("markov + path  (cheap re-score)", font_size=17, color=MARK).move_to([xs[1], -1.9, 0])
        self.play(Create(e_ab), Create(e_ac), FadeIn(B, scale=0.7), FadeIn(C, scale=0.7), FadeIn(tag2))
        self.play(FadeOut(h2copy))
        set_nodes(3)
        self.play(Indicate(passes[1], color=FLOW, scale_factor=1.15))

        # depth 3
        D = chip("the", NODE).move_to([xs[2], 1.85, 0])
        E = chip("a", ALT).move_to([xs[2], 0.95, 0])
        F = chip("not", ALT).move_to([xs[2], -0.4, 0])
        e_bd = Line(B.get_right(), D.get_left(), color=MUT, stroke_width=2.5)
        e_be = Line(B.get_right(), E.get_left(), color=MUT, stroke_width=2.5)
        e_cf = Line(C.get_right(), F.get_left(), color=MUT, stroke_width=2.5)
        self.play(FadeOut(tag2))
        self.play(Create(e_bd), Create(e_be), Create(e_cf),
                  FadeIn(D, scale=0.7), FadeIn(E, scale=0.7), FadeIn(F, scale=0.7))
        set_nodes(6)

        # depth 4
        G = chip("sum", NODE).move_to([xs[3], 2.1, 0])
        H = chip("total", ALT).move_to([xs[3], 1.3, 0])
        I = chip("first", ALT).move_to([xs[3], 0.5, 0])
        e_dg = Line(D.get_right(), G.get_left(), color=MUT, stroke_width=2.5)
        e_dh = Line(D.get_right(), H.get_left(), color=MUT, stroke_width=2.5)
        e_ei = Line(E.get_right(), I.get_left(), color=MUT, stroke_width=2.5)
        self.play(Create(e_dg), Create(e_dh), Create(e_ei),
                  FadeIn(G, scale=0.7), FadeIn(H, scale=0.7), FadeIn(I, scale=0.7))
        set_nodes(9)
        self.wait(0.4)
        self.play(FadeOut(note2))

        stay = Text("flow passes stayed at 1 the whole time  •  branches are just cheap conditioning",
                    font_size=22, color=FLOW).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(stay), Indicate(passes[1], color=FLOW, scale_factor=1.3))
        self.wait(1.3)
        self.play(FadeOut(stay))

        # ---------- verify: accept the longest correct path ----------
        acc_edges = VGroup(e_ab.copy(), e_bd.copy(), e_dg.copy()).set_color(ACC).set_stroke(width=6)
        acc_nodes = VGroup(A[0].copy(), B[0].copy(), D[0].copy(), G[0].copy()).set_stroke(color=ACC, width=4).set_fill(ACC, opacity=0.35)
        vtxt = Text("backbone verifies → accept the longest correct PATH", font_size=23, color=ACC).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(vtxt))
        self.play(Create(acc_edges), FadeIn(acc_nodes), run_time=1.2)
        self.wait(0.6)

        final = Text("1 flow pass  →  a tree that accepts ~5 tokens/step", font_size=30, color=INK, weight=BOLD)
        final.to_edge(DOWN, buff=1.4)
        self.play(FadeOut(vtxt), Write(final))
        self.wait(1.8)
