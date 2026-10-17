"""Ball-physics HUD overlay (toggle with H, together with the top-left panel).

Rows: ball Vz, ball Vxy, closest car -> ball relative Vxy (speeds in uu/s AND km/h), and the angle between the
ball's horizontal velocity and the nearer post of the opponent's goal.

Zero PIL / texture uploads at runtime: the panel (background, labels, separators) is baked ONCE per team into a
texture and the changing numbers are glyph quads from a one-off atlas, rebuilt only when a displayed (rounded) value
changes and uploaded in a single tiny vertex write. The panel size follows the window height."""
import math
import os
import time

import moderngl
import numpy as np
from pyrr import Matrix44

UU_TO_KPH = 0.036            # 1 uu = 1 cm  ->  uu/s * 0.01 m/s * 3.6
POST_X = 892.755
GOAL_Y = 5120.0
MIN_SPEED = 25.0             # below this the heading is meaningless

_VERT = """
#version 330
uniform mat4 m_vp;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = m_vp * vec4(in_pos, 0.0, 1.0); }
"""
_FRAG = """
#version 330
uniform sampler2D Tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(Tex, v_uv); }      // premultiplied RGBA
"""
_GVERT = """
#version 330
uniform mat4 m_vp;
uniform vec3 place;       // x0, y0, scale
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_col;
out vec2 v_uv;
out vec4 v_col;
void main() { v_uv = in_uv; v_col = in_col; gl_Position = m_vp * vec4(place.xy + in_pos * place.z, 0.0, 1.0); }
"""
_GFRAG = """
#version 330
uniform sampler2D Tex;
in vec2 v_uv;
in vec4 v_col;
out vec4 f_color;
void main() { float a = texture(Tex, v_uv).r * v_col.a; f_color = vec4(v_col.rgb * a, a); }
"""

W, H_ROW, PAD, BAR = 380, 62, 14, 6            # base layout (px at SS=1)
ROWS = 4
H = PAD * 2 + H_ROW * ROWS
SS = 2                                         # supersampling of the baked textures
WHITE, GREY, DIM = (244, 246, 250), (150, 160, 178), (96, 104, 120)
GREEN, AMBER = (92, 224, 140), (255, 196, 84)
BLUE, ORANGE = (70, 140, 255), (255, 150, 40)
GLYPHS = "0123456789+−.,°— /kmhuUsABCDEFGHIJKLMNOPQRSTVWXYZ%"
MINUS = "\u2212"


def _font(names, size):
    from PIL import ImageFont
    d = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(d, n), size)
        except OSError:
            pass
    return ImageFont.load_default()


def compute(state, interp, spectated):
    """-> dict of the HUD numbers (uu/s unless noted), None where undefined."""
    b = state.ball_state
    bv = b.get_vel(interp)
    bp = b.get_pos(interp)
    bvx, bvy, bvz = float(bv[0]), float(bv[1]), float(bv[2])
    bpx, bpy, bpz = float(bp[0]), float(bp[1]), float(bp[2])
    out = {"vz": bvz, "vxy": math.hypot(bvx, bvy), "rel": None, "ang": None, "on_target": False, "team": 0}
    team = 0
    if 0 <= spectated < len(state.car_states):
        team = int(state.car_states[spectated].team_num) & 1
    out["team"] = team
    best, bd = None, 1e18
    for c in state.car_states:
        if c.is_demoed:
            continue
        p = c.phys.get_pos(interp)
        d = (p[0] - bpx) ** 2 + (p[1] - bpy) ** 2 + (p[2] - bpz) ** 2
        if d < bd:
            best, bd = c, d
    if best is not None:
        cv = best.phys.get_vel(interp)
        out["rel"] = math.hypot(cv[0] - bvx, cv[1] - bvy)
    if out["vxy"] >= MIN_SPEED:
        gy = GOAL_Y if team == 0 else -GOAL_Y          # blue attacks +y (orange net)
        angs, crosses, dots = [], [], []
        for px in (-POST_X, POST_X):
            dx, dy = px - bpx, gy - bpy
            cr = bvx * dy - bvy * dx
            dt = bvx * dx + bvy * dy
            angs.append(abs(math.atan2(cr, dt)))
            crosses.append(cr)
            dots.append(dt)
        out["ang"] = math.degrees(min(angs))
        out["on_target"] = crosses[0] * crosses[1] < 0 and max(dots) > 0
    return out


class PhysHud:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.vbo = ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")])
        self.gprog = ctx.program(vertex_shader=_GVERT, fragment_shader=_GFRAG)
        self.gvbo = ctx.buffer(reserve=6 * 8 * 4 * 128, dynamic=True)
        self.gvao = ctx.vertex_array(self.gprog, [(self.gvbo, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.f_lab = _font(("segoeuib.ttf", "arialbd.ttf"), 13 * SS)
        self.f_val = _font(("segoeuisb.ttf", "seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"), 29 * SS)
        self.f_unit = _font(("segoeui.ttf", "arial.ttf"), 14 * SS)
        self.f_kph = _font(("segoeuisb.ttf", "segoeuib.ttf", "arialbd.ttf"), 20 * SS)
        self.panels = {t: self._bake_panel(t) for t in (0, 1)}
        self._build_atlas()
        self._cache_key = None
        self._t = 0.0
        self._n = 0

    # -- one-off bakes ------------------------------------------------------------------------ #
    @staticmethod
    def _rgba_bytes(img):
        a = np.asarray(img, "f4") / 255.0
        a[..., :3] *= a[..., 3:4]                                   # premultiply
        a = np.ascontiguousarray(a[::-1])                           # GL origin is bottom-left
        return (a * 255.0 + 0.5).astype("u1").tobytes()

    def _bake_panel(self, team):
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        g = ImageDraw.Draw(img)
        accent = BLUE if team == 0 else ORANGE
        g.rounded_rectangle((0, 0, W * SS - 1, H * SS - 1), radius=14 * SS, fill=(10, 14, 22, 168),
                            outline=(255, 255, 255, 34), width=SS)
        g.rounded_rectangle((PAD * SS * 0.6, PAD * SS, PAD * SS * 0.6 + BAR * SS * 0.5, (H - PAD) * SS),
                            radius=2 * SS, fill=accent + (235,))
        x0, x1 = (PAD + BAR + 6) * SS, (W - PAD) * SS
        net = "ORANGE" if team == 0 else "BLUE"
        labs = ("BALL  Vz", "BALL  Vxy", "NEAREST CAR \u2194 BALL  Vxy", "BALL HEADING \u2192 %s NET" % net)
        for i, lab in enumerate(labs):
            y = (PAD + i * H_ROW) * SS
            if i:
                g.line((x0, y - 2 * SS, x1, y - 2 * SS), fill=(255, 255, 255, 22), width=SS)
            g.text((x0, y + 3 * SS), lab, font=self.f_lab, fill=GREY)
        t = self.ctx.texture((W * SS, H * SS), 4, self._rgba_bytes(img))
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        t.build_mipmaps()
        return t

    def _build_atlas(self):
        from PIL import Image, ImageDraw
        fonts = {"val": self.f_val, "unit": self.f_unit, "kph": self.f_kph}
        cw, rh = 40 * SS, 48 * SS
        img = Image.new("L", (cw * len(GLYPHS), rh * len(fonts)), 0)
        g = ImageDraw.Draw(img)
        self.glyph = {}
        for r, (name, f) in enumerate(fonts.items()):
            m = {}
            for k, ch in enumerate(GLYPHS):
                adv = g.textlength(ch, font=f)
                if ch.strip():
                    l, t, rr, b = f.getbbox(ch, anchor="ls")
                    ox, oy = k * cw + 4, r * rh + 38 * SS
                    g.text((ox, oy), ch, font=f, fill=255, anchor="ls")
                    # quad = glyph box grown by 2 px; atlas v measured from the TOP of the image
                    m[ch] = (adv, l - 2, t - 2, (ox + l - 2) / img.width, (oy + t - 2) / img.height,
                             (ox + rr + 2) / img.width, (oy + b + 2) / img.height, rr - l + 4, b - t + 4)
                else:
                    m[ch] = (adv, 0, 0, 0, 0, 0, 0, 0, 0)
            self.glyph[name] = m
        self.atlas = self.ctx.texture(img.size, 1, np.ascontiguousarray(np.asarray(img)[::-1]).tobytes())
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # -- text as quads (baked-texture pixel units, baseline at by) ----------------------------- #
    def _width(self, font, text):
        m = self.glyph[font]
        return sum(m[c][0] for c in text if c in m)

    def _quads(self, out, font, text, bx, by, col):
        m = self.glyph[font]
        x = bx
        for c in text:
            e = m.get(c)
            if e is None:
                continue
            adv, l, t, u0, v0, u1, v1, w, h = e
            if w:
                qx, qy = x + l, by + t
                fv0, fv1 = 1.0 - v0, 1.0 - v1                       # texture row 0 = image bottom
                out += [(qx, qy, u0, fv0) + col, (qx + w, qy, u1, fv0) + col, (qx + w, qy + h, u1, fv1) + col,
                        (qx, qy, u0, fv0) + col, (qx + w, qy + h, u1, fv1) + col, (qx, qy + h, u0, fv1) + col]
            x += adv

    @staticmethod
    def _fmt(v, signed=False):
        return ("{:+,.0f}" if signed else "{:,.0f}").format(v).replace("-", MINUS)

    def _layout(self, d):
        x0, x1 = (PAD + BAR + 6) * SS, (W - PAD) * SS
        out = []
        accent = BLUE if d["team"] == 0 else ORANGE

        def c(rgb):
            return tuple(v / 255.0 for v in rgb) + (1.0,)

        def speed(i, v, signed=False):
            by = (PAD + i * H_ROW + 45) * SS
            if v is None:
                self._quads(out, "val", "\u2014", x0, by, c(WHITE))
                return
            val = self._fmt(v, signed)
            self._quads(out, "val", val, x0, by, c(WHITE))
            self._quads(out, "unit", "uu/s", x0 + self._width("val", val) + 6 * SS, by, c(DIM))
            k = self._fmt(v * UU_TO_KPH, signed).replace(",", "") + " km/h"
            self._quads(out, "kph", k, x1 - self._width("kph", k), by, c(accent))
        speed(0, d["vz"], True)
        speed(1, d["vxy"])
        speed(2, d["rel"])
        by = (PAD + 3 * H_ROW + 45) * SS
        if d["ang"] is None:
            self._quads(out, "val", "\u2014", x0, by, c(WHITE))
        else:
            col = GREEN if d["on_target"] else AMBER
            self._quads(out, "val", "{:.1f}\u00b0".format(d["ang"]), x0, by, c(col))
            tag = "ON TARGET" if d["on_target"] else "MISSING"
            self._quads(out, "kph", tag, x1 - self._width("kph", tag), by, c(col))
        return np.asarray(out, "f4")

    # -- drawing ------------------------------------------------------------------------------ #
    def render(self, target, state, interp, spectated, width, height):
        d = compute(state, interp, spectated)
        key = (round(d["vz"] * UU_TO_KPH), round(d["vxy"] * UU_TO_KPH),
               None if d["rel"] is None else round(d["rel"] * UU_TO_KPH),
               None if d["ang"] is None else round(d["ang"], 1), d["on_target"], d["team"],
               round(d["vz"] / 4), round(d["vxy"] / 4), None if d["rel"] is None else round(d["rel"] / 4))
        now = time.monotonic()
        if key != self._cache_key and now - self._t >= 1.0 / 15.0:     # text rebuilt at <= 15 Hz, only on change
            self._cache_key, self._t = key, now
            g = self._layout(d)
            self._n = len(g)
            self.gvbo.write(g.tobytes())
        s = max(0.55, min(2.2, height / 1080.0))                    # scales with the window height
        s = min(s, 0.42 * width / W)
        qw, qh = W * s, H * s
        m = 18.0 * s
        x0, y1 = m, height - m - 6 * s
        y0 = y1 - qh
        self.vbo.write(np.asarray([(x0, y0, 0.0, 1.0), (x0 + qw, y0, 1.0, 1.0), (x0 + qw, y1, 1.0, 0.0),
                                   (x0, y0, 0.0, 1.0), (x0 + qw, y1, 1.0, 0.0), (x0, y1, 0.0, 0.0)], "f4").tobytes())
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0).astype("f4").tobytes()
        target.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog["m_vp"].write(ortho)
        self.prog["Tex"].value = 0
        self.panels[d["team"]].use(location=0)
        self.vao.render(moderngl.TRIANGLES, vertices=6)
        self.gprog["m_vp"].write(ortho)
        self.gprog["place"].value = (x0, y0, s / SS)
        self.gprog["Tex"].value = 0
        self.atlas.use(location=0)
        self.gvao.render(moderngl.TRIANGLES, vertices=self._n)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
