"""Ball-physics HUD overlay (toggle with H, together with the top-left panel).

Rows: ball Vz, ball Vxy, closest car -> ball relative Vxy (speeds in uu/s AND km/h), and the angle between the
ball's horizontal velocity and the nearer post of the opponent's goal. The panel is drawn with PIL into an RGBA
texture (rebuilt at most ~20 Hz, only when a displayed number changed) and blitted as one quad whose size follows
the window height, so it scales with the window and costs almost nothing."""
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
uniform float alpha;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(Tex, v_uv) * alpha; }      // premultiplied RGBA
"""

W, H_ROW, PAD, BAR = 380, 62, 14, 6            # base layout (px at SS=1)
ROWS = 4
H = PAD * 2 + H_ROW * ROWS
SS = 2                                         # supersampling of the baked texture
WHITE, GREY, DIM = (244, 246, 250), (150, 160, 178), (96, 104, 120)
GREEN, AMBER = (92, 224, 140), (255, 196, 84)
BLUE, ORANGE = (70, 140, 255), (255, 150, 40)


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
    out = {"vz": float(bv[2]), "vxy": math.hypot(bv[0], bv[1]), "rel": None, "ang": None,
           "on_target": False, "team": 0}
    team = 0
    if 0 <= spectated < len(state.car_states):
        team = int(state.car_states[spectated].team_num) & 1
    out["team"] = team
    best, bd = None, 1e18
    for c in state.car_states:
        if c.is_demoed:
            continue
        p = c.phys.get_pos(interp)
        d = (p[0] - bp[0]) ** 2 + (p[1] - bp[1]) ** 2 + (p[2] - bp[2]) ** 2
        if d < bd:
            best, bd = c, d
    if best is not None:
        cv = best.phys.get_vel(interp)
        out["rel"] = math.hypot(cv[0] - bv[0], cv[1] - bv[1])
    if out["vxy"] >= MIN_SPEED:
        gy = GOAL_Y if team == 0 else -GOAL_Y          # blue attacks +y (orange net)
        angs, crosses, dots = [], [], []
        for px in (-POST_X, POST_X):
            dx, dy = px - bp[0], gy - bp[1]
            cr = bv[0] * dy - bv[1] * dx
            dt = bv[0] * dx + bv[1] * dy
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
        self.tex = None
        self._key = None
        self._t = 0.0
        self.f_lab = _font(("segoeuib.ttf", "arialbd.ttf"), 13 * SS)
        self.f_val = _font(("segoeuisb.ttf", "seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"), 29 * SS)
        self.f_unit = _font(("segoeui.ttf", "arial.ttf"), 14 * SS)
        self.f_kph = _font(("segoeuisb.ttf", "segoeuib.ttf", "arialbd.ttf"), 20 * SS)

    # -- baking ------------------------------------------------------------------------------- #
    @staticmethod
    def _fmt(v, signed=False):
        return ("{:+,.0f}" if signed else "{:,.0f}").format(v).replace("-", "\u2212")

    def _rows(self, d):
        def speed(v, signed=False):
            if v is None:
                return "\u2014", "", ""
            return self._fmt(v, signed), "uu/s", "{} km/h".format(self._fmt(v * UU_TO_KPH, signed).replace(",", ""))
        rows = [("BALL  Vz", *speed(d["vz"], True), WHITE),
                ("BALL  Vxy", *speed(d["vxy"]), WHITE),
                ("NEAREST CAR \u2194 BALL  Vxy", *speed(d["rel"]), WHITE)]
        if d["ang"] is None:
            rows.append(("BALL HEADING \u2192 %s NET" % ("ORANGE" if d["team"] == 0 else "BLUE"), "\u2014", "", "", WHITE))
        else:
            tag = "ON TARGET" if d["on_target"] else "MISSING"
            rows.append(("BALL HEADING \u2192 %s NET" % ("ORANGE" if d["team"] == 0 else "BLUE"),
                         "{:.1f}\u00b0".format(d["ang"]), "to nearest post", tag, GREEN if d["on_target"] else AMBER))
        return rows

    def _bake(self, d):
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        g = ImageDraw.Draw(img)
        accent = BLUE if d["team"] == 0 else ORANGE
        g.rounded_rectangle((0, 0, W * SS - 1, H * SS - 1), radius=14 * SS, fill=(10, 14, 22, 168),
                            outline=(255, 255, 255, 34), width=SS)
        g.rounded_rectangle((PAD * SS * 0.6, PAD * SS, PAD * SS * 0.6 + BAR * SS * 0.5, (H - PAD) * SS),
                            radius=2 * SS, fill=accent + (235,))
        x0 = (PAD + BAR + 6) * SS
        x1 = (W - PAD) * SS
        for i, (lab, val, unit, extra, col) in enumerate(self._rows(d)):
            y = (PAD + i * H_ROW) * SS
            if i:
                g.line((x0, y - 2 * SS, x1, y - 2 * SS), fill=(255, 255, 255, 22), width=SS)
            g.text((x0, y + 3 * SS), lab, font=self.f_lab, fill=GREY)
            g.text((x0, y + 19 * SS), val, font=self.f_val, fill=col)
            vw = g.textlength(val, font=self.f_val)
            if unit:
                g.text((x0 + vw + 6 * SS, y + 33 * SS), unit, font=self.f_unit, fill=DIM)
            if extra:
                ec = col if i == 3 else accent
                tw = g.textlength(extra, font=self.f_kph)
                g.text((x1 - tw, y + 28 * SS), extra, font=self.f_kph, fill=ec)
        a = np.asarray(img, "f4") / 255.0
        a[..., :3] *= a[..., 3:4]                                   # premultiply
        a = np.ascontiguousarray(a[::-1])                           # GL origin is bottom-left
        data = (a * 255.0 + 0.5).astype("u1").tobytes()
        if self.tex is None:
            self.tex = self.ctx.texture((W * SS, H * SS), 4, data)
            self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.tex.anisotropy = 1.0
        else:
            self.tex.write(data)
        self.tex.build_mipmaps()

    # -- drawing ------------------------------------------------------------------------------ #
    def render(self, target, state, interp, spectated, width, height):
        d = compute(state, interp, spectated)
        key = (round(d["vz"] * UU_TO_KPH), round(d["vxy"] * UU_TO_KPH), None if d["rel"] is None else round(d["rel"] * UU_TO_KPH),
               None if d["ang"] is None else round(d["ang"], 1), d["on_target"], d["team"],
               round(d["vz"] / 4), round(d["vxy"] / 4), None if d["rel"] is None else round(d["rel"] / 4))
        now = time.monotonic()
        if key != self._key and (self.tex is None or now - self._t > 0.05):
            self._key, self._t = key, now
            self._bake(d)
        # size follows the window height (1080p -> 1.0 x base), kept within sane bounds of the width
        s = max(0.55, min(2.2, height / 1080.0))
        s = min(s, 0.42 * width / W)
        qw, qh = W * s, H * s
        m = 18.0 * s
        x0, y1 = m, height - m - 6 * s
        y0 = y1 - qh
        arr = np.asarray([(x0, y0, 0.0, 1.0), (x0 + qw, y0, 1.0, 1.0), (x0 + qw, y1, 1.0, 0.0),
                          (x0, y0, 0.0, 1.0), (x0 + qw, y1, 1.0, 0.0), (x0, y1, 0.0, 0.0)], "f4")
        self.vbo.write(arr.tobytes())
        self.prog["m_vp"].write(Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0).astype("f4"))
        self.prog["Tex"].value = 0
        self.prog["alpha"].value = 1.0
        target.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.tex.use(location=0)
        self.vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
