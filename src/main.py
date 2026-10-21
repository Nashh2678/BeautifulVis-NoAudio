import os
import json
import sys

# numpy's OpenBLAS pre-commits a scratch buffer per CPU thread at import (~480 MB of Windows commit
# on this 16-thread machine) -- the vis only does tiny matrix math, and training runs near the commit
# limit. Must be set before numpy is first imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# PyOpenGL checks glGetError after EVERY call by default (a round-trip each) -- pure overhead here.
import OpenGL
OpenGL.ERROR_CHECKING = False
import math
import random
import threading
import copy
import time

import fastvec  # fast pyrr Vector3 operators -- must come before any vector math
from const import *
from shaders import *
from arena_shaders import *
from socket_listener import SocketListener
from state_manager import *
import state_manager   # also need the module itself for LIVE attrs (pose_cam, hud_text, capture)
from ribbon import *
# outline_renderer (OutlineRenderer) is disabled -- not imported (it cost ~1 s at startup)
from clip_recorder import ClipRecorder
import ui
from ui import get_ui, QRSVWindow
from config import Config
import audio as rl_audio
import events as rl_events
import fx as rl_fx
import rl_shaders
import padmesh
import carrig
import landscape
import maps as rl_maps

import moderngl
import moderngl_window
import moderngl_window.loaders.scene.wavefront as wvf
from moderngl_window import resources
from moderngl_window.meta import TextureDescription

from PyQt5 import QtOpenGL, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QScreen

# (PyOpenGL's GL/GLU star imports cost ~0.6 s at startup and were only used for GL_LINES)

import numpy as np

from pyrr import Matrix44, Vector3, Vector4

import pywavefront

# Set RSV_PERF=1 to print render fps + per-stage timing (deepcopy / render / capture) to stdout.
PERF = bool(os.environ.get("RSV_PERF"))

# TODO: Move
def safe_normalize(vec: pyrr.Vector3):
    length = max(vec.length, 1e-6)
    return vec / length

# Scene render budget. The 3D scene is drawn into its own MSAA framebuffer at the window size, scaled
# down only if that exceeds RSV_MAX_RENDER_MP megapixels (default 2.1 ~= 1080p, cheap enough for integrated GPUs),
# then upscaled to the window; HUD/overlays are drawn afterwards at full native resolution so they stay crisp.
RENDER_SCALE = float(os.environ.get("RSV_RENDER_SCALE", "1.0"))
MAX_RENDER_MP = float(os.environ.get("RSV_MAX_RENDER_MP", "2.1"))
MSAA = int(os.environ.get("RSV_MSAA", "4"))
_SKIP = set(os.environ.get("RSV_SKIP", "").split(","))   # dev: skip passes to profile GPU cost
TEAM_BODY = [(0.030, 0.20, 0.95), (1.0, 0.30, 0.02)]      # linear-space team paint (blue, orange)


def _icosphere(subdiv):
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, "f8") / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    ico_faces = list(faces)
    for _ in range(subdiv):
        cache = {}

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[k] = len(verts) - 1
            return cache[k]
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    v = np.array(verts, "f4")
    base = v[:12]
    centres = np.array([v[list(f)].mean(0) for f in ico_faces], "f4")
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return v, np.array(faces, "i4"), np.concatenate([base, centres], 0)


# TODO: Move game logic out of here
def _streak_points(pos):
    """The car MODEL's tips the flip streaks come off (read from the loaded mesh, car space): both spoiler tips
    (rear, outermost and highest) and both front fender corners (front, outermost and highest)."""
    P = np.asarray(pos, "f4")
    out = []
    for rear in (False, True):
        m0 = P[:, 0] < -20.0 if rear else P[:, 0] > 30.0
        for side in (1.0, -1.0):
            q = P[m0 & (np.sign(P[:, 1]) == side)]
            if len(q) == 0:
                continue
            score = np.abs(q[:, 1]) + q[:, 2] + (-0.3 if rear else 0.5) * q[:, 0]
            out.append(tuple(float(v) for v in q[int(np.argmax(score))]))
    return out


def _octane_from_obj(obj_path, tex_path):
    """RocketSimVis's own Octane.obj -> the Octane_RL.npz layout (pos, nrm, per-vertex material id for
    the car shader: 0 team paint, 1 trim, 2 tire, 3 metal, 5 lights, 7 matte black), classified from
    the colour of its texture at each vertex's UV. Wheels are baked into this body: no separate wheels."""
    from PIL import Image
    obj = pywavefront.Wavefront(obj_path, collect_faces=True, create_materials=True, parse=True)
    tex = np.asarray(Image.open(tex_path).convert("RGB"), "f4") / 255.0
    th, tw = tex.shape[:2]
    pos, nrm, mat = [], [], []
    for m in obj.materials.values():
        fmt = m.vertex_format                       # e.g. "T2F_N3F_V3F"
        stride = {"T2F_N3F_V3F": 8, "N3F_V3F": 6, "T2F_V3F": 5, "V3F": 3}[fmt]
        v = np.asarray(m.vertices, "f4").reshape(-1, stride)
        p = v[:, -3:]
        n = v[:, -6:-3] if "N3F" in fmt else np.tile([0.0, 0.0, 1.0], (len(v), 1))
        if "T2F" in fmt:
            uv = v[:, :2]
            x = np.clip((uv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
            y = np.clip((1.0 - uv[:, 1] % 1.0) * th, 0, th - 1).astype(int)
            c = tex[y, x]
        else:
            c = np.tile([0.5, 0.6, 0.8], (len(v), 1))
        mx, mn = c.max(1), c.min(1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        k = np.full(len(v), 1.0, "f4")                                  # trim
        k[mx < 0.09] = 2.0                                              # tires
        k[(mx >= 0.09) & (mx < 0.16)] = 7.0                             # matte black
        k[(sat < 0.2) & (mx >= 0.35)] = 3.0                             # metal / rims
        k[(sat >= 0.25) & (c[:, 2] > c[:, 0])] = 0.0                    # blue = team paint
        k[(sat >= 0.25) & (c[:, 0] > c[:, 2]) & (mx > 0.5)] = 5.0       # yellow-white = lights
        pos.append(p); nrm.append(n); mat.append(k)
    out = {"pos": np.concatenate(pos), "nrm": np.concatenate(nrm), "mat": np.concatenate(mat)}
    empty = np.zeros((0, 3), "f4")
    for w in range(4):
        out["w%d_pos" % w], out["w%d_nrm" % w], out["w%d_mat" % w] = empty, empty, np.zeros(0, "f4")
    # RocketSim Octane wheel anchors (only used by the suspension rig)
    out["wheel_centers"] = np.array([[51.2, -25.0, -3.3], [51.2, 24.8, -4.0], [-33.7, -28.1, -1.8], [-33.8, 27.7, -2.6]], "f4")
    out["wheel_radius"] = np.array([12.5, 12.5, 15.0, 15.0], "f4")
    return out


def _read_settings():
    import json as _json
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_settings.json"), "r", encoding="utf-8") as f:
            return _json.load(f)
    except (OSError, ValueError):
        return {}


def _write_settings(d):
    """Merge d into rsv_settings.json (atomic replace)."""
    import json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_settings.json")
    cur = _read_settings()
    cur.update(d)
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            _json.dump(cur, f, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


class RSVRenderer:
    """All rendering + camera + effects/sound logic, independent of Qt: it draws into whatever
    framebuffer it is handed. QRSVGLWidget (bottom of this file) wraps it for the real window;
    tools/headless_test.py drives it with a standalone GL context to render PNGs + benchmark with no
    window at all."""

    def __init__(self):

        self.config = Config()

        self.spectate_count = 0
        self.spectate_idx = 0
        self.prev_interp_ratio = 0
        self.car_cam_time = 0
        self.cam_manual = None   # None = AUTO (dribble-driven car cam); True/False = user forced via Space
        # Optional GUI-controlled target selection. A new closest car must hold that lead for the
        # confirmation interval before the view changes, avoiding distracting one-frame cuts.
        self._vis_auto_switch = False
        self._auto_cam_key = None   # None = follow the GUI's Vis auto switch; True/False = forced with A
        self._vis_auto_next_poll = 0.0
        self._vis_auto_control_mtime = None
        self._vis_auto_candidate = -1
        self._vis_auto_candidate_since = 0.0
        self._car_cam_offset_smooth = None   # exponential-smoothed car-cam offset/dir (None = snap next frame)
        self._car_cam_dir_smooth = None
        self.last_render_time = time.time()
        self.fps_counter = 0
        self.last_fps = 0
        self.prev_state = None # type: GameState

        # Rolling gameplay clip recorder (last 12s of rendered frames -> mp4 on demand).
        self.clip_recorder = ClipRecorder()

        # Cached static boost-pad draw data (positions never move) + hoisted constants.
        self._zero_color = Vector4((0, 0, 0, 0)).astype('f4')
        self._pad_static = None   # list of (is_big, model_matrix_f4)
        self._pad_sig = None
        self._pad_vaos = ["BoostPad_Small_0.obj", "BoostPad_Small_1.obj",
                          "BoostPad_Big_0.obj", "BoostPad_Big_1.obj"]  # indexed 2*is_big + active

        # RSV_PERF timing accumulators
        self._perf_render = 0.0
        self._perf_capture = 0.0
        self._perf_deepcopy = 0.0
        self._perf_pads = 0.0
        self._perf_cars = 0.0
        self._perf_arena = 0.0
        self._perf_ball = 0.0
        self._perf_tail = 0.0
        self._perf_lock = 0.0
        self._perf_lines = 0.0
        self._perf_hud = 0.0
        self._perf_pai = 0.0
        self._perf_ui = 0.0
        self._perf_frames = 0
        self._perf_t0 = time.time()

        ########################################################################

        # MSAA now lives ONLY in the offscreen scene framebuffer (see _ensure_render_target); the
        # window's own framebuffer is single-sample, so the final blit, HUD and clip readback are cheap.
        self.samples = max(0, MSAA)
        self.window_mode = False       # True in the real window: the Graphics settings apply (not in clips/tests)

        self.render_target = None       # framebuffer draws currently go to (scene FBO, then screen for HUD)
        self.screen_fb = None           # the window (or headless) framebuffer handed to paint()
        self.cap_fbo = None
        self.cap_resolve_fbo = None
        self._cap_size = None
        self._quad_prog = None
        self._quad_vao = None

        # Game audio + effects (created in init_gl once a context exists; audio needs no GL).
        self.audio = rl_audio.Audio()
        self.fx = None
        self._boost_last = {}          # car index -> last time it was boosting (sound hold)
        self._spectated_now = -1
        self.gpu_name = "?"
        self._last_boosting = {}
        self._prev_speed_ss = {}

    def load_texture_2d(self, path: str) -> moderngl.Texture:
        return resources.textures.load(TextureDescription(path=path))

    def _build_digit_atlas(self):
        """0-9 rendered once with a heavy italic font (RL's boost number style) into a texture."""
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for fn in ("seguibli.ttf", "seguibl.ttf", "arialbi.ttf", "segoeuib.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", fn), 112)
                break
            except OSError:
                pass
        if font is None:
            font = ImageFont.load_default()
        cell_w, cell_h = 96, 140
        img = Image.new("L", (cell_w * 10, cell_h), 0)
        d = ImageDraw.Draw(img)
        metrics = []
        for k in range(10):
            ch = str(k)
            bb = d.textbbox((0, 0), ch, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            x = k * cell_w + (cell_w - w) // 2 - bb[0]
            y = (cell_h - h) // 2 - bb[1]
            d.text((x, y), ch, fill=255, font=font)
            metrics.append(w / float(cell_w))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        tex = self.ctx.texture(img.size, 1, img.tobytes())
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        return tex, (cell_w, cell_h, metrics)

    def _label_texture(self, text):
        """A single-channel text texture (Bahnschrift SemiBold), cached: (texture, aspect, cap-height fraction)."""
        cache = self.__dict__.setdefault("_label_tex", {})
        if text in cache:
            return cache[text]
        from PIL import Image, ImageDraw, ImageFont
        fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        font = None
        for fn in ("bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(fonts, fn), 96)
                try:
                    font.set_variation_by_name("SemiBold")
                except Exception:
                    pass
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        pad = 8
        bb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=font)
        cap = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), "H", font=font)
        w, h = bb[2] - bb[0] + 2 * pad, cap[3] - cap[1] + 2 * pad + 30
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((pad - bb[0], pad - cap[1]), text, fill=255, font=font)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        t = self.ctx.texture(im.size, 1, im.tobytes())
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        t.build_mipmaps()
        cache[text] = (t, w / float(h), (cap[3] - cap[1]) / float(h), pad / float(h))
        return cache[text]

    def _draw_label(self, text, x, y_top, cap_h, rgba, ortho):
        tex, aspect, cap_frac, pad_frac = self._label_texture(text)
        qh = cap_h / cap_frac
        qw = qh * aspect
        x0, y0 = x - qw * pad_frac / aspect, y_top - qh * pad_frac
        arr = np.asarray([(x0, y0, 0.0, 1.0), (x0 + qw, y0, 1.0, 1.0), (x0 + qw, y0 + qh, 1.0, 0.0),
                          (x0, y0, 0.0, 1.0), (x0 + qw, y0 + qh, 1.0, 0.0), (x0, y0 + qh, 0.0, 0.0)], "f4")
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(ortho.astype("f4"))
        self.prog_text["Tex"].value = 0
        self.prog_text["color"].value = tuple(rgba)
        tex.use(location=0)
        self.render_target.use()
        self.text_vao.render(moderngl.TRIANGLES, vertices=6)
        return qw * (1.0 - 2.0 * pad_frac / aspect)

    @staticmethod
    def _rect_tris(x0, y0, x1, y1, rgba):
        return [(x0, y0, 0.0, *rgba), (x1, y0, 0.0, *rgba), (x1, y1, 0.0, *rgba),
                (x0, y0, 0.0, *rgba), (x1, y1, 0.0, *rgba), (x0, y1, 0.0, *rgba)]

    def render_clip_indicator(self, width, height):
        """Top-left clip status (scales with the window like the boost gauge): a blinking red dot + "Clipping..."
        and a progress bar while the clip renders offline, then "Clip saved!" for 1 s (or "Clip failed")."""
        st = self.clip_recorder.clip_status() if hasattr(self.clip_recorder, "clip_status") else None
        if st is None:
            return
        kind, v = st
        s = height / 1080.0
        m = 26.0 * s
        y = m
        u = get_ui()
        try:
            if u is not None and u.isVisible():                  # stay clear of the stats / settings panel
                y += u.window().devicePixelRatioF() * (u.geometry().bottom() + 1)
        except Exception:
            pass
        cap = 22.0 * s
        pw, ph = 300.0 * s, (78.0 if kind == "rendering" else 58.0) * s
        a = 1.0 if kind == "rendering" else min(1.0, (1.0 if kind == "saved" else 2.5) - v) / 1.0
        a = max(0.0, min(1.0, a * 3.0))
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tris = []
        tris += self._rect_tris(m, y, m + pw, y + ph, (0.03, 0.035, 0.05, 0.72 * a))           # panel
        accent = {"rendering": (0.95, 0.22, 0.22), "saved": (0.30, 0.90, 0.45), "error": (1.0, 0.30, 0.25)}[kind]
        tris += self._rect_tris(m, y, m + 5.0 * s, y + ph, (*accent, 0.95 * a))                # accent edge
        cx, cy, r = m + 26.0 * s, y + 29.0 * s, 8.0 * s
        if kind == "rendering":
            blink = 0.55 + 0.45 * math.sin(time.time() * 6.0)
            dot = (1.0, 0.25, 0.25, blink * a)
            for k in range(20):                                                          # recording dot
                a0, a1 = 2 * math.pi * k / 20, 2 * math.pi * (k + 1) / 20
                tris += [(cx, cy, 0.0, *dot), (cx + r * math.cos(a0), cy + r * math.sin(a0), 0.0, *dot),
                         (cx + r * math.cos(a1), cy + r * math.sin(a1), 0.0, *dot)]
            bx0, bx1, by0, by1 = m + 18.0 * s, m + pw - 18.0 * s, y + ph - 22.0 * s, y + ph - 14.0 * s
            tris += self._rect_tris(bx0, by0, bx1, by1, (1.0, 1.0, 1.0, 0.15 * a))           # bar track
            tris += self._rect_tris(bx0, by0, bx0 + (bx1 - bx0) * max(0.02, v), by1, (*accent, 0.95 * a))
        arr = np.asarray(tris, "f4")
        if len(arr) * 7 * 4 <= self.hud_c_max_verts * 7 * 4:
            self._hud_draw_tris(arr)
        label = {"rendering": "Clipping...", "saved": "Clip saved!", "error": "Clip failed"}[kind]
        tx = m + (44.0 if kind == "rendering" else 20.0) * s
        tcol = (1.0, 1.0, 1.0, a) if kind == "rendering" else (*accent, a)
        self._draw_label(label, tx, cy - cap / 2.0, cap, tcol, ortho)
        if kind == "rendering":
            pct = "%d%%" % int(round(v * 100))
            tex, aspect, cap_frac, pad_frac = self._label_texture(pct)
            wpx = (cap * 0.8) / cap_frac * aspect * (1.0 - 2.0 * pad_frac / aspect)
            self._draw_label(pct, m + pw - 18.0 * s - wpx, cy - cap * 0.4, cap * 0.8, (1.0, 1.0, 1.0, 0.75 * a), ortho)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    MAP_KEYS = {Qt.Key_Up: "valley", Qt.Key_Left: "temple", Qt.Key_Right: "paris", Qt.Key_Down: "space"}

    def _theme_programs(self):
        return (self.prog_rl_arena, self.prog_car, self.prog_ball, self.prog_sky, self.prog_stadium, self.prog_pad,
                self.prog_scene, self.prog_scene_low, self.prog_crowd, self.prog_grass)

    # ---- 3D grass (GRASS_VERT) ---------------------------------------------------------------------------------- #
    GRASS_TILE = 128.0
    # config gfx_grass -> (blades per tile at full density, full-density radius, max distance, blade height)
    GRASS_LEVELS = {1: (1100.0, 300.0, 1500.0, 5.5), 2: (2200.0, 370.0, 2000.0, 6.0), 3: (3600.0, 460.0, 2800.0, 6.0)}
    TURF_BAKE = (2048, 2560)                     # top-down turf colour for the blades: ~4.1 uu per texel

    def _init_grass(self):
        self.prog_grass = self.ctx.program(vertex_shader=rl_shaders.GRASS_VERT, fragment_shader=rl_shaders.GRASS_FRAG)
        T = self.GRASS_TILE
        xs = np.arange(-3840.0, 3840.0, T)
        ys = np.arange(-4864.0, 4864.0, T)
        gx, gy = np.meshgrid(xs, ys)
        self._grass_tiles = np.stack([gx.ravel(), gy.ravel()], 1).astype("f4")      # tile min corners
        self._grass_ncol, self._grass_nrow = len(xs), len(ys)
        self._grass_ctr = (self._grass_tiles + T / 2).astype("f8")
        self._grass_buckets = []
        self._grass_draws = []
        self._grass_key = None
        self.prog_grass["tileSize"].value = T
        self.prog_grass["albedoTex"].value = 8
        self.prog_grass["padMask"].value = 9
        self._turf_tex = None
        self._turf_map = None
        # 1 = grass, 0 = a boost pad's footprint (soft edge): blades must not poke through the pads
        import states as _st
        MW, MH = 525, 650                                   # 16 uu / texel over x +-4200, y +-5200
        gx = (np.arange(MW) + 0.5) / MW * 8400.0 - 4200.0
        gy = (np.arange(MH) + 0.5) / MH * 10400.0 - 5200.0
        X, Y = np.meshgrid(gx, gy)
        mask = np.ones((MH, MW), "f4")
        for loc in _st.default_boost_pad_locations:
            big = float(loc[2]) == 73.0
            rad = 92.0 if big else 50.0
            mask = np.minimum(mask, np.clip((np.hypot(X - float(loc[0]), Y - float(loc[1])) - rad * 0.6) / (rad * 0.5), 0.0, 1.0))
        self._pad_mask = self.ctx.texture((MW, MH), 1, (mask * 255.0).astype("u1").tobytes())
        self._pad_mask.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _bake_turf(self):
        """The floor shader's unlit turf + markings colour, rendered top-down once per map (texture unit 8)."""
        W, H = self.TURF_BAKE
        if self._turf_tex is None:
            self._turf_tex = self.ctx.texture((W, H), 4, dtype="f1")
            self._turf_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._turf_depth = self.ctx.depth_renderbuffer((W, H))
            self._turf_fbo = self.ctx.framebuffer(color_attachments=[self._turf_tex], depth_attachment=self._turf_depth)
        # orthographic, straight down: x -> +-4200, y -> +-5200 (column-major bytes; diagonal, so no transpose)
        ortho = np.diag([1.0 / 4200.0, 1.0 / 5200.0, -1.0 / 3000.0, 1.0]).astype("f4")
        pa = self.prog_rl_arena
        self._turf_fbo.use()
        self._turf_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        pa["m_vp"].write(ortho.tobytes())
        pa["camPos"].value = (0.0, 0.0, 3000.0)
        pa["bakeAlbedo"].value = 1
        pa["detailBias"].value = 1.0
        self._blade_tex.use(location=5)
        self._grain_tex.use(location=7)
        self.vaos['ArenaMeshCustom.obj'].render(moderngl.TRIANGLES)
        pa["bakeAlbedo"].value = 0
        self.ctx.enable(moderngl.BLEND)
        self._turf_map = self.map_name

    def _render_grass(self, vp, camera_pos, tnow, cst, cfw, cf3, cu3, n_casters, ball_mark):
        level = int(getattr(self.config, "gfx_grass", 2))
        if level <= 0 or self.map_name == "space" or "grass" in _SKIP:
            return
        dens, near, far, bh = self.GRASS_LEVELS.get(level, self.GRASS_LEVELS[2])
        if self._turf_map != self.map_name:
            self._bake_turf()
            self.render_target.use()
        cx, cy, cz = float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
        # Tile selection + blade-count buckets depend only on the camera POSITION (the view frustum is culled per tile
        # in GRASS_VERT), so they are rebuilt only when the camera has moved ~40 uu -- most frames reuse them as is.
        key = (round(cx / 150.0), round(cy / 150.0), round(cz / 150.0), level, self.map_name)
        if key != self._grass_key:
            self._grass_key = key
            self._grass_draws = []
            h = self.GRASS_TILE / 2
            T = self.GRASS_TILE
            ncol = self._grass_ncol
            c0 = max(0, int((cx - far + 3840.0) // T)); c1 = min(ncol, int((cx + far + 3840.0) // T) + 1)
            r0 = max(0, int((cy - far + 4864.0) // T)); r1 = min(self._grass_nrow, int((cy + far + 4864.0) // T) + 1)
            if c0 < c1 and r0 < r1:
                win = (np.arange(r0, r1)[:, None] * ncol + np.arange(c0, c1)[None, :]).ravel()
                ctr_w = self._grass_ctr[win]
                dx = np.maximum(np.abs(ctr_w[:, 0] - cx) - h, 0.0)
                dy = np.maximum(np.abs(ctr_w[:, 1] - cy) - h, 0.0)
                dmin = np.sqrt(dx * dx + dy * dy + cz * cz)
                sel = dmin < far + 160.0
                idx, dmin = win[sel], dmin[sel]
                dm_ = np.maximum(dmin - 150.0, 1.0)
                want = dens * np.minimum(1.0, (near * near) / (dm_ * dm_))
                lo, bi = dens, 0
                while lo >= 16.0 and len(idx):
                    bm = (want <= lo) & (want > lo / 1.4142) if lo < dens else (want > lo / 1.4142)
                    if bm.any():
                        t = self._grass_tiles[idx[bm]]
                        buf, vao = self._grass_bucket(bi)
                        buf.orphan(len(t) * 8)
                        buf.write(t.tobytes())
                        self._grass_draws.append((vao, int(math.ceil(lo)) * 3, len(t)))
                        bi += 1
                    lo /= 1.4142                              # finer buckets: at most ~40% extra (culled) blades
        if not self._grass_draws:
            return
        pg = self.prog_grass
        pg["m_vp"].write(vp.tobytes())
        pg["camPos"].value = (cx, cy, cz)
        pg["time"].value = tnow
        pg["density0"].value = dens
        pg["nearD"].value = near
        pg["farD"].value = far
        pg["bladeH"].value = bh
        pg["casters"].write(cst.tobytes())
        pg["casterFwd"].write(cfw.tobytes())
        pg["nCasters"].value = n_casters
        self._turf_tex.use(location=8)
        self._pad_mask.use(location=9)
        self.ctx.disable(moderngl.CULL_FACE)
        for vao, nv, ni in self._grass_draws:
            vao.render(moderngl.TRIANGLES, vertices=nv, instances=ni)
        self.ctx.enable(moderngl.CULL_FACE)

    def _grass_bucket(self, i):
        """Instance buffer + VAO of blade-count bucket i (made once, reused)."""
        while len(self._grass_buckets) <= i:
            buf = self.ctx.buffer(reserve=len(self._grass_tiles) * 8, dynamic=True)
            self._grass_buckets.append((buf, self.ctx.vertex_array(self.prog_grass, [(buf, "2f/i", "i_tile")])))
        return self._grass_buckets[i]

    def set_map(self, name, save=True):
        """Switch the scenery, sky, light and field style (maps.py). Builds the map's mesh the first time (cached
        on disk afterwards) and remembers the choice in rsv_settings.json."""
        th = rl_maps.THEMES[name]
        self.map_name = name
        mid = rl_maps.MAP_ID[name]
        for prog in self._theme_programs():
            for key, val in (("uSunDir", th["sun_dir"]), ("uSunCol", th["sun_col"]), ("uSkyZen", th["zen"]),
                             ("uSkyMid", th["mid"]), ("uSkyHor", th["hor"]), ("uSunGlow", th["glow"]),
                             ("uSkyGround", th["ground"]), ("uAmb", th["amb"]), ("haze", th["haze"])):
                if key in prog:
                    prog[key].value = tuple(val)
        self.prog_rl_arena["mapId"].value = mid
        self.prog_rl_arena["grassCol"].value = tuple(th["grass"])
        self.prog_rl_arena["glassK"].value = float(th.get("glass", 1.0))
        self.prog_scene["uNight"].value = float(th.get("night", 1.0))
        if "uNight" in self.prog_scene_low:
            self.prog_scene_low["uNight"].value = float(th.get("night", 1.0))
        self.prog_sky["mapId"].value = mid
        self.prog_sky["cloudA"].value = tuple(th["cloudA"])
        self.prog_sky["cloudB"].value = tuple(th["cloudB"])
        self.prog_sky["starK"].value = float(th.get("stars", 0.0))
        self.prog_sky["cloudShape"].value = tuple(th.get("cloud_shape", (1.9, 1.9)))
        if name == "space" and self._space_cube is None:
            self._bake_space_sky()
        if name in rl_maps.BUILDERS and name not in self._scenes:
            mesh, crowd = rl_maps.load_or_build(name, DATA_DIR_PATH)
            # shuffled once: the crowd quality setting draws a prefix = an evenly thinned crowd
            crowd = crowd[np.random.default_rng(5).permutation(len(crowd))] if len(crowd) else crowd
            cvao = None
            if len(crowd):
                cvao = self.ctx.vertex_array(self.prog_crowd, [
                    (self.egg_vbo, "2f", "in_corner"),
                    (self.ctx.buffer(crowd.tobytes()), "3f 3f 2f/i", "i_pos", "i_col", "i_ps")])
            self._scenes[name] = {"mesh": mesh, "cvao": cvao, "n_eggs": len(crowd), "lod": {}}
        if save:
            _write_settings({"map": name})
        print("[map] {}".format(rl_maps.TITLE[name]), flush=True)

    def _cheer_ages(self):
        """Seconds since the last goal / save each crowd (blue fans, orange fans) cheers for -> CROWD_VERT."""
        now = time.time()
        return (min(now - self._cheer_t[0], 1e4), min(now - self._cheer_t[1], 1e4))

    MAP_DETAIL_MIN_AREA = (10000.0, 2000.0, 0.0)     # Map detail Low / Medium / High: drop triangles smaller than this
    CROWD_FRAC = (0.3, 0.65, 1.0)                    # Crowd Low / Medium / High

    @staticmethod
    def _decimate(mesh, stride_pos=3, min_area=0.0):
        """Drop the scenery's tiniest triangles (small details: < ~1% of its area) -> fewer polygons."""
        if min_area <= 0.0:
            return mesh
        P = mesh[:, :stride_pos].reshape(-1, 3, 3)
        A = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        keep = np.repeat(A >= min_area, 3)
        return np.ascontiguousarray(mesh[keep])

    def _render_scenery(self, vp_bytes, cam_bytes, tnow):
        q = max(0, min(2, int(getattr(self.config, "q_map", 2))))
        if self.map_name == "valley":                      # landscape.py's valley + the extras (maps.build_valley)
            lods = self.__dict__.setdefault("_stadium_lods", {})
            if q not in lods:
                if q == 2:
                    lods[q] = (self.stadium_vao, self.stadium_n)
                else:
                    m = self._decimate(self._stadium_mesh, 3, self.MAP_DETAIL_MIN_AREA[q])
                    lods[q] = (self.ctx.vertex_array(self.prog_stadium, [(self.ctx.buffer(m.tobytes()), "3f 2f",
                                                                          "in_position", "in_uv")]), len(m))
            svao, sn = lods[q]
            self.prog_stadium["m_vp"].write(vp_bytes)
            self.prog_stadium["camPos"].write(cam_bytes)
            if "stadium" not in _SKIP:
                svao.render(moderngl.TRIANGLES, vertices=sn)
        if self.map_name not in self._scenes:
            return
        sc = self._scenes[self.map_name]
        ps = self.prog_scene_low if q == 0 else self.prog_scene
        if q not in sc["lod"]:
            m = self._decimate(sc["mesh"], 3, self.MAP_DETAIL_MIN_AREA[q])
            sc["lod"][q] = (self.ctx.vertex_array(ps, [(self.ctx.buffer(m.tobytes()), "3f 3f 2f",
                                                       "in_position", "in_col", "in_ek")]), len(m))
        vao, n = sc["lod"][q]
        cvao = sc["cvao"]
        n_eggs = int(sc["n_eggs"] * self.CROWD_FRAC[max(0, min(2, int(getattr(self.config, "q_crowd", 2))))])
        ps["m_vp"].write(vp_bytes)
        ps["camPos"].write(cam_bytes)
        ps["time"].value = tnow
        if "stadium" not in _SKIP:
            vao.render(moderngl.TRIANGLES, vertices=n)
        if cvao is not None and n_eggs > 0 and "crowd" not in _SKIP:
            pc = self.prog_crowd
            pc["m_vp"].write(vp_bytes)
            pc["camPos"].write(cam_bytes)
            pc["time"].value = tnow
            pc["cheerAge"].value = self._cheer_ages()
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable_direct(0x809E)                 # GL_SAMPLE_ALPHA_TO_COVERAGE: smooth egg edges
            cvao.render(moderngl.TRIANGLES, instances=n_eggs)
            self.ctx.disable_direct(0x809E)

    def _bake_blades(self):
        """The grass blade pattern -> a repeating, mipmapped R8 texture (texture unit 5) read by the arena shader."""
        prog = self.ctx.program(vertex_shader=rl_shaders.SKY_VERT, fragment_shader=rl_shaders.BLADE_FRAG)
        tex = self.ctx.texture((2048, 2048), 1, dtype="f1")
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self.ctx.viewport = (0, 0, 2048, 2048)
        self.ctx.vertex_array(prog, []).render(moderngl.TRIANGLES, vertices=3)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = True
        try:
            tex.anisotropy = 8.0
        except Exception:
            pass
        self._blade_tex = tex
        self.prog_rl_arena["bladeTex"].value = 5
        # the turf grain, the same way (texture unit 7)
        prog = self.ctx.program(vertex_shader=rl_shaders.SKY_VERT, fragment_shader=rl_shaders.GRAIN_FRAG)
        tex = self.ctx.texture((2048, 2048), 1, dtype="f1")
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self.ctx.vertex_array(prog, []).render(moderngl.TRIANGLES, vertices=3)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = True
        try:
            tex.anisotropy = 8.0
        except Exception:
            pass
        self._grain_tex = tex
        self.prog_rl_arena["grainTex"].value = 7

    SPACE_CUBE_N = 1024

    def _bake_space_sky(self):
        """The static part of the Orbit sky (spaceStatic) -> a cube map, once: the sky pass then only adds the stars
        and the sun (it was ~1.4 ms per frame on the iGPU)."""
        N = self.SPACE_CUBE_N
        tex2d = self.ctx.texture((N, N), 4, dtype="f1")
        fbo = self.ctx.framebuffer(color_attachments=[tex2d])
        cube = self.ctx.texture_cube((N, N), 4, dtype="f1")
        ps = self.prog_sky
        ps["bakeN"].value = float(N)
        ps["time"].value = 0.0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        for face in range(6):
            fbo.use()
            self.ctx.viewport = (0, 0, N, N)
            ps["skyBake"].value = face
            self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
            cube.write(face, fbo.read(components=4, alignment=1))
        ps["skyBake"].value = -1
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        cube.filter = (moderngl.LINEAR, moderngl.LINEAR)
        try:
            self.ctx.enable_direct(0x884F)                  # GL_TEXTURE_CUBE_MAP_SEAMLESS
        except Exception:
            pass
        fbo.release(); tex2d.release()
        self._space_cube = cube

    SKY_CUBE_N = 1024
    SKY_STRIPS = 4                                  # strips per face

    def _update_sky_cube(self, tnow):
        """Map detail Low / Medium: the sky (clouds: 11 noise lookups a pixel, ~0.8 ms full screen on the iGPU) is baked
        into a cube map and the sky pass just samples it. Double-buffered: the back cube is baked a strip every 2nd frame,
        every strip with the SAME time, and only swapped in once complete -- strips baked at different times did not
        line up (the clouds drift), which made the sky patchy. High keeps the live per-pixel sky."""
        use = int(getattr(self.config, "q_map", 2)) < 2 and self.map_name != "space"
        ps = self.prog_sky
        if not use:
            ps["skyCubeOn"].value = 0
            return
        from OpenGL import GL as _gl
        N = self.SKY_CUBE_N
        S = self.SKY_STRIPS
        if self._sky_cube is None:
            self._sky_cubes = []
            for _ in range(2):
                cb = self.ctx.texture_cube((N, N), 4, dtype="f1")
                cb.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._sky_cubes.append(cb)
            self._sky_cube = self._sky_cubes[0]
            self._sky_tex2d = self.ctx.texture((N, N), 4, dtype="f1")
            self._sky_fbo = self.ctx.framebuffer(color_attachments=[self._sky_tex2d])
            try:
                self.ctx.enable_direct(0x884F)              # GL_TEXTURE_CUBE_MAP_SEAMLESS: no seams at the face edges
            except Exception:
                pass
        jobs = None
        if getattr(self, "_sky_cube_map", None) != self.map_name:     # new map: bake everything now, both cubes
            self._sky_cube_map = self.map_name
            self._sky_strip, self._sky_bake_t = 0, tnow
            jobs = [(cb, f, k) for cb in self._sky_cubes for f in range(6) for k in range(S)]
        else:
            self._sky_tick = getattr(self, "_sky_tick", 0) + 1
            if self._sky_tick % 2 == 0:
                if self._sky_strip == 0:
                    self._sky_bake_t = tnow
                back = self._sky_cubes[1] if self._sky_cube is self._sky_cubes[0] else self._sky_cubes[0]
                f, k = divmod(self._sky_strip, S)
                jobs = [(back, f, k)]
                self._sky_strip += 1
        if jobs:
            h = N // S
            vp0 = self.ctx.viewport
            ps["bakeN"].value = float(N)
            ps["time"].value = self._sky_bake_t
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.disable(moderngl.CULL_FACE)
            self._sky_fbo.use()
            _gl.glActiveTexture(_gl.GL_TEXTURE15)
            for cb, face, k in jobs:
                y0 = k * h
                self.ctx.viewport = (0, y0, N, h)
                ps["skyBake"].value = 10 + face
                self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
                _gl.glBindTexture(_gl.GL_TEXTURE_CUBE_MAP, cb.glo)
                _gl.glCopyTexSubImage2D(_gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, 0, y0, 0, y0, N, h)
            _gl.glActiveTexture(_gl.GL_TEXTURE0)
            self.ctx.viewport = vp0
            ps["skyBake"].value = -1
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            if self._sky_strip >= 6 * S:                    # back cube complete: show it
                self._sky_cube = jobs[-1][0]
                self._sky_strip = 0
        self._sky_cube.use(location=14)
        ps["skyCubeOn"].value = 1

    def _render_pad_ghosts(self, vp_bytes, cam_bytes):
        """The returning orbs of recharging pads. Translucent, so drawn AFTER the sky: drawn with the pads, the sky
        (which only fills pixels nothing wrote depth to) painted over them wherever the sky was behind -- they were
        only visible from above, against the pad."""
        ghosts = getattr(self, "_pad_ghosts", None)
        if not ghosts:
            return
        pp = self.prog_pad
        pp["m_vp"].write(vp_bytes)
        pp["camPos"].write(cam_bytes)
        self.t_padgen.use(location=0)
        pp["Texture"].value = 0
        pp["flash"].value = 0.0
        self.render_target.use()
        self.ctx.fbo.depth_mask = False
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for mat, is_big, g in ghosts:
            pp["m_model"].write(mat)
            pp["orbZ"].value = 16.0 if is_big else -10.0
            pp["orbCz"].value = 29.65 if is_big else -1000.0      # big orb centre (object space)
            pp["ghost"].value = max(0.02, g)
            self.pad_vaos_rl[self._pad_vaos[2 * is_big + 1]].render(moderngl.TRIANGLES)
        pp["ghost"].value = 0.0
        self.ctx.fbo.depth_mask = True
        self._pad_ghosts = []

    BALL_SPIN_SHUTTER = 0.7          # fraction of the last frame's rotation the ball's spin blur covers

    def _ball_spin_blur(self, f, u, teleported):
        """(object-space axis, angle) the ball turned since the previous frame, x shutter, for the spin motion blur
        in BALL_FRAG. The object frame is the model matrix's columns (forward, left, up)."""
        # plain floats: numpy's per-call overhead on 3x3 matrices cost ~0.15 ms a frame
        f = (float(f[0]), float(f[1]), float(f[2])); u = (float(u[0]), float(u[1]), float(u[2]))
        l_ = (u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0])
        M = (f, l_, u)                                   # columns
        prev, self._ball_prev_rot = getattr(self, "_ball_prev_rot", None), M
        if prev is None or teleported or not isinstance(prev, tuple):
            return (0.0, 0.0, 1.0, 0.0)
        # R = M @ prev^T = sum_k col_k(M) col_k(prev)^T
        def Rij(i, j):
            return M[0][i] * prev[0][j] + M[1][i] * prev[1][j] + M[2][i] * prev[2][j]
        tr = Rij(0, 0) + Rij(1, 1) + Rij(2, 2)
        ang = math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5)))
        if ang < 1e-4 or ang > 1.2:                  # still, or a jump (reset) rather than a spin
            return (0.0, 0.0, 1.0, 0.0)
        ax = (Rij(2, 1) - Rij(1, 2), Rij(0, 2) - Rij(2, 0), Rij(1, 0) - Rij(0, 1))
        n_ = max(math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]), 1e-9)
        ax = (ax[0] / n_, ax[1] / n_, ax[2] / n_)
        ao = [M[c][0] * ax[0] + M[c][1] * ax[1] + M[c][2] * ax[2] for c in range(3)]   # M^T @ ax
        return (ao[0], ao[1], ao[2], ang * self.BALL_SPIN_SHUTTER)

    BALL_MARK_TOP = 1000.0           # ball height (above the surface under it) where the inner marker ring is 4 dots

    def _ball_mark(self, state, ball_pos):
        """ballMark uniform for the arena shader: (x, y, z, height factor), or w < 0 while the ball is hidden."""
        if getattr(state, "ball_hidden", False) or not int(getattr(self.config, "gfx_ball_marker", 1)):
            return (0.0, 0.0, 0.0, -1.0)
        x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        # the arena surface straight below the ball (the floor, or the floor-wall curve near a wall): bisect the
        # arena distance field between the ball centre (inside) and below the floor (outside)
        lo, hi = -60.0, z
        if rl_events.arena_distance((x, y, hi)) > 0.0 and rl_events.arena_distance((x, y, lo)) <= 0.0:
            for _ in range(16):
                m = 0.5 * (lo + hi)
                if rl_events.arena_distance((x, y, m)) > 0.0:
                    hi = m
                else:
                    lo = m
            ground = hi
        else:
            ground = 0.0
        h = max(0.0, z - 91.25 - ground)
        return (x, y, z, min(1.0, h / self.BALL_MARK_TOP))

    PAD_RESPAWN_BIG = 10.0           # RL boost pad respawn times (s)
    PAD_RESPAWN_SMALL = 4.0

    GOAL_BANNER_S = 3.0          # on screen after a goal (longer while a goal celebration hides the ball)

    def _banner_textures(self, team):
        """'BLUE SCORED!' / 'ORANGE SCORED!' rendered once: a crisp text mask + a blurred glow mask. The PIL part is
        prepared on a background thread at startup (_prewarm_banners): built on the first goal, its blur stalled that
        frame for ~80 ms."""
        tex = self._banner_tex.get(team)
        if tex is not None:
            return tex
        imgs = self._banner_img.get(team) or self._banner_images(team)
        out = []
        for data, size in imgs[:2]:
            t = self.ctx.texture(size, 1, data)
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.build_mipmaps()
            out.append(t)
        tex = self._banner_tex[team] = (out[0], out[1], imgs[2], imgs[3])
        return tex

    def _prewarm_banners(self):
        def work():
            for team in (0, 1):
                try:
                    self._banner_img[team] = self._banner_images(team)
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _banner_images(self, team):
        """-> ((core bytes, size), (glow bytes, size), aspect, glyph fraction), GL-free (runs on any thread)."""
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        text = ("BLUE" if team == 0 else "ORANGE") + " SCORED!"
        fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        font = None
        for fn in ("bahnschrift.ttf", "segoeuil.ttf", "segoeui.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(os.path.join(fonts, fn), 150)
                try:
                    font.set_variation_by_name("SemiLight")      # thin strokes, like RL's goal text
                except Exception:
                    pass
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        pad = 60
        bb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=font)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        core = Image.new("L", (w, h), 0)
        ImageDraw.Draw(core).text((pad - bb[0], pad - bb[1]), text, fill=255, font=font)
        glow = core.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(14))
        out = []
        for im in (core, glow):
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            out.append((im.tobytes(), im.size))
        return out[0], out[1], w / float(h), (bb[3] - bb[1]) / float(h)

    def render_goal_banner(self, width, height, state):
        """RL's goal text: golden 'BLUE SCORED!' / 'ORANGE SCORED!' with a warm glow, pops in, holds, fades."""
        if self._goal_banner is None:
            return
        team, t0 = self._goal_banner
        age = time.time() - t0
        cel = getattr(state, "celebration_pos", None) is not None or getattr(state, "ball_hidden", False)
        end = self.GOAL_BANNER_S if not cel else max(self.GOAL_BANNER_S, age + 0.35)
        if age > end:
            self._goal_banner = None
            return
        a = min(1.0, age / 0.18) * min(1.0, max(0.0, (end - age) / 0.35))
        scale = 1.0 + 0.18 * (1.0 - min(1.0, age / 0.22)) ** 2       # quick settle from 118%
        core, glow, aspect, glyph_frac = self._banner_textures(team)
        text_h = height * 0.075 * scale                               # cap height ~7.5% of the screen
        qh = text_h / glyph_frac
        qw = qh * aspect
        cx, cy = width * 0.5, height * 0.30
        x0, x1, y0, y1 = cx - qw / 2, cx + qw / 2, cy - qh / 2, cy + qh / 2
        arr = np.asarray([(x0, y0, 0.0, 1.0), (x1, y0, 1.0, 1.0), (x1, y1, 1.0, 0.0),
                          (x0, y0, 0.0, 1.0), (x1, y1, 1.0, 0.0), (x0, y1, 0.0, 0.0)], "f4")
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(ortho.astype("f4"))
        self.prog_text["Tex"].value = 0
        self.render_target.use()
        self.ctx.disable(moderngl.DEPTH_TEST)       # a 2D overlay: the window's depth buffer is never cleared
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        for tex, rgba in ((glow, (1.0, 0.55, 0.12, 0.55 * a)), (core, (1.0, 0.80, 0.36, a))):
            tex.use(location=0)
            self.prog_text["color"].value = rgba
            self.text_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _draw_number(self, text, cx, cy, height, rgba):
        cell_w, cell_h, widths = self._digit_metrics
        scale = height / cell_h
        adv = [max(0.45, widths[int(c)]) * cell_w * scale * 0.92 for c in text]
        x = cx - sum(adv) / 2.0
        verts = []
        for c, a in zip(text, adv):
            k = int(c)
            u0, u1 = k / 10.0, (k + 1) / 10.0
            w = cell_w * scale
            x0 = x + a / 2.0 - w / 2.0
            y0, y1 = cy - height / 2.0, cy + height / 2.0
            verts += [(x0, y0, u0, 1.0), (x0 + w, y0, u1, 1.0), (x0 + w, y1, u1, 0.0),
                      (x0, y0, u0, 1.0), (x0 + w, y1, u1, 0.0), (x0, y1, u0, 0.0)]
            x += a
        arr = np.asarray(verts, "f4")
        self.text_vbo.write(arr.tobytes())
        self.prog_text["m_vp"].write(self._hud_ortho.astype("f4"))
        self.prog_text["color"].value = tuple(rgba)
        self._digit_tex.use(location=0)
        self.prog_text["Tex"].value = 0
        self.render_target.use()
        self.text_vao.render(moderngl.TRIANGLES, vertices=len(arr))

    def _load_pad_texture(self, path):
        """Boost-pad palette with the flat yellow disc recoloured to RL's warm orange-gold."""
        from PIL import Image
        im = np.asarray(Image.open(path).convert("RGBA")).astype("f4") / 255.0
        rgb = im[..., :3]
        sat = rgb.max(-1) - rgb.min(-1)
        m = (sat > 0.25)[..., None]
        gold = np.array([1.0, 0.62, 0.12], "f4") * (0.55 + 0.45 * rgb.max(-1, keepdims=True))
        im[..., :3] = np.where(m, gold, rgb)
        im = np.ascontiguousarray(im[::-1])                  # GL origin is bottom-left
        tex = self.ctx.texture((im.shape[1], im.shape[0]), 4, (im * 255).astype("u1").tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def init_gl(self, ctx):

        self.ctx = ctx
        moderngl_window.activate_context(None, self.ctx)

        # WHICH PHYSICAL GPU is actually drawing. On a hybrid laptop (iGPU + dGPU) an OpenGL app can
        # silently land on the integrated GPU, which looks exactly like "the renderer is slow" -- this
        # prints the ground truth instead of guessing.
        try:
            self.gpu_name = str(self.ctx.info.get("GL_RENDERER", "?")).split("/")[0]
            print("[gpu] GL_RENDERER = {}".format(self.ctx.info.get("GL_RENDERER")), flush=True)
            print("[gpu] GL_VENDOR   = {}".format(self.ctx.info.get("GL_VENDOR")), flush=True)
        except Exception as _e:
            print("[gpu] could not read GL info: {!r}".format(_e), flush=True)

        ##########################################

        print("Creating shader programs...")

        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        self.prog_arena = self.ctx.program(
            vertex_shader=ARENA_VERT_SHADER,
            fragment_shader=ARENA_FRAG_SHADER,
            geometry_shader=ARENA_GEOM_SHADER
        )

        print("Creating outline renderer...")
        #self.outline_renderer = OutlineRenderer(self.ctx, (self.width(), self.height())) # TODO: Fix resizing bugs
        self.outline_renderer = None # Disabled due to weird shader compilation issues

        print("Linking shader varaibles...")
        self.pr_m_vp = self.prog['m_vp']
        self.pr_m_model = self.prog['m_model']
        self.pr_global_color = self.prog['globalColor']
        self.pr_camera_pos = self.prog['cameraPos']

        self.pra_m_vp = self.prog_arena['m_vp']
        self.pra_m_model = self.prog_arena['m_model']
        self.pra_ball_pos = self.prog_arena['ballPos']

        ##########################################

        self.ball_ribbon = RibbonEmitter()
        self.ball_trail = RibbonEmitter()      # RL ball trail (last-touch team colour, fast ball only)
        self._flip_until = {}                  # car -> time its flip streaks stop being emitted
        self._corner_ribs = {}                 # car -> 4 RibbonEmitters (upper corners of the car)
        self._ball_trail_on = False
        self._goal_banner = None               # (scoring team, time) -> "BLUE SCORED!" / "ORANGE SCORED!"
        self._banner_tex = {}
        self._banner_img = {}
        self._prewarm_banners()
        self.car_ribbons = []

        print("Data path:", DATA_DIR_PATH)
        print("Loading models and textures...")

        self.vaos = {}
        # RL-style programs (rl_shaders.py). The arena no longer goes through the geometry-shader
        # wireframe program -- that pass is what drew the blue/red triangle-edge lines.
        self._prog_arena_opq = self.ctx.program(vertex_shader=rl_shaders.ARENA_VERT, fragment_shader=rl_shaders.ARENA_FRAG_OPAQUE)
        self._prog_arena_glass = self.ctx.program(vertex_shader=rl_shaders.ARENA_VERT, fragment_shader=rl_shaders.ARENA_FRAG_GLASS)
        self.prog_rl_arena = _ProgPair(self._prog_arena_opq, self._prog_arena_glass)   # uniforms go to both
        self.prog_car = self.ctx.program(vertex_shader=rl_shaders.CAR_VERT, fragment_shader=rl_shaders.CAR_FRAG)
        self.prog_ball = self.ctx.program(vertex_shader=rl_shaders.BALL_VERT, fragment_shader=rl_shaders.BALL_FRAG)
        self.prog_sky = self.ctx.program(vertex_shader=rl_shaders.SKY_VERT, fragment_shader=rl_shaders.SKY_FRAG)
        self.sky_vao = self.ctx.vertex_array(self.prog_sky, [])
        self.prog_sky["skyBake"].value = -1
        self.prog_sky["spaceCube"].value = 6
        self._space_cube = None
        self.prog_sky["skyCube"].value = 14
        self._sky_cube = None                     # Low / Medium: the sky baked into a cube map, a strip a frame
        self._sky_strip = 0
        self._bake_blades()
        # low-poly valley around the arena (landscape.py); the built mesh is cached next to the data
        self.prog_stadium = self.ctx.program(vertex_shader=rl_shaders.LANDSCAPE_VERT, fragment_shader=rl_shaders.LANDSCAPE_FRAG)
        st_mesh = landscape.load_or_build(os.path.join(DATA_DIR_PATH, "landscape_cache.npy"))
        self.stadium_vao = self.ctx.vertex_array(
            self.prog_stadium, [(self.ctx.buffer(st_mesh.tobytes()), "3f 2f", "in_position", "in_uv")])
        self.stadium_n = len(st_mesh)
        self._stadium_mesh = st_mesh
        self.load_vao("ArenaMeshCustom.obj", self._prog_arena_opq)
        self._split_arena()
        for prog in (self.prog_rl_arena,):
            prog["blueCol"].value = (0.10, 0.40, 1.00)
            prog["orangeCol"].value = (1.00, 0.42, 0.06)
        self._init_grass()

        # Car body: an optional detailed mesh data/Octane_RL.npz (pos, normal, material id, 4 separate
        # wheels) -- not included here -- else RocketSimVis's low-poly Octane.obj converted to the same
        # format (wheels are part of its body).
        if os.path.exists(DATA_DIR_PATH + "Octane_RL.npz"):
            oct_npz = np.load(DATA_DIR_PATH + "Octane_RL.npz")
        else:
            oct_npz = _octane_from_obj(DATA_DIR_PATH + "Octane.obj", DATA_DIR_PATH + "T_Octane_B.png")
        oct_inter = np.concatenate([oct_npz["pos"], oct_npz["nrm"], oct_npz["mat"][:, None]], 1).astype("f4")
        self.car_vbo = self.ctx.buffer(oct_inter.tobytes())
        self.car_vao = self.ctx.vertex_array(self.prog_car, [(self.car_vbo, "3f 3f 1f", "in_position", "in_normal", "in_mat")])
        self.car_vert_count = len(oct_inter)
        self.wheel_vaos = []
        self._wheel_vbos = []
        for k in range(4):
            if len(oct_npz["w%d_pos" % k]) == 0:
                continue                             # wheels baked into the body (Octane.obj fallback)
            wi = np.concatenate([oct_npz["w%d_pos" % k], oct_npz["w%d_nrm" % k], oct_npz["w%d_mat" % k][:, None]], 1).astype("f4")
            vbo = self.ctx.buffer(wi.tobytes())
            self.wheel_vaos.append(self.ctx.vertex_array(self.prog_car, [(vbo, "3f 3f 1f", "in_position", "in_normal", "in_mat")]))
            self._wheel_vbos.append(vbo)
        self.wheel_rig = carrig.WheelRig(oct_npz["wheel_centers"], oct_npz["wheel_radius"])
        self.car_streak_points = _streak_points(oct_npz["pos"])   # flip streaks come off these model points

        # Ball: smooth icosphere (radius = RocketSim soccar ball) + procedural panel shader.
        sv, sf_, panel_dirs = _icosphere(4)
        self.ball_vbo = self.ctx.buffer((sv * 91.25).astype("f4").tobytes())
        self.ball_ibo = self.ctx.buffer(sf_.astype("i4").tobytes())
        self.ball_vao = self.ctx.vertex_array(self.prog_ball, [(self.ball_vbo, "3f", "in_position")], self.ball_ibo)

        # Shadow atlas: each caster rendered from the sun into its own tile (see rl_shaders.shadowAt)
        self.prog_shadow = self.ctx.program(vertex_shader=rl_shaders.SHADOW_CASTER_VERT,
                                            fragment_shader=rl_shaders.SHADOW_CASTER_FRAG)
        self._sh_car_vao = self.ctx.vertex_array(self.prog_shadow, [(self.car_vbo, "3f 16x", "in_position")])
        self._sh_wheel_vaos = [self.ctx.vertex_array(self.prog_shadow, [(v, "3f 16x", "in_position")])
                               for v in self._wheel_vbos]
        self._sh_ball_vao = self.ctx.vertex_array(self.prog_shadow, [(self.ball_vbo, "3f", "in_position")], self.ball_ibo)
        self._sh_tex = self.ctx.texture((3 * self.SH_TILE, 3 * self.SH_TILE), 2, dtype="f2")
        self._sh_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._sh_tex.repeat_x = False
        self._sh_tex.repeat_y = False
        self._sh_fbo = self.ctx.framebuffer(color_attachments=[self._sh_tex],
                                            depth_attachment=self.ctx.depth_renderbuffer((3 * self.SH_TILE, 3 * self.SH_TILE)))

        self.fx = rl_fx.FX(self.ctx)

        # Boost pads: own program (glowing gold top, respawn ghost, respawn flash)
        self.prog_pad = self.ctx.program(vertex_shader=rl_shaders.PAD_VERT, fragment_shader=rl_shaders.PAD_FRAG)
        self.pad_vaos_rl = {}
        # generated smooth, high-poly pads (padmesh.py) in place of the low-poly OBJs, with their own 2-texel
        # metal / gold texture
        for name, data in padmesh.all_meshes().items():
            self.pad_vaos_rl[name] = self.ctx.vertex_array(self.prog_pad, [(self.ctx.buffer(data.tobytes()), "3f 3f 2f",
                                                                            "in_position", "in_normal", "in_texcoord_0")])
        self._pad_inst_buf, self._pad_inst_vao = {}, {}
        for name, data in padmesh.all_meshes().items():
            buf = self.ctx.buffer(reserve=64 * 6 * 4, dynamic=True)
            self._pad_inst_buf[name] = buf
            self._pad_inst_vao[name] = self.ctx.vertex_array(self.prog_pad, [
                (self.ctx.buffer(data.tobytes()), "3f 3f 2f", "in_position", "in_normal", "in_texcoord_0"),
                (buf, "4f 2f/i", "i_a", "i_b")])
        self.t_padgen = self.ctx.texture((2, 1), 4, padmesh.PAD_TEX.tobytes())
        self.t_padgen.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._pad_prev = None
        self._pad_pick_t = []
        self._pad_spawn_t = []

        # Maps (maps.py): scenery + crowd per map, switched live with the arrow keys; the last one is remembered
        self.prog_scene = self.ctx.program(vertex_shader=rl_shaders.SCENE_VERT, fragment_shader=rl_shaders.SCENE_FRAG)
        self.prog_scene_low = self.ctx.program(vertex_shader=rl_shaders.SCENE_VERT, fragment_shader=rl_shaders.SCENE_FRAG_LOW)
        self.prog_crowd = self.ctx.program(vertex_shader=rl_shaders.CROWD_VERT, fragment_shader=rl_shaders.CROWD_FRAG)
        self.egg_vbo = self.ctx.buffer(np.array([(-1, 0), (1, 0), (1, 1), (-1, 0), (1, 1), (-1, 1)], "f4").tobytes())
        self._scenes = {}
        self._cheer_t = [-1e9, -1e9]            # last goal / save each crowd cheers for: blue fans, orange fans
        saved_map = _read_settings().get("map", "valley")
        self.set_map(saved_map if saved_map in rl_maps.ORDER else "valley", save=False)

        # Boost meter: analytic gauge + font atlas digits
        self.prog_gauge = self.ctx.program(vertex_shader=rl_shaders.GAUGE_VERT, fragment_shader=rl_shaders.GAUGE_FRAG)
        self.gauge_vbo = self.ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.gauge_vao = self.ctx.vertex_array(self.prog_gauge, [(self.gauge_vbo, "2f 2f", "in_pos", "in_q")])
        self.prog_text = self.ctx.program(vertex_shader=rl_shaders.TEXT_VERT, fragment_shader=rl_shaders.TEXT_FRAG)
        self.text_vbo = self.ctx.buffer(reserve=6 * 4 * 4 * 16, dynamic=True)
        self.text_vao = self.ctx.vertex_array(self.prog_text, [(self.text_vbo, "2f 2f", "in_pos", "in_uv")])
        self._digit_tex, self._digit_metrics = self._build_digit_atlas()
        self.show_phys_hud = bool(_read_settings().get("hud_visible", True))   # H (with the top-left panel, remembered across runs); drawn by phys_hud.PhysHud
        self._phys_hud = None
        self._gpu_query = self.ctx.query(time=True) if PERF and os.environ.get("RSV_PERF") != "2" else None   # 2 = no GPU timer (it syncs CPU+GPU)

        # (Octane.obj / Ball.obj are no longer drawn -- replaced by Octane_RL.npz + the procedural ball --
        # so they aren't parsed at startup any more; pywavefront is slow.)

        self.load_vao("BoostPad_Small_0.obj")
        self.load_vao("BoostPad_Small_1.obj")
        self.load_vao("BoostPad_Big_0.obj")
        self.load_vao("BoostPad_Big_1.obj")

        self.ts_octane = [
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_B.png"),
            self.load_texture_2d(DATA_DIR_PATH + "T_Octane_O.png")
        ]
        self.t_ball = self.load_texture_2d(DATA_DIR_PATH + "T_Ball.png")
        self.t_boostpad = self._load_pad_texture(DATA_DIR_PATH + "T_BoostPad.png")
        self.t_boost_glow = self.load_texture_2d(DATA_DIR_PATH + "T_Boost_Glow.png")
        self.t_black = self.load_texture_2d(DATA_DIR_PATH + "T_Black.png")
        self.t_none = self.load_texture_2d(DATA_DIR_PATH + "T_None.png")

        ############################################

        # Make ribbon mesh
        self.ribbon_max_verts = 1000
        self.ribbon_verts = np.random.randn(self.ribbon_max_verts * 3) * 100
        self.ribbon_vbo = self.ctx.buffer(self.ribbon_verts.astype('f4'))
        self.ribbon_vao = self.ctx.simple_vertex_array(self.prog, self.ribbon_vbo, "in_position")
        self.vaos['ribbon'] = self.ribbon_vao

        # Make debug lines mesh
        self.lines_max_verts = RenderState.MAX_LINES * 2
        self.lines_verts = np.random.randn(self.lines_max_verts * 3) * 100
        self.lines_vbo = self.ctx.buffer(self.lines_verts.astype('f4'))
        self.lines_vao = self.ctx.simple_vertex_array(self.prog, self.lines_vbo, "in_position")
        self.vaos['render_lines'] = self.lines_vao

        # Make 2D HUD mesh (screen-space overlay, e.g. the boost gauge)
        self.hud_max_verts = 65536   # was 4096: the boost gauge now batches ~15.6k verts per frame
        self.hud_vbo = self.ctx.buffer(reserve=self.hud_max_verts * 3 * 4)  # 3 floats/vert
        self.hud_vao = self.ctx.simple_vertex_array(self.prog, self.hud_vbo, "in_position")
        self.vaos['hud'] = self.hud_vao

        ############################################

        # Fullscreen-triangle blit shader: draws the capped-resolution resolve texture (see
        # _ensure_render_target) scaled up to fill the real window. No vertex buffer needed -- the
        # 3 vertices come from gl_VertexID, the standard attributeless "big triangle" trick.
        self._quad_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                out vec2 v_uv;
                void main() {
                    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                    v_uv = pos;
                    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Tex;
                in vec2 v_uv;
                out vec4 f_color;
                // NO y-flip: cap_resolve_fbo was rendered with the SAME bottom-left-origin convention
                // ctx.screen uses (it's another GL framebuffer, not a loaded top-down image file), so
                // sampling it straight reproduces the frame right-side-up. (v_uv.y flipped here once --
                // that's what turned the picture upside down.)
                void main() { f_color = texture(Tex, v_uv); }
            ''',
        )
        self._quad_vao = self.ctx.vertex_array(self._quad_prog, [])

        # Per-vertex-COLOUR HUD program. The shared `prog` carries colour in a globalColor UNIFORM, so
        # every differently-coloured band needs its own draw (63 of them for the boost gauge, ~40 us
        # each in Python/moderngl overhead). Carrying colour per VERTEX instead lets the whole gauge --
        # every band, every alpha -- go out as ONE stitched triangle strip.
        self._hud_c_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 m_vp;
                in vec3 in_position;
                in vec4 in_color;
                out vec4 v_color;
                void main() { v_color = in_color; gl_Position = m_vp * vec4(in_position, 1.0); }
            ''',
            fragment_shader='''
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() { f_color = v_color; }
            ''',
        )
        self.hud_c_max_verts = 65536
        self.hud_c_vbo = self.ctx.buffer(reserve=self.hud_c_max_verts * 7 * 4)   # 3 pos + 4 colour
        self.hud_c_vao = self.ctx.vertex_array(
            self._hud_c_prog, [(self.hud_c_vbo, '3f 4f', 'in_position', 'in_color')])
        self._hud_ortho = None      # set by whichever 2D pass is active (see render_boost_hud)

        ############################################

        # Auto-enable multisampling if we have multiple samples
        self.ctx.multisample = self.samples > 1

        ############################################

        print("Done.")

    def _split_arena(self):
        """The arena mesh is drawn twice (opaque floor/curves, then the translucent glass walls/ceiling) and every
        pass shaded -- then discarded -- the pixels of the other: the whole floor was rasterised again for the glass.
        Split the triangles once into the ones that CAN produce opaque pixels and the ones that CAN produce glass
        pixels (conservative, the shader still classifies per pixel), so each pass only rasterises its own part."""
        P, N = [], []
        tris = []
        with open(os.path.join(DATA_DIR_PATH, "ArenaMeshCustom.obj"), "r") as f:
            for ln in f:
                if ln.startswith("v "):
                    P.append([float(x) for x in ln.split()[1:4]])
                elif ln.startswith("vn "):
                    N.append([float(x) for x in ln.split()[1:4]])
                elif ln.startswith("f "):
                    tris.append([tuple(int(x) - 1 if x else -1 for x in c.split("/")) for c in ln.split()[1:4]])
        P = np.asarray(P, "f4"); N = np.asarray(N, "f4")
        vi = np.asarray([[c[0] for c in t] for t in tris]); ni = np.asarray([[c[2] for c in t] for t in tris])
        tp, tn = P[vi], N[ni]                                  # (T, 3, 3)
        z, nz, ay = tp[:, :, 2], tn[:, :, 2], np.abs(tp[:, :, 1])
        grid = ((z > -7.2) & (z < -7.0)).all(1)
        goal = (ay > 5130.0).any(1)
        glass = ~grid & ((nz < -0.12).any(1) | (z >= 235.0).any(1) | goal)
        opaque = grid | goal | ((nz > -0.5) & (z < 265.0)).any(1)
        self._arena_parts = []
        for m, prog in ((opaque, self._prog_arena_opq), (glass, self._prog_arena_glass)):
            data = np.concatenate([tp[m], tn[m]], 2).reshape(-1, 6).astype("f4")
            self._arena_parts.append(self.ctx.vertex_array(
                prog, [(self.ctx.buffer(data.tobytes()), "3f 3f", "in_position", "in_normal")]))
        print("[arena] {} triangles: {} opaque-pass, {} glass-pass".format(len(tris), int(opaque.sum()), int(glass.sum())))

    def load_vao(self, model_name, program = None):
        loader = wvf.Loader(wvf.SceneDescription(path = DATA_DIR_PATH + "/" + model_name))
        model = loader.load()
        self.vaos[model_name] = model.root_nodes[0].mesh.vao.instance(self.prog if (program is None) else program)
        if not (self.outline_renderer is None):
            self.outline_renderer.load_vao(model_name, model)

    def _ensure_pad_cache(self, state):
        """Build the per-pad (is_big, static model matrix) cache once. Pads never move, so this only
        rebuilds if the pad layout actually changes (e.g. a custom map sends new locations)."""
        pads = state.boost_pad_locations
        sig = (len(pads), float(pads[0].x) if pads else 0.0, float(pads[-1].y) if pads else 0.0)
        if self._pad_static is not None and self._pad_sig == sig:
            return
        self._pad_sig = sig
        self._pad_static = []
        s = 2.5
        for p in pads:
            is_big = bool(p.z == 73)
            pos = Vector3((p.x, p.y, 0.0))
            forward = Vector3((1, 0, 0)) * s
            right = Vector3(pyrr.vector3.cross(Vector3((1, 0, 0)), Vector3((0, 0, 1)))) * s
            up = Vector3((0, 0, 1)) * s
            mat = Matrix44([
                forward[0], forward[1], forward[2], 0,
                -right[0], -right[1], -right[2], 0,
                up[0], up[1], up[2], 0,
                pos[0], pos[1], pos[2], 1,
            ]).astype('f4')
            self._pad_static.append((is_big, mat))

    def render_model(self,
                     pos, forward, up,
                     model_name, texture, scale = 1.0, global_color = None,
                     mode = moderngl.TRIANGLES, outline_color: Vector3 = None, vert_amount = None,
                     first_vert = 0):

        if pos is None:
            model_mat = Matrix44.identity()
        else:
            pos = Vector3(pos)
            forward = Vector3(forward)
            up = Vector3(up)
            right = fastvec.cross(forward, up)

            forward *= scale
            right *= scale
            up *= scale

            model_mat = Matrix44([
                forward[0], forward[1], forward[2], 0,
                -right[0], -right[1], -right[2], 0,
                up[0], up[1], up[2], 0,
                pos[0], pos[1], pos[2], 1
            ])

        self.pr_m_model.write(model_mat.astype('f4'))
        self.pra_m_model.write(model_mat.astype('f4'))

        if global_color is None:
            global_color = Vector4((0, 0, 0, 0))
        self.pr_global_color.write(global_color.astype('f4'))

        if texture is not None:
            texture.use()
        else:
            self.t_none.use()

        self.render_target.use()
        self.vaos[model_name].render(mode, vertices=(-1 if (vert_amount is None) else vert_amount),
                                     first=first_vert)

        if outline_color is not None:
            self.outline_renderer.use_framebuf()
            self.outline_renderer.pr_m_model.write(model_mat.astype('f4'))
            self.outline_renderer.pr_color.write(Vector4((outline_color.x, outline_color.y, outline_color.z, 1)).astype('f4'))
            self.outline_renderer.vaos[model_name].render(mode)

    def render_ribbon(self, ribbon: RibbonEmitter, camera_pos, lifetime, width, start_taper_time, color):
        if len(ribbon.points) == 0:
            return

        first_point = ribbon.points[0]
        cam_to_ribbon_dir = safe_normalize(-(Vector3(first_point.pos) - camera_pos))
        ribbon_away_dir = safe_normalize(Vector3(first_point.vel if first_point.vel is not None else (1.0, 0.0, 0.0)))
        ribbon_sideways_dir = ribbon_away_dir.cross(cam_to_ribbon_dir)

        # VECTORISED build (identical geometry). The old version ran a Python loop doing pyrr Vector3
        # arithmetic per point and appended two Vector3s each -- every one of those allocates, and it
        # runs for EVERY ribbon (one per car plus the ball) EVERY frame, which is the bulk of the
        # "cars" stage. Positions/times are pulled once into arrays and the offsets applied in numpy.
        pts = [pt for pt in ribbon.points if pt.connected]
        if not pts:
            return
        pos = np.array([pt.pos for pt in pts], dtype='f4')                   # (P,3)
        ta = np.array([ribbon.clock - pt.t0 for pt in pts], dtype='f4')      # (P,)
        with np.errstate(divide='ignore', invalid='ignore'):
            ws = np.where(ta < start_taper_time,
                          ta / max(start_taper_time, 1e-9),
                          1.0 - (ta / max(lifetime, 1e-9)))
        side = np.asarray(tuple(ribbon_sideways_dir), dtype='f4')
        off = side[None, :] * (float(width) * ws)[:, None]                   # (P,3)
        verts = np.empty((2 * len(pos), 3), dtype='f4')
        verts[0::2] = pos - off                                              # i=0 -> offset_scale -1
        verts[1::2] = pos + off                                              # i=1 -> offset_scale +1
        if len(verts) > self.ribbon_max_verts:
            verts = verts[:self.ribbon_max_verts]
        vertices = verts

        n = len(vertices)
        if n == 0:
            return

        # Upload + draw only the actual vertices. Previously every ribbon was padded out to
        # ribbon_max_verts (1000) with a Python loop and a 1000-vertex triangle-strip draw, almost
        # all of it degenerate — a big chunk of the per-frame "cars" cost. A partial VBO write +
        # vert_amount draw renders the identical visible trail for a fraction of the work.
        self.ribbon_vbo.write(vertices, 0)

        self.ctx.disable(moderngl.CULL_FACE)
        self.render_model(
            None, None, None,
            "ribbon", self.t_none, scale=20,
            global_color=color,
            mode=moderngl.TRIANGLE_STRIP,
            vert_amount=n
        )
        self.ctx.enable(moderngl.CULL_FACE)

    # ---- 2D HUD (screen-space overlay) ------------------------------------

    SS_TRAIL_LIFE = 0.13             # supersonic trail length in seconds (~1.3 car lengths at supersonic speed)
    # Car speed (uu/s) at/above which the car is supersonic
    SUPERSONIC_SPEED = 2200.0
    BOOST_SOUND_HOLD = 0.35     # s: feathered boost presses closer than this = one boost (sound only)

    # 7-segment layout: which of (a, b, c, d, e, f, g) are lit per digit
    #    _a_
    #  f|   |b
    #    _g_
    #  e|   |c
    #    _d_
    _SEVEN_SEG = {
        '0': (1, 1, 1, 1, 1, 1, 0),
        '1': (0, 1, 1, 0, 0, 0, 0),
        '2': (1, 1, 0, 1, 1, 0, 1),
        '3': (1, 1, 1, 1, 0, 0, 1),
        '4': (0, 1, 1, 0, 0, 1, 1),
        '5': (1, 0, 1, 1, 0, 1, 1),
        '6': (1, 0, 1, 1, 1, 1, 1),
        '7': (1, 1, 1, 0, 0, 0, 0),
        '8': (1, 1, 1, 1, 1, 1, 1),
        '9': (1, 1, 1, 1, 0, 1, 1),
    }

    def _hud_batch_draw(self, items):
        """items = [(verts, color, mode), ...] -> ONE hud_vbo.write + one draw per item at a vertex
        offset, preserving draw order (alpha blending depends on it).

        Why this exists: _hud_draw writes the SHARED hud_vbo and immediately draws it, so calling it
        N times per frame forces N GPU pipeline stalls (each write must wait for the previous draw to
        release the buffer). The boost gauge's radial fade issues 120 such calls per frame, which
        measured at 10-16 ms -- the entire frame budget -- while the 34 boost-pad draws cost 0.11 ms
        total. Draw calls are cheap; write-then-draw on a live buffer is not. Batching removes 119 of
        the 120 writes and leaves the visuals byte-identical (same geometry, colours and order)."""
        items = [(v, c, m) for (v, c, m) in items if len(v) > 0]
        if not items:
            return
        blocks, offsets, off = [], [], 0
        for verts, _c, _m in items:
            arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
            if arr.dtype != np.float32:
                arr = arr.astype('f4')
            offsets.append((off, len(arr)))
            blocks.append(arr)
            off += len(arr)
        if off > self.hud_max_verts:                      # never overrun the buffer; drop the tail
            keep = 0
            while keep < len(offsets) and offsets[keep][0] + offsets[keep][1] <= self.hud_max_verts:
                keep += 1
            if keep == 0:
                return
            items, offsets, blocks = items[:keep], offsets[:keep], blocks[:keep]
    def _hud_stitch(self, items):
        """[(verts, rgba, TRIANGLE_STRIP), ...] -> one interleaved (N,7) pos+colour array, the strips
        joined by DEGENERATE (zero-area) triangles. Zero-area triangles emit no fragments and strip
        order is preserved, so the blended result is identical to drawing each band separately.
        Cull/depth are already off for 2D, so the winding flips stitching causes are harmless."""
        items = [(v, c, m) for (v, c, m) in items if len(v) > 0]
        if not items:
            return None
        total = sum(len(v) for v, _c, _m in items) + 2 * (len(items) - 1)
        if total > self.hud_c_max_verts:
            return None
        inter = np.empty((total, 7), dtype='f4')
        w = 0
        for i, (verts, color, _m) in enumerate(items):
            arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
            if arr.dtype != np.float32:
                arr = arr.astype('f4')
            rgba = np.asarray(color, dtype='f4')
            if i > 0:                                   # degenerate bridge: prev last, then new first
                inter[w] = inter[w - 1]
                inter[w + 1, 0:3] = arr[0]
                inter[w + 1, 3:7] = rgba
                w += 2
            n = len(arr)
            inter[w:w + n, 0:3] = arr
            inter[w:w + n, 3:7] = rgba
            w += n
        return inter[:w]

    def _hud_draw_strip(self, inter):
        """Draw a pre-stitched interleaved strip from _hud_stitch (one write + one draw)."""
        if inter is None or len(inter) == 0 or self._hud_ortho is None:
            return
        self.hud_c_vbo.write(inter.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(self._hud_ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLE_STRIP, vertices=len(inter))

    _IDENT_F4 = np.eye(4, dtype="f4").tobytes()

    def _hud_draw(self, verts, color, mode):
        """One HUD draw with a uniform colour. Lean path (no render_model / pyrr per call), and each
        draw's vertices go to a fresh region of hud_vbo (offset reset once per frame in paint()), so
        a write never has to wait for the GPU to finish reading the previous draw's vertices."""
        n = len(verts)
        if n == 0:
            return
        arr = verts if isinstance(verts, np.ndarray) else np.asarray(verts, dtype='f4')
        if arr.dtype != np.float32:
            arr = arr.astype('f4')
        off = getattr(self, "_hud_off", 0)
        if off + n > self.hud_max_verts:
            off = 0
        self.hud_vbo.write(arr.tobytes(), off * 12)
        self._hud_off = off + n
        self.pr_m_model.write(self._IDENT_F4)
        self.pr_global_color.write(np.asarray(color, dtype='f4').tobytes())
        self.t_none.use()
        self.render_target.use()
        self.vaos["hud"].render(mode, vertices=n, first=off)

    def _hud_draw_tris(self, inter):
        """Per-vertex-colour triangles (interleaved (N,7) pos+rgba) in one draw."""
        if inter is None or len(inter) == 0 or self._hud_ortho is None:
            return
        self.hud_c_vbo.write(inter.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(self._hud_ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLES, vertices=len(inter))

    @staticmethod
    def _disc_verts(cx, cy, r, segments=48):
        # Triangle fan: centre + perimeter (screen coords, y points down)
        verts = [(cx, cy, 0.0)]
        for k in range(segments + 1):
            a = 2.0 * math.pi * k / segments
            verts.append((cx + r * math.sin(a), cy - r * math.cos(a), 0.0))
        return verts

    @staticmethod
    def _ring_verts(cx, cy, r_in, r_out, frac, segments=48):
        # Triangle strip annulus sweeping `frac` of a full turn, starting at top
        frac = max(0.0, min(1.0, frac))
        if frac <= 0.0:
            return []
        n = max(1, int(round(segments * frac)))
        verts = []
        for k in range(n + 1):
            a = 2.0 * math.pi * frac * (k / n)
            sa, ca = math.sin(a), math.cos(a)
            verts.append((cx + r_out * sa, cy - r_out * ca, 0.0))
            verts.append((cx + r_in * sa, cy - r_in * ca, 0.0))
        return verts

    @staticmethod
    def _arc_ring_verts(cx, cy, r_in, r_out, start_deg, span_deg, frac, segments=64):
        """Triangle strip annulus over an ARBITRARY arc (not a full circle) -- the actual Rocket League
        boost-meter shape: a ring with a GAP at the bottom, not a closed loop. Angle convention: 0deg =
        straight up (12 o'clock), increasing CLOCKWISE (90=3 o'clock/right, 180=6 o'clock/bottom,
        270=9 o'clock/left) -- matches how a viewer reads clock positions on screen (y-down space).
        Sweeps `frac` (0..1) of `span_deg` starting at `start_deg`, so frac=1 draws the FULL arc (the
        boost track) and frac=<boost/100> draws just the filled portion."""
        frac = max(0.0, min(1.0, frac))
        if frac <= 0.0 or span_deg <= 0.0:
            return []
        sweep = span_deg * frac
        n = max(1, int(round(segments * frac)))
        verts = []
        for k in range(n + 1):
            a = math.radians(start_deg + sweep * (k / n))
            sa, ca = math.sin(a), math.cos(a)
            verts.append((cx + r_out * sa, cy - r_out * ca, 0.0))
            verts.append((cx + r_in * sa, cy - r_in * ca, 0.0))
        return verts

    @staticmethod
    def _rect_verts(x0, y0, x1, y1):
        # Two triangles for an axis-aligned rectangle
        return [
            (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0),
            (x0, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
        ]

    # Minimal 5x7 dot-matrix font used ONLY by the GAIL p_ai HUD (render_pai_hud), which needs
    # letters/punctuation that the 7-segment digit font above can't draw. Covers exactly the
    # characters that string can contain: "P_AI <val|N/A>  <label> (HARD)". Add glyphs here if the
    # sender ever emits a wider vocabulary. Lookup is upper-cased, so lowercase input still resolves.
    _FONT_5X7 = {
        '0': ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        '1': ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        '2': ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        '3': ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
        '4': ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        '5': ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
        '6': ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
        '7': ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        '8': ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        '9': ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
        'A': ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        'D': ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        'H': ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        'R': ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        'N': ("10001", "11001", "10101", "10101", "10011", "10001", "10001"),
        'P': ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        'I': ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
        'E': ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        'L': ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        'O': ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        'T': ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        '+': ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
        '_': ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
        '.': ("00000", "00000", "00000", "00000", "00000", "00000", "00100"),
        '-': ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
        '/': ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
        '(': ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
        ')': ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
        ' ': ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    }

    @classmethod
    def _char_pixel_verts(cls, ch, x, y, w, h):
        """Filled-pixel 5x7 dot-matrix glyph, top-left at (x, y), cell size (w, h). Unknown
        characters (not in _FONT_5X7) draw nothing rather than raising, so a stray/unsupported
        character in a sender-supplied string just leaves a blank cell instead of crashing render()."""
        pattern = cls._FONT_5X7.get(ch.upper())
        if not pattern:
            return []
        cols, rows = 5, 7
        cell_w = w / cols
        cell_h = h / rows
        pad = min(cell_w, cell_h) * 0.14  # small gap between pixels for a legible "LED" look
        verts = []
        for r, row in enumerate(pattern):
            for c, bit in enumerate(row):
                if bit == '1':   # patterns are '1'/'0' bit-strings (was '#' -> never matched -> blank text)
                    x0 = x + c * cell_w + pad
                    x1 = x + (c + 1) * cell_w - pad
                    y0 = y + r * cell_h + pad
                    y1 = y + (r + 1) * cell_h - pad
                    verts += cls._rect_verts(x0, y0, x1, y1)
        return verts

    @staticmethod
    def digit_advance(ch, w):
        # '1' is just a single bar, so it gets a narrower cell -> proper centring
        return w * 0.45 if ch == '1' else w

    @classmethod
    def _digit_verts(cls, ch, x, y, w, h, t):
        """Filled-quad 7-segment digit, top-left at (x, y), size (w, h), thickness t."""
        # '1' has no left column, so draw its bar centred in the cell instead of
        # hugging the right edge (otherwise numbers like "100" look off-centre).
        if ch == '1':
            bx = x + (w - t) / 2.0
            return cls._rect_verts(bx, y, bx + t, y + h)

        segs = cls._SEVEN_SEG.get(ch)
        if segs is None:
            return []
        a, b, c, d, e, f, g = segs
        hh = h / 2.0
        rects = []  # (x0, y0, x1, y1)

        if a: rects.append((x, y, x + w, y + t))                       # top
        if g: rects.append((x, y + hh - t / 2.0, x + w, y + hh + t / 2.0))  # middle
        if d: rects.append((x, y + h - t, x + w, y + h))               # bottom
        if f: rects.append((x, y, x + t, y + hh))                      # top-left
        if b: rects.append((x + w - t, y, x + w, y + hh))              # top-right
        if e: rects.append((x, y + hh, x + t, y + h))                  # bottom-left
        if c: rects.append((x + w - t, y + hh, x + w, y + h))          # bottom-right

        verts = []
        for (x0, y0, x1, y1) in rects:
            verts += cls._rect_verts(x0, y0, x1, y1)
        return verts

    def render_boost_hud(self, width, height, boost_amount, team_num=0):
        """RL-style boost meter, bottom-right: dark disc, segmented arc (gap at the bottom-right) filled
        with a team gradient + glow, and the boost number in a heavy italic font. The ring is ONE
        analytic-shader quad (smooth at any size) and the number is a font-atlas text draw."""
        boost = int(round(max(0.0, min(100.0, boost_amount))))
        R = max(52.0, min(width, height) * 0.095)
        margin = R * 0.45 + min(width, height) * 0.02
        cx, cy = width - R - margin, height - R - margin
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        if int(team_num) == 1:
            c1, c2, track = (1.0, 0.72, 0.18), (1.0, 0.36, 0.04), (0.22, 0.12, 0.04)
        else:
            c1, c2, track = (0.45, 0.85, 1.0), (0.12, 0.42, 1.0), (0.05, 0.10, 0.22)
        e = R * 1.12
        q = 1.12
        quad = np.asarray([(cx - e, cy - e, -q, -q), (cx + e, cy - e, q, -q), (cx + e, cy + e, q, q),
                           (cx - e, cy - e, -q, -q), (cx + e, cy + e, q, q), (cx - e, cy + e, -q, q)], "f4")
        self.gauge_vbo.write(quad.tobytes())
        g = self.prog_gauge
        g["m_vp"].write(ortho.astype("f4"))
        g["fill"].value = boost / 100.0
        g["fillCol"].value = c1
        g["fillCol2"].value = c2
        g["trackCol"].value = track
        g["startDeg"].value = 197.0
        g["spanDeg"].value = 250.0
        self.render_target.use()
        self.gauge_vao.render(moderngl.TRIANGLES, vertices=6)
        self._draw_number(str(boost), cx, cy + R * 0.02, R * 0.78, (1.0, 1.0, 1.0, 1.0))
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render_scoreboard_hud(self, width, height, scoreboard):
        """Top-center match scoreboard: "<blue>   <M:SS>   <orange>" -- clock BETWEEN the scores,
        replaced by "OT" once regulation time is up. Handles multi-digit scores (widths are measured,
        not assumed). The vis runs a real match clock via EnvSetConfig::fullGame, re-rolled per
        episode (randomizeScoreboardOnReset) since only the displayed arena is ever stepped.

        FLAG-GATED like render_pai_hud: `scoreboard` comes from socket_listener's parse of the
        optional packet field "scoreboard" ({"score_blue", "score_orange", "time_left",
        "is_overtime"}) and defaults to None when the sender doesn't include it -- so senders that
        never set it (state-set editor, clip players) draw nothing.

        Reuses render_boost_hud's 7-segment digit font for the numerals and the 5x7 dot-matrix font
        only for the "OT" letters, which the 7-segment font can't draw.
        """
        if not scoreboard:
            return

        score_blue = int(scoreboard.get("score_blue", 0) or 0)
        score_orange = int(scoreboard.get("score_orange", 0) or 0)
        time_left = float(scoreboard.get("time_left", 0.0) or 0.0)
        is_overtime = bool(scoreboard.get("is_overtime", False))

        mins, secs = divmod(max(0, int(round(time_left))), 60)
        clock_str = "{:d}:{:02d}".format(mins, secs)

        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self.pr_m_vp.write(ortho.astype('f4'))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Layout: [blue score]   [clock]   [orange score] -- clock BETWEEN the two scores, one
        # centered row, no panels. Every size derives from `ch` so the whole thing scales with the
        # window and stays correct through a live resize.
        ch = float(np.clip(height * 0.052, 34.0, 104.0))    # clock digit height (the dominant element)
        cw = ch * 0.58
        ct = ch * 0.125                                     # 7-segment stroke thickness (lighter)
        cgap = cw * 0.36                                    # more air between digits
        dh = ch * 0.86                                      # score digits, slightly smaller than the clock
        dw = dh * 0.58
        dt = dh * 0.135
        dgap = dw * 0.36
        side_gap = ch * 0.85                                # space between a score and the clock

        COLON_ADV = 0.42                                    # colon cell is narrower than a digit

        def measure(text, w, gap):
            """(total_width, advances) -- MUST mirror the advances used when drawing below, or the
            centering and the backing panel drift from what actually appears."""
            adv = [w * COLON_ADV if c == ':' else self.digit_advance(c, w) for c in text]
            return sum(adv) + gap * max(0, len(text) - 1), adv

        blue_str, orange_str = str(score_blue), str(score_orange)   # 2+ digits handled by measure()
        blue_w, blue_adv = measure(blue_str, dw, dgap)
        orange_w, orange_adv = measure(orange_str, dw, dgap)
        if is_overtime:
            mid_adv = [cw] * 2
            mid_w = 2 * cw + cgap
        else:
            mid_w, mid_adv = measure(clock_str, cw, cgap)

        total_w = blue_w + side_gap + mid_w + side_gap + orange_w
        top_y = max(14.0, height * 0.022)
        x0 = width / 2.0 - total_w / 2.0

        # Backing panel sized to the CLOCK row (the tallest element), with the scores centred in it.
        pad_x, pad_y = ch * 0.55, ch * 0.26
        self._hud_draw(
            self._rect_verts(x0 - pad_x, top_y - pad_y, x0 + total_w + pad_x, top_y + ch + pad_y),
            (0.0, 0.0, 0.0, 0.62), moderngl.TRIANGLES
        )

        blue_rgb, orange_rgb = (0.30, 0.65, 1.0), (1.0, 0.60, 0.10)
        score_y = top_y + (ch - dh) / 2.0                   # vertically centre scores against the clock

        x = x0
        verts = []
        for c, adv in zip(blue_str, blue_adv):
            verts += self._digit_verts(c, x, score_y, adv, dh, dt)
            x += adv + dgap
        self._hud_draw(verts, (blue_rgb[0], blue_rgb[1], blue_rgb[2], 1.0), moderngl.TRIANGLES)

        x = x0 + blue_w + side_gap
        clock_color = (1.0, 0.85, 0.2, 1.0) if is_overtime else (0.96, 0.96, 0.96, 1.0)
        verts = []
        if is_overtime:
            for c in "OT":
                verts += self._char_pixel_verts(c, x, top_y, cw, ch)
                x += cw + cgap
        else:
            for c, adv in zip(clock_str, mid_adv):
                if c == ':':
                    # Two square pips, inset from the digit top/bottom so the colon reads as part of
                    # the same 7-segment face instead of a stray pair of blocks.
                    s = ct * 1.45
                    px = x + (adv - s) / 2.0
                    verts += self._rect_verts(px, top_y + ch * 0.26, px + s, top_y + ch * 0.26 + s)
                    verts += self._rect_verts(px, top_y + ch * 0.60, px + s, top_y + ch * 0.60 + s)
                else:
                    verts += self._digit_verts(c, x, top_y, adv, ch, ct)
                x += adv + cgap
        self._hud_draw(verts, clock_color, moderngl.TRIANGLES)

        x = x0 + blue_w + side_gap + mid_w + side_gap
        verts = []
        for c, adv in zip(orange_str, orange_adv):
            verts += self._digit_verts(c, x, score_y, adv, dh, dt)
            x += adv + dgap
        self._hud_draw(verts, (orange_rgb[0], orange_rgb[1], orange_rgb[2], 1.0), moderngl.TRIANGLES)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    # Supersonic speed lines (RL): thin white streaks in the air around and ahead of the spectated car, lying along
    # its velocity, that it flies past -- in the 3D scene (so they sit in the picture and follow the car's direction
    # in perspective), not 2D darts from the screen edges. Rare and faint.
    SPEED_LINE_RATE = 7.0            # new lines per second at full supersonic

    SPEED_LINE_BURST_S = 0.8         # the strong first moments after the car becomes supersonic
    SPEED_LINE_RATE = 3.0            # new lines per second afterwards (rare)

    def _update_speed_lines(self, state, interp_ratio, spectated):
        """Supersonic speed lines (RL): thin white 3D streaks lying along the spectated car's direction of travel, at
        fixed places in the world ahead of and around it -- the car rushes past them, so on screen they slide the
        opposite way to the car. Each fades to nothing toward both of its ends (no hard start or end) and keeps its
        brightness until the car has passed it (no fade in / out over time). A burst right after the car goes
        supersonic, then only a few faint ones."""
        lines = self.__dict__.setdefault("_speed_lines", [])
        now = time.time()
        dt = max(0.0, min(0.1, now - getattr(self, "_sl_last", now)))
        self._sl_last = now
        cam = getattr(self, "_frame_cam", None)
        car = None
        if 0 <= spectated < len(state.car_states) and not state.car_states[spectated].is_demoed:
            car = state.car_states[spectated]
        sp = 0.0
        if car is not None:
            v = np.asarray(tuple(car.phys.get_vel(interp_ratio)), "f8")
            sp = float(np.linalg.norm(v))
        if car is None or sp < self.SUPERSONIC_SPEED:
            self._sl_on = False
            self._speed_lines = []
            return
        if not getattr(self, "_sl_on", False):
            self._sl_on, self._sl_t0 = True, now
        burst = max(0.0, 1.0 - (now - self._sl_t0) / self.SPEED_LINE_BURST_S)
        vd = v / sp
        pos = np.asarray(tuple(car.phys.get_pos(interp_ratio)), "f8")
        self._sl_acc = min(getattr(self, "_sl_acc", 0.0) + dt * (self.SPEED_LINE_RATE + 40.0 * burst * burst), 30.0)
        a_ = np.asarray(fastvec.cross(vd, (0.0, 0.0, 1.0)), "f8")
        a_ = a_ / np.linalg.norm(a_) if np.linalg.norm(a_) > 1e-3 else np.array([1.0, 0.0, 0.0])
        b_ = np.asarray(fastvec.cross(vd, a_), "f8")
        alpha = 0.22 + 0.33 * burst
        while self._sl_acc >= 1.0:
            self._sl_acc -= 1.0
            ang = random.uniform(0.0, 2.0 * math.pi)
            p = pos + vd * random.uniform(300.0, 1600.0) + (a_ * math.cos(ang) + b_ * math.sin(ang)) * random.uniform(90.0, 520.0)
            p[2] = max(p[2], random.uniform(8.0, 60.0))                      # never under the floor
            lines.append((p, vd.copy(), random.uniform(260.0, 620.0), alpha * random.uniform(0.7, 1.0),
                          random.uniform(1.4, 2.4)))
        if not lines:
            self._speed_lines = []
            return
        P = np.array([ln[0] for ln in lines]); D = np.array([ln[1] for ln in lines])
        Ls = np.array([ln[2] for ln in lines]); A0 = np.array([ln[3] for ln in lines]); Wd = np.array([ln[4] for ln in lines])
        # gone once the car (and the camera) has passed it
        ok = (P - pos) @ vd >= -900.0
        if cam is not None:
            ok &= (P - np.array((float(cam[0]), float(cam[1]), float(cam[2])))) @ vd >= -Ls
        idx = np.nonzero(ok)[0][-80:]
        pts = np.stack([P[idx], P[idx] - D[idx] * (Ls[idx] * 0.5)[:, None], P[idx] - D[idx] * Ls[idx][:, None]], 1).astype("f4")
        wid = (Wd[idx][:, None] * np.array([0.3, 1.0, 0.3])).astype("f4")
        col = np.zeros((len(idx), 3, 4), "f4"); col[:, :, :3] = 1.0; col[:, 1, 3] = A0[idx]
        if len(idx):
            self.fx._beam_blocks.append((pts, wid, col))
        self._speed_lines = [lines[i] for i in idx]

    def _heading_point(self, vel, width, height):
        """Screen point the car is heading to (its velocity's vanishing point), or None (no speed / behind)."""
        vp, cam = getattr(self, "_frame_vp", None), getattr(self, "_frame_cam", None)
        v = np.asarray(tuple(vel), "f8")
        sp = float(np.linalg.norm(v))
        if vp is None or cam is None or sp < 1.0:
            return None
        p = np.asarray(tuple(cam), "f8") + v / sp * 20000.0
        clip = np.array([p[0], p[1], p[2], 1.0]) @ np.asarray(vp, "f8")
        if clip[3] <= 1e-3:
            return None
        return ((clip[0] / clip[3] * 0.5 + 0.5) * width, (0.5 - clip[1] / clip[3] * 0.5) * height)

    SS_BURST_S = 0.9                 # the strong first moments after the car becomes supersonic
    SS_BASE_RATE = 2.6               # lines per second afterwards (rare)

    def render_supersonic_streaks(self, width, height, intensity, t, foe=None):
        """Supersonic speed lines, drawn in 2D but laid out like 3D: thin straight lines radiating from the point the car
        is heading to (the vanishing point of its velocity), sliding outward and growing the way something passing by in
        3D would. No fades: a line is at one steady brightness for its short life and simply appears / disappears.
        Right after the car turns supersonic there is a burst of them (like RL's), then only a few faint ones."""
        now = float(t)
        st = self.__dict__.setdefault("_ssl", {"lines": [], "on": False, "t0": 0.0, "acc": 0.0, "last": now})
        dt = max(0.0, min(0.1, now - st["last"]))
        st["last"] = now
        on = intensity > 0.0
        if not on:
            st["lines"] = []
            st["on"] = False
            return
        if not st["on"]:
            st["on"] = True
            st["t0"] = now
        cx, cy = foe if foe is not None else (width * 0.5, height * 0.5)
        md = float(min(width, height))
        burst = max(0.0, 1.0 - (now - st["t0"]) / self.SS_BURST_S)
        st["acc"] += dt * (self.SS_BASE_RATE + 46.0 * burst * burst)
        alpha = 0.16 + 0.42 * burst
        lines = st["lines"]
        while st["acc"] >= 1.0 and len(lines) < 60:
            st["acc"] -= 1.0
            lines.append([random.uniform(0.0, 2.0 * math.pi), md * random.uniform(0.09, 0.34),
                          md * random.uniform(0.13, 0.30), now, random.uniform(0.22, 0.42), alpha,
                          random.uniform(0.9, 1.5)])
        st["acc"] = min(st["acc"], 1.0)
        keep, verts = [], []
        px = md / 1080.0
        for ln in lines:
            ang, r0, L0, t0, life, a0, wpx = ln
            age = now - t0
            if age >= life:
                continue
            keep.append(ln)
            g = math.exp(1.7 * age)                       # perspective: things move out from the vanishing point faster the farther out
            r_in, L = r0 * g, L0 * g
            dx, dy = math.cos(ang), math.sin(ang)
            x0, y0 = cx + dx * r_in, cy + dy * r_in
            x1, y1 = cx + dx * (r_in + L), cy + dy * (r_in + L)
            w0, w1 = 0.4 * wpx * px, wpx * px * (0.8 + 0.8 * min(1.0, (r_in + L) / md))
            nx, ny = -dy, dx
            for (x, y) in ((x0 + nx * w0, y0 + ny * w0), (x0 - nx * w0, y0 - ny * w0), (x1 + nx * w1, y1 + ny * w1),
                           (x0 - nx * w0, y0 - ny * w0), (x1 - nx * w1, y1 - ny * w1), (x1 + nx * w1, y1 + ny * w1)):
                verts.append((x, y, 0.0, 1.0, 1.0, 1.0, a0))
        st["lines"] = keep
        if not verts:
            return
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.pr_m_vp.write(ortho.astype('f4'))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self._hud_draw_tris(np.asarray(verts, dtype="f4"))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render_pai_hud(self, width, height, gail_hud):
        """GAIL-training-only discriminator readout: LARGE RED text, top-right, e.g.
        "P_AI 0.05  D2" (appends " (HARD)" when flagged, shows "N/A" when p_ai is null).

        FLAG-GATED: `gail_hud` comes straight from socket_listener's parse of the optional
        packet field "gail_hud" ({"p_ai": float|null, "label": str, "hard": bool}) and defaults to
        None when the sender doesn't include it. None/empty here means normal (non-GAIL) play mode
        -> this function draws nothing at all. The mere presence of the field is the flag; there is
        no separate on/off switch to keep in sync.

        Mirrors render_boost_hud's approach (ortho projection + the same hud_vbo/_hud_draw quad
        pipeline), but needs letters the 7-segment digit font can't draw, so it uses the 5x7
        dot-matrix font (_FONT_5X7 / _char_pixel_verts) defined above instead.

        Resize-safe: every size/position below is recomputed from the CURRENT width/height on
        every call (no cached/absolute pixel values), so it stays correctly scaled and anchored
        top-right through a live window resize.
        """
        if not gail_hud:
            return

        p_ai = gail_hud.get("p_ai")
        label = str(gail_hud.get("label") or "D")
        hard = bool(gail_hud.get("hard", False))
        elo = gail_hud.get("elo")

        pai_str = "N/A" if p_ai is None else "{:.2f}".format(float(p_ai))
        line1 = "P_AI {}  {}".format(pai_str, label)
        if hard:
            line1 += " (HARD)"
        lines = [line1]
        if elo is not None:                                # ELO under the p_ai text, SAME size
            lines.append("ELO {:+d}".format(int(elo)))

        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.pr_m_vp.write(ortho.astype('f4'))

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # the biggest HUD text
        ch_h = float(np.clip(height * 0.035, 21.0, 91.0))
        ch_w = ch_h * 0.62
        gap = ch_w * 0.32
        advance = ch_w + gap
        line_gap = ch_h * 0.35                              # vertical gap between the stacked lines
        total_w = max(advance * len(t) - gap for t in lines)
        block_h = ch_h * len(lines) + line_gap * (len(lines) - 1)

        margin = max(20.0, min(width, height) * 0.025)
        right_x = width - margin
        top_y = margin

        # Dim backing panel behind BOTH lines so red text stays legible over a bright sky.
        pad_x, pad_y = ch_w * 0.5, ch_h * 0.22
        self._hud_draw(
            self._rect_verts(right_x - total_w - pad_x, top_y - pad_y, right_x + pad_x, top_y + block_h + pad_y),
            (0.0, 0.0, 0.0, 0.55), moderngl.TRIANGLES
        )

        # The 5x7 dot-matrix glyphs are rebuilt as individual quads in Python, ~2 ms/frame for two
        # lines. The text only changes when p_ai/Elo change (once per training iteration at most), so
        # cache the geometry and rebuild only when the strings or the layout actually differ.
        ck = (tuple(lines), round(right_x, 2), round(top_y, 2), round(ch_w, 3), round(ch_h, 3),
              round(advance, 3), round(gap, 3), round(line_gap, 3))
        if getattr(self, "_pai_cache_key", None) != ck:
            verts = []
            y = top_y
            for t in lines:
                lw = advance * len(t) - gap                 # right-align each line under the other
                x = right_x - lw
                for ch in t:
                    verts += self._char_pixel_verts(ch, x, y, ch_w, ch_h)
                    x += advance
                y += ch_h + line_gap
            self._pai_cache_key = ck
            self._pai_cache_verts = np.asarray(verts, dtype='f4') if verts else np.zeros((0, 3), 'f4')
        self._hud_draw(self._pai_cache_verts, (1.0, 0.08, 0.05, 1.0), moderngl.TRIANGLES)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    @staticmethod
    def _pitch_dir(d, angle_deg):
        """Rotate direction d up (+) / down (-) by angle_deg around the horizontal axis."""
        d = safe_normalize(Vector3(d))
        up = Vector3((0.0, 0.0, 1.0)) - d * float(d[2])
        if up.length < 1e-4:
            return d
        up = safe_normalize(up)
        a = math.radians(angle_deg)
        return safe_normalize(d * math.cos(a) + up * math.sin(a))

    def calc_camera_state(self, state, interp_ratio, delta_time):
        self._update_auto_spectate(state, interp_ratio)
        # Pose-editor override: sit behind the selected object (billiard/chase POV) while posing.
        pc = getattr(state_manager, "pose_cam", None)
        if pc is not None:
            aspect = max(getattr(self, "_aspect", 16 / 9), 1e-3)     # config FOV is horizontal
            return Vector3(pc[0]), Vector3(pc[1]), math.degrees(
                2.0 * math.atan(math.tan(math.radians(self.config.camera_fov.val) / 2.0) / aspect))

        pos = Vector3((-4000, 0, 1000))
        ball_pos = state.ball_state.get_pos(interp_ratio)

        cam_dir = safe_normalize(ball_pos - pos)

        is_spectating_car = self.spectate_idx > -1 and len(state.car_states) > self.spectate_idx
        if is_spectating_car:
            car_pos = state.car_states[self.spectate_idx].phys.get_pos(interp_ratio)
            car_vel = state.car_states[self.spectate_idx].phys.get_vel(interp_ratio)
            car_forward = state.car_states[self.spectate_idx].phys.get_forward(interp_ratio)

            # Ball cam -- Rocket League's model: the camera sits at the car + a YAW-ONLY offset of
            # (-Distance, 0, +Height), yawed to face the ball, looking at the ball, then pitched by the
            # Angle setting. Stiffness < 1 lets the camera pull back at speed (RL behaviour).
            if True:
                speed = car_vel.length
                stiff = self.config.camera_stiffness.val
                dist = self.config.camera_distance.val * (1.0 + (1.0 - stiff) * 0.30 * min(1.0, speed / 2300.0))
                height = self.config.camera_height.val
                to_ball = ball_pos - car_pos
                yaw_dir = Vector3((to_ball[0], to_ball[1], 0.0))
                if yaw_dir.length < 1.0:
                    yaw_dir = Vector3((car_forward[0], car_forward[1], 0.0))
                yaw_dir = safe_normalize(yaw_dir)
                ball_cam_pos = car_pos - yaw_dir * dist + Vector3((0.0, 0.0, height))
                ball_cam_dir = self._pitch_dir(safe_normalize(ball_pos - ball_cam_pos), self.config.camera_angle.val)

            # Calculate car cam dir: genuinely AIRBORNE (>=200uu above the floor AND not touching any
            # surface) -> follow MOMENTUM (velocity direction, full 3D) so an aerial dribble stays
            # stable even as the car's facing spins around underneath the ball. Near the ground OR
            # touching a WALL/ceiling (on_ground is a general surface-contact flag here, not floor-only
            # -- a car driving up a wall is NOT "in the air") -> follow the car's FACING, XY-only
            # (ignore pitch) so it reads as a normal level chase cam. car_pos.z is height above the
            # floor (floor = z=0).
            if True:
                on_surface = bool(state.car_states[self.spectate_idx].on_ground)
                height_above_floor = car_pos.z
                if on_surface or height_above_floor < 200.0:
                    cd = car_forward * Vector3((1, 1, 0))
                    car_cam_dir_raw = safe_normalize(cd) if cd.length > 0.1 else Vector3((1, 1, 0))
                else:
                    if car_vel.length > 50:
                        car_cam_dir_raw = safe_normalize(car_vel)
                    else:
                        cd = car_forward * Vector3((1, 1, 0))
                        car_cam_dir_raw = safe_normalize(cd) if cd.length > 0.1 else Vector3((1, 1, 0))
                # SAME base distance as ball cam (camera_distance.val), and -- like ball cam's own
                # "make sure we are actually of the correct distance" step -- RENORMALIZED back to that
                # exact distance AFTER the z-override below. Without this, however much of the unit
                # direction's length happened to be vertical (car_cam_dir_raw.z, discarded by the
                # z-override) was simply LOST, so the resulting 3D offset length varied by mode: ground/
                # wall's direction is always purely horizontal (nothing lost, offset stayed correct-
                # looking "by luck"), while air's direction (raw 3D velocity) often has a big vertical
                # component, silently shrinking the horizontal spread and giving air a DIFFERENT net
                # distance than ground/wall for the exact same config value. Renormalizing guarantees
                # all three modes sit the identical `dist` away from the car, matching ball cam exactly.
                # RL car cam: same yaw-only (-Distance, +Height) offset as ball cam (no renormalising
                # height into distance -- that is what made equal settings feel unlike the game),
                # looking along the heading pitched by Angle.
                dist_cc = dist
                h_dir = safe_normalize(Vector3((car_cam_dir_raw[0], car_cam_dir_raw[1], 0.0)))
                car_cam_offset_raw = -h_dir * dist_cc + Vector3((0.0, 0.0, self.config.camera_height.val))
                car_cam_dir_raw = self._pitch_dir(car_cam_dir_raw, self.config.camera_angle.val)

                # SMOOTHED car-cam movement: exponential low-pass on the CAR-RELATIVE OFFSET (not the
                # absolute world-space camera position!). Smoothing the absolute position made the
                # camera chase a moving target -- for a fast-moving car (much faster in the air than on
                # ground/wall) the steady-state lag grows with speed (lag ~= car_speed * TAU), which is
                # exactly the "2x too far, especially in the air" bug. Smoothing the OFFSET instead
                # still irons out jerky direction changes but recombines with the CURRENT car_pos every
                # frame, so there is zero translational lag regardless of how fast the car moves. Reset
                # (snap, no lag) whenever we start/re-start spectating a car so it doesn't drift in from
                # a stale previous car's state.
                # Z (vertical) gets extra smoothing (TAU doubled) so horizontal tracking stays just as
                # responsive while vertical bob/height changes ease in more gently.
                TAU_XY = 0.24
                TAU_Z = 0.48
                lerp_xy = 1.0 - math.exp(-delta_time / TAU_XY)
                lerp_z = 1.0 - math.exp(-delta_time / TAU_Z)
                prev_offset = getattr(self, "_car_cam_offset_smooth", None)
                prev_cam_dir = getattr(self, "_car_cam_dir_smooth", None)
                if prev_offset is None:
                    car_cam_offset = car_cam_offset_raw
                    car_cam_dir = car_cam_dir_raw
                else:
                    d_off = car_cam_offset_raw - prev_offset
                    car_cam_offset = (prev_offset + d_off * Vector3((1, 1, 0)) * lerp_xy
                                      + d_off * Vector3((0, 0, 1)) * lerp_z)
                    car_cam_dir = safe_normalize(prev_cam_dir + (car_cam_dir_raw - prev_cam_dir) * lerp_xy)
                self._car_cam_offset_smooth = car_cam_offset
                self._car_cam_dir_smooth = car_cam_dir
                car_cam_pos = car_pos + car_cam_offset

            # Determine if dribbling (raw, per-frame)
            car_ball_delta = ball_pos - car_pos
            dribbling = False
            if car_vel.length > 500:
                if 90 < car_ball_delta.z < 200:
                    if (car_ball_delta * Vector3((1, 1, 0))).length < 135:
                        if ball_pos.z < 300:
                            dribbling = True

            # Duration timers: how long `dribbling` has been continuously TRUE, and continuously FALSE
            # (each resets the instant the other becomes true) -- single-frame blips on EITHER side
            # shouldn't flip the cam.
            self.dribble_duration = (getattr(self, "dribble_duration", 0.0) + delta_time) if dribbling else 0.0
            self.no_dribble_duration = 0.0 if dribbling else (getattr(self, "no_dribble_duration", 0.0) + delta_time)
            confirmed_now = self.dribble_duration > 0.3
            confirmed_ended = self.no_dribble_duration > 0.3

            # Closest OPPONENT gate: >500uu from the spectated car, checked ONLY at the MOMENT the
            # dribble is first confirmed (the instant confirmed_now flips false->true this frame) -- an
            # uncontested dribble START is safe to follow tight on the car for its whole duration, even
            # if an opponent closes in later. No opponents on the field -> gate passes vacuously.
            was_confirmed = getattr(self, "_dribble_confirmed_prev", False)
            just_confirmed = confirmed_now and not was_confirmed
            self._dribble_confirmed_prev = confirmed_now
            engaged = getattr(self, "car_cam_engaged", False)
            if not engaged and just_confirmed:
                my_team = state.car_states[self.spectate_idx].team_num
                opp_dists = [(car_pos - cs.phys.get_pos(interp_ratio)).length
                             for i, cs in enumerate(state.car_states)
                             if i != self.spectate_idx and cs.team_num != my_team and not cs.is_demoed]
                opponent_clear_at_start = (min(opp_dists) > 500.0) if opp_dists else True
                if opponent_clear_at_start:
                    engaged = True
            elif engaged and confirmed_ended:
                engaged = False
            self.car_cam_engaged = engaged

            # cam_manual: None = AUTO (sticky `engaged` state above); True/False = user forced ball/car
            # cam via Space (see keyPressEvent) -- a manual override bypasses the dribble/opponent
            # gating entirely. Either way the desired state is a single boolean target that
            # self.car_cam_time (0..1) ramps toward LINEARLY at a constant rate so a full switch always
            # takes exactly TRANSITION_SEC, in EITHER direction -- ball-cam<->car-cam is never an
            # instant cut.
            cam_manual = getattr(self, "cam_manual", None)
            want_car_cam = engaged if cam_manual is None else (not cam_manual)
            TRANSITION_SEC = 0.6 / max(0.5, self.config.camera_transition_speed.val)
            target = 1.0 if want_car_cam else 0.0
            step = delta_time / TRANSITION_SEC
            if self.car_cam_time < target:
                self.car_cam_time = min(target, self.car_cam_time + step)
            else:
                self.car_cam_time = max(target, self.car_cam_time - step)
            # Slightly smoothed transition (S-curve / smoothstep): self.car_cam_time
            # is still the RAW linear 0..1 progress (so the switch still takes exactly TRANSITION_SEC
            # total), but the actual blend weight eases in/out -- slow-start, faster through the middle,
            # slow-finish -- instead of a constant-speed linear blend.
            t = self.car_cam_time
            car_cam_ratio = t * t * (3.0 - 2.0 * t)          # classic cubic smoothstep

            pos = ball_cam_pos*(1-car_cam_ratio) + car_cam_pos*car_cam_ratio
            cam_dir = safe_normalize(ball_cam_dir*(1-car_cam_ratio) + car_cam_dir*car_cam_ratio)
        else:
            self.car_cam_time = 0
            self.dribble_duration = 0.0
            self.no_dribble_duration = 0.0
            self._dribble_confirmed_prev = False
            self.car_cam_engaged = False
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None

        # RL's FOV setting is HORIZONTAL; the projection wants vertical.
        aspect = max(getattr(self, "_aspect", 16 / 9), 1e-3)
        if is_spectating_car:
            vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(self.config.camera_fov.val) / 2.0) / aspect))
        else:
            vfov = self.config.camera_bird_fov.val
        # Goal celebration: look at where the ball went in (eased in/out over ~0.4 s).
        cel = getattr(state, "celebration_pos", None)
        if cel is not None:
            self._cel_last = cel
        w0 = getattr(self, "_cel_w", 0.0)
        self._cel_w = min(1.0, w0 + delta_time / 0.4) if cel is not None else max(0.0, w0 - delta_time / 0.4)
        if self._cel_w > 0.0 and getattr(self, "_cel_last", None) is not None:
            to_goal = safe_normalize(Vector3(self._cel_last) - pos)
            wgt = self._cel_w * self._cel_w * (3.0 - 2.0 * self._cel_w)
            cam_dir = safe_normalize(cam_dir * (1.0 - wgt) + to_goal * wgt)
        return pos, pos + cam_dir, vfov

    def auto_cam_enabled(self):
        return self._vis_auto_switch if self._auto_cam_key is None else self._auto_cam_key

    def _update_auto_spectate(self, state, interp_ratio):
        """Follow a stably closest active car when auto camera is on (A key, or the GUI's Vis auto switch)."""
        now = time.monotonic()
        if now >= self._vis_auto_next_poll:
            self._vis_auto_next_poll = now + 0.25
            path = os.environ.get("RSV_CONTROL_PATH", "")
            try:
                mtime = os.path.getmtime(path)
                if mtime != self._vis_auto_control_mtime:
                    with open(path, "r", encoding="utf-8") as f:
                        ctrl = json.load(f)
                    self._vis_auto_switch = bool(ctrl.get("vis_auto_switch", False))
                    self._vis_auto_control_mtime = mtime
            except (OSError, ValueError, TypeError):
                # A control.json replacement can be observed between unlink and rename. Keep the
                # last known setting and retry on the next short poll rather than affecting rendering.
                pass

        if not self.auto_cam_enabled():
            self._vis_auto_candidate = -1
            return

        # Kickoff (ball parked at centre field): follow a RANDOM car until the ball is first touched,
        # instead of whoever happens to be closest.
        bp = state.ball_state.next_pos
        # (a ball parked out of play during a goal celebration is NOT a kickoff: it sits at 0,0 too)
        at_kickoff = (abs(float(bp[0])) < 5 and abs(float(bp[1])) < 5 and 50 < float(bp[2]) < 100
                      and state.ball_state.next_vel.length < 1.0 and not getattr(state, "ball_hidden", False))
        if at_kickoff:
            if not getattr(self, "_kickoff_pick_done", False):
                active_idx = [i for i, cs in enumerate(state.car_states) if not cs.is_demoed]
                if active_idx:
                    self.spectate_idx = random.choice(active_idx)
                    self._car_cam_offset_smooth = None
                    self._car_cam_dir_smooth = None
                    self._kickoff_pick_done = True
            self._vis_auto_candidate = -1
            return
        self._kickoff_pick_done = False

        active = [(i, cs) for i, cs in enumerate(state.car_states) if not cs.is_demoed]
        if not active:
            return
        ball_pos = state.ball_state.get_pos(interp_ratio)
        closest_idx = min(active, key=lambda item: (item[1].phys.get_pos(interp_ratio) - ball_pos).length)[0]
        if closest_idx == self.spectate_idx:
            self._vis_auto_candidate = -1
            return

        if closest_idx != self._vis_auto_candidate:
            self._vis_auto_candidate = closest_idx
            self._vis_auto_candidate_since = now
            return

        # The new car has been closest continuously for 0.75 seconds: switch and snap the relative
        # camera smoothing state to that car. This is intentionally independent of the Space/P camera
        # controls: turning the GUI option off immediately restores fully manual target selection.
        if now - self._vis_auto_candidate_since >= 0.75:
            self.spectate_idx = closest_idx
            self._vis_auto_candidate = -1
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None

    def _ensure_render_target(self, width, height):
        """-> (render_w, render_h). The 3D scene ALWAYS renders into our own MSAA framebuffer (the
        window's framebuffer is single-sample): at the window size times RSV_RENDER_SCALE, reduced only
        if that exceeds RSV_MAX_RENDER_MP megapixels. paint() resolves it and blits it (bilinear) to the
        window, then draws the HUD straight onto the window at full native resolution."""
        scale, cap, samples = RENDER_SCALE, MAX_RENDER_MP, self.samples
        if self.window_mode:               # Edit Settings > Graphics
            res = self.config.gfx_resolution
            scale = {"150": 1.5, "200": 2.0}.get(res, 1.0)
            cap = MAX_RENDER_MP if res == "auto" else 1e9
            samples = min(int(self.config.gfx_aa), int(self.ctx.info.get("GL_MAX_SAMPLES", 8)))
        mp = width * height * scale * scale / 1e6
        if mp > cap:
            scale *= math.sqrt(cap / mp)
        scale *= (0.7, 0.85, 1.0)[max(0, min(2, int(getattr(self.config, "q_res", 2))))]   # Render resolution slider
        render_w, render_h = max(1, round(width * scale)), max(1, round(height * scale))
        if self._cap_size != (render_w, render_h, samples):
            if self.cap_fbo is not None:
                self.cap_fbo.release(); self.cap_resolve_fbo.release()
            s = samples if samples > 1 else 0
            self.ctx.multisample = s > 1
            self.cap_fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((render_w, render_h), samples=s)],
                depth_attachment=self.ctx.depth_renderbuffer((render_w, render_h), samples=s),
            )
            tex = self.ctx.texture((render_w, render_h), 3)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.cap_resolve_fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._cap_size = (render_w, render_h, samples)
        self.render_target = self.cap_fbo
        return render_w, render_h

    def paint(self, width, height, screen_fb, clip_capture=True):
        """Render one frame into screen_fb (the window's framebuffer, or an offscreen one headless)."""
        self.screen_fb = screen_fb
        render_w, render_h = self._ensure_render_target(width, height)
        self.render_target.use()
        self.ctx.viewport = (0, 0, render_w, render_h)

        cur_time = time.time()
        delta_time = cur_time - self.last_render_time

        _r0 = time.perf_counter()
        if self._gpu_query is not None:
            self._gpu_query.__enter__()
        state, interp_ratio, spectated = self.render(cur_time, delta_time, render_w, render_h)

        # MSAA resolve (same size), then scale into the window with the fullscreen-triangle blit.
        self.ctx.copy_framebuffer(self.cap_resolve_fbo, self.cap_fbo)
        screen_fb.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.cap_resolve_fbo.color_attachments[0].use(location=0)
        self._quad_prog['Tex'].value = 0
        self._quad_vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.BLEND)

        # HUD / overlays at full native resolution, straight onto the window framebuffer.
        self.render_target = screen_fb
        self._hud_off = 0
        _h0 = time.perf_counter() if PERF else 0.0
        self.render_hud(state, interp_ratio, spectated, cur_time, width, height)
        if PERF:
            self._perf_hud += time.perf_counter() - _h0
        if self._gpu_query is not None:
            self._gpu_query.__exit__(None, None, None)
            self._perf_gpu = getattr(self, "_perf_gpu", 0.0) + self._gpu_query.elapsed / 1e6
        _r1 = time.perf_counter()

        # Feed the finished frame (scene + HUD) into the rolling clip buffer, and honour a clip request
        # from the training GUI (the 'C' hotkey triggers the same save directly).
        if clip_capture:
            state_manager.record_cam(cur_time, int(self.spectate_idx), self.cam_manual)
            self.clip_recorder.capture(self.ctx, width, height, screen_fb)
            self.clip_recorder.poll_signal()
        _r2 = time.perf_counter()

        self.fps_counter += 1

        if int(cur_time) > int(self.last_render_time):
            self.last_fps = self.fps_counter
            self.fps_counter = 0

        if PERF:
            self._perf_render += (_r1 - _r0)
            self._perf_capture += (_r2 - _r1)
            self._perf_frames += 1
            span = cur_time - self._perf_t0
            if span >= 2.0:
                n = max(1, self._perf_frames)
                print("[perf] fps={:.0f} cpu_render={:.2f} gpu={:.2f} lock={:.2f} deepcopy={:.2f} pads={:.2f} ball={:.2f} cars={:.2f} arena={:.2f} fx={:.2f} hud={:.2f} capture={:.2f} (ms) scene={}x{}".format(
                    self._perf_frames / span,
                    self._perf_render / n * 1000.0,
                    getattr(self, "_perf_gpu", 0.0) / n,
                    self._perf_lock / n * 1000.0,
                    self._perf_deepcopy / n * 1000.0,
                    self._perf_pads / n * 1000.0,
                    self._perf_ball / n * 1000.0,
                    self._perf_cars / n * 1000.0,
                    self._perf_arena / n * 1000.0,
                    self._perf_tail / n * 1000.0,
                    self._perf_hud / n * 1000.0,
                    self._perf_capture / n * 1000.0, render_w, render_h), flush=True)
                self._perf_render = self._perf_capture = self._perf_deepcopy = 0.0
                self._perf_lock = self._perf_gpu = 0.0
                self._perf_pads = self._perf_cars = self._perf_arena = 0.0
                self._perf_ball = self._perf_tail = 0.0
                self._perf_lines = self._perf_hud = self._perf_pai = self._perf_ui = 0.0
                self._perf_frames = 0
                self._perf_t0 = cur_time

        self.last_render_time = cur_time

    SH_TILE = 160                    # shadow atlas tile size (px) -> 1.25 uu per texel over the 200 uu tile

    def _shadow_basis(self):
        """Light direction for the shadows: the sun's, with its elevation clamped (a low evening sun would otherwise
        stretch a car's shadow across half the field) + two axes perpendicular to it."""
        key = getattr(self, "map_name", "valley")
        hit = getattr(self, "_sh_basis_cache", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        sd = np.asarray(rl_maps.THEMES[key]["sun_dir"], "f8")
        h = float(np.hypot(sd[0], sd[1]))
        if h < 1e-6:
            L = np.array([0.0, 0.0, 1.0])
        else:
            L = np.array([sd[0], sd[1], max(sd[2], 0.62 * h)])
        L /= np.linalg.norm(L)
        U = np.cross(L, [0.0, 0.0, 1.0]) if abs(L[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
        U /= np.linalg.norm(U)
        V = np.cross(L, U)
        out = tuple(float(x) for x in L), tuple(float(x) for x in U), tuple(float(x) for x in V)
        self._sh_basis_cache = (key, out)
        return out

    def _render_shadow_atlas(self, casters, jobs):
        """casters: [(x, y, z, r)] (ball first when present), jobs: per caster [(vao, model_bytes)] -> atlas tiles."""
        q = max(0, min(2, int(getattr(self.config, "q_shadow", 2))))
        on = int(bool(getattr(self.config, "gfx_shadows", 1)))
        taps = (0, 2, 4)[q]
        if getattr(self, "_sh_state", None) != (on, taps):
            self._sh_state = (on, taps)
            for prog in (self.prog_rl_arena, self.prog_grass):
                if "shadowsOn" in prog:
                    prog["shadowsOn"].value = on
                if "shTaps" in prog:
                    prog["shTaps"].value = taps
        if not on:
            return
        tile = (96, 128, 192)[q]
        if tile != self.SH_TILE:
            self.SH_TILE = tile
            self._sh_tex.release()
            self._sh_tex = self.ctx.texture((3 * tile, 3 * tile), 2, dtype="f2")
            self._sh_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._sh_tex.repeat_x = False
            self._sh_tex.repeat_y = False
            self._sh_fbo.release()
            self._sh_fbo = self.ctx.framebuffer(color_attachments=[self._sh_tex],
                                                depth_attachment=self.ctx.depth_renderbuffer((3 * tile, 3 * tile)))
        L, U, V = self._shadow_basis()
        if getattr(self, "_sh_uniforms_for", None) != (L, U, V):
            self._sh_uniforms_for = (L, U, V)
            for prog in (self.prog_shadow, self.prog_rl_arena, self.prog_grass):
                for k, v in (("shL", L), ("shU", U), ("shV", V)):
                    if k in prog:
                        prog[k].value = v
            for prog in (self.prog_rl_arena, self.prog_grass):
                if "shadowAtlas" in prog:
                    prog["shadowAtlas"].value = 11
        fbo = self._sh_fbo
        fbo.use()
        fbo.clear(0.0, 1.0e4, 0.0, 0.0, depth=1.0)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        sp = self.prog_shadow
        T = self.SH_TILE
        for i, (c, parts) in enumerate(zip(casters, jobs)):
            fbo.viewport = ((i % 3) * T, (i // 3) * T, T, T)
            sp["cpos"].value = (float(c[0]), float(c[1]), float(c[2]))
            for vao, mb in parts:
                sp["m_model"].write(mb)
                vao.render(moderngl.TRIANGLES)
        fbo.viewport = (0, 0, 3 * T, 3 * T)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self._sh_tex.use(location=11)
        self.render_target.use()

    @staticmethod
    def _model_matrix(pos, forward, up, scale=1.0):
        fx_, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        # right = forward x up ; the matrix's 2nd column is -right (= left), as in render_model
        rx, ry, rz = fy * uz - fz * uy, fz * ux - fx_ * uz, fx_ * uy - fy * ux
        return np.array([
            fx_ * scale, fy * scale, fz * scale, 0,
            -rx * scale, -ry * scale, -rz * scale, 0,
            ux * scale, uy * scale, uz * scale, 0,
            float(pos[0]), float(pos[1]), float(pos[2]), 1,
        ], dtype="f4")

    def render(self, total_time, delta_time, width, height):
        """Draw the 3D scene into self.render_target. Returns (state, interp_ratio, spectated) for the
        HUD pass that paint() runs afterwards at native resolution."""
        # NOTE the split: _perf_lock is time spent WAITING for the socket-listener thread to release
        # the state mutex (plus rotate_with_ang_vel), _perf_deepcopy is the copy itself.
        _lk0 = time.perf_counter() if PERF else 0.0
        with global_state_mutex:
            state_manager.apply_due_packets(time.time())
            _gs = global_state_manager.state
            if not _gs.ball_state.has_rot:
                _r = (time.time() - _gs.recv_time) / max(_gs.recv_interval, 1e-6)
                _gs.ball_state.rotate_with_ang_vel(delta_time, _r)

            _dc0 = time.perf_counter() if PERF else 0.0
            state = copy.deepcopy(global_state_manager.state)
            if PERF:
                self._perf_deepcopy += time.perf_counter() - _dc0
                self._perf_lock += _dc0 - _lk0
        self.prev_state = state
        self.spectate_count = len(state.car_states)

        # Streaming tools can lock the camera to a specific car via the "spectate_idx" packet field
        # (e.g. view_credit.py aims the POV at the highest-credit player). Opt-in: only when sent.
        _fsi = getattr(state_manager, "forced_spectate_idx", None)
        if _fsi is not None and 0 <= int(_fsi) < self.spectate_count:
            if int(_fsi) != self.spectate_idx:
                self._car_cam_offset_smooth = None
                self._car_cam_dir_smooth = None
            self.spectate_idx = int(_fsi)

        # two supersonic trail ribbons per car (rear wheels)
        while len(self.car_ribbons) != 2 * len(state.car_states):
            if len(self.car_ribbons) < 2 * len(state.car_states):
                self.car_ribbons.append(RibbonEmitter())
            else:
                self.car_ribbons.pop()

        cur_time = time.time()
        interp_interval = max(state.recv_interval, 1e-6)
        interp_ratio = min(max((cur_time - state.recv_time) / interp_interval, 0), 1)

        self.ctx.clear(0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.cull_face = "back"
        self.ctx.front_face = "cw"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._aspect = width / max(1, height)
        camera_pos, camera_target_pos, camera_fov = self.calc_camera_state(state, interp_ratio, delta_time)
        # near 10 uu (not 1): 10x the depth precision far out -- distant trims no longer z-fight (flicker)
        proj = Matrix44.perspective_projection(camera_fov, -width/height, 10.0, 120 * 1000.0)
        lookat = Matrix44(fastvec.look_at(camera_pos, camera_target_pos, (0.0, 0.0, 1.0)))
        vp = (proj * lookat).astype('f4')
        vp_bytes = vp.tobytes()
        self._frame_vp, self._frame_cam = vp, camera_pos
        self.fx.cam_pos = (float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]))
        self.fx.set_quality(getattr(self.config, "q_particles", 2))
        cam_bytes = Vector3(camera_pos).astype('f4').tobytes()

        self.pr_camera_pos.write(cam_bytes)
        self.pr_m_vp.write(vp_bytes)
        self.pra_m_vp.write(vp_bytes)
        for prog in (self.prog_rl_arena, self.prog_car, self.prog_ball):
            prog["m_vp"].write(vp_bytes)
            prog["camPos"].write(cam_bytes)

        # Listener for 3D sound: "right" = the world direction that appears on the RIGHT of the screen
        # (the projection mirrors x, see the negative aspect above).
        fwd_v = safe_normalize(Vector3(camera_target_pos) - Vector3(camera_pos))
        scr_right = safe_normalize(fastvec.cross((0.0, 0.0, 1.0), fwd_v))
        self.audio.set_listener(camera_pos, scr_right)

        spectated = self.spectate_idx if 0 <= self.spectate_idx < len(state.car_states) else -1
        self._spectated_now = spectated

        tnow = float(time.time() % 3600.0)

        _p0 = time.perf_counter() if PERF else 0.0
        if not (state.boost_pad_states is None) and state.gamemode != "heatseeker": # Render boost pads
            self._ensure_pad_cache(state)
            states = state.boost_pad_states
            cache = self._pad_static
            n_p = min(len(states), len(cache))
            locs = state.boost_pad_locations
            now_p = time.time()
            if self._pad_prev is None or len(self._pad_prev) != n_p:
                self._pad_prev = [bool(x) for x in states[:n_p]]
                self._pad_pick_t = [None] * n_p
                self._pad_spawn_t = [None] * n_p
            for i in range(n_p):
                act = bool(states[i])
                if self._pad_prev[i] and not act:              # just picked up
                    self._pad_pick_t[i] = now_p
                    self.fx.pad_pickup((float(locs[i][0]), float(locs[i][1]), 0.0), bool(cache[i][0]))
                elif act and not self._pad_prev[i]:            # just respawned
                    self._pad_spawn_t[i] = now_p
                    lst = self.audio.listener
                    rp = (float(locs[i][0]), float(locs[i][1]), 40.0)
                    if (self._pad_pick_t[i] is not None and lst is not None      # RL: Play_Boost_Pickup_Respawn_End,
                            and math.dist(rp, lst[0]) < 3000.0):                # only for pads near the camera
                        self.audio.play("pad_respawn_end", rp, 0.9 / (1.0 + (math.dist(rp, lst[0]) / 1200.0) ** 2))
                self._pad_prev[i] = act
            pp = self.prog_pad
            pp["m_vp"].write(vp_bytes)
            pp["camPos"].write(cam_bytes)
            self.t_padgen.use(location=0)
            pp["Texture"].value = 0
            pp["ghost"].value = 0.0
            self.render_target.use()
            # Recharge (RL): the empty pad's base stays black for the first half, then whitens from the edge in
            # (PAD_FRAG `charge`), and in the last ~1 s the orb fades back in as a whitish glass sphere before it
            # pops back gold.
            ghosts, charge, glows, recharging = [], [], [], []
            pp["instanced"].value = 1
            inst = ([], [], [], [])                      # per mesh (2 * is_big + active): x, y, flash, pulse, charge, padR
            for i in range(n_p):
                is_big, mat = cache[i]
                x_, y_ = float(locs[i][0]), float(locs[i][1])
                if states[i]:
                    st_ = self._pad_spawn_t[i]
                    fl_ = max(0.0, 1.0 - (now_p - st_) / 0.15) if st_ is not None else 0.0
                    inst[2 * is_big + 1].append((x_, y_, fl_, math.sin(now_p * 3.0 + i * 1.7), -1.0, 0.0))
                    glows.append((x_, y_, is_big, True, 1.0, 0.0))
                else:
                    pt = self._pad_pick_t[i]
                    dur = self.PAD_RESPAWN_BIG if is_big else self.PAD_RESPAWN_SMALL
                    prog = 0.0 if pt is None else min(1.0, (now_p - pt) / dur)     # unknown start: stays black
                    inst[2 * is_big].append((x_, y_, 0.0, 0.0, prog, 37.8 if is_big else 20.8))
                    g_ = 0.0
                    if pt is not None:
                        g_win = 1.2 if is_big else 0.7
                        left = dur - (now_p - pt)
                        if left < g_win:
                            g_ = 1.0 - max(0.0, left) / g_win
                            ghosts.append((mat, is_big, g_))
                    glows.append((x_, y_, is_big, False, prog, g_))
                    if pt is not None:
                        recharging.append((i, (x_, y_, 40.0), prog))
            for k_, rows in enumerate(inst):
                if rows:
                    name = self._pad_vaos[k_]
                    self._pad_inst_buf[name].write(np.asarray(rows, "f4").tobytes())
                    self._pad_inst_vao[name].render(moderngl.TRIANGLES, instances=len(rows))
            pp["instanced"].value = 0
            self.fx.pad_glows(glows, now_p)
            self.audio.update_pad_respawn(recharging)
            self._pad_ghosts = ghosts            # translucent: drawn after the sky (see _render_pad_ghosts)
        if PERF:
            self._perf_pads += time.perf_counter() - _p0

        _b0 = time.perf_counter() if PERF else 0.0
        ball_phys = state.ball_state
        ball_pos = ball_phys.get_pos(interp_ratio)
        self.ctx.disable(moderngl.CULL_FACE)
        b_f, b_u = ball_phys.get_forward(interp_ratio), ball_phys.get_up(interp_ratio)
        self.prog_ball["m_model"].write(self._model_matrix(ball_pos, b_f, b_u).tobytes())
        self.prog_ball["blurRot"].value = self._ball_spin_blur(b_f, b_u, ball_phys.is_teleporting())
        bx_, by_, bz_ = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        in_goal = abs(bx_) < 893.0 and bz_ < 642.775 and abs(by_) > 5124.25 - 91.25
        if in_goal != getattr(self, "_ball_in_goal", None):
            self._ball_in_goal = in_goal
            self.prog_ball["inGoal"].value = 1.0 if in_goal else 0.0
        self.render_target.use()
        if not getattr(state, "ball_hidden", False):          # out of play during a goal celebration
            self.ball_vao.render(moderngl.TRIANGLES)
        self._update_ball_trail(state, ball_phys, ball_pos, interp_ratio, delta_time)
        if state.gamemode == "heatseeker":
            ball_speed = ball_phys.get_vel(interp_ratio).length
            self.ball_ribbon.update(ball_speed > 600, 0, ball_pos, Vector3((100, 0, 0)), 0.8, delta_time)
            if ball_phys.is_teleporting():
                self.ball_ribbon.points.clear()
            self.render_ribbon(self.ball_ribbon, camera_pos, 0.8, width=50, start_taper_time=0.08,
                               color=Vector4((1, 1, 1, 0.75)))
        if PERF:
            self._perf_ball += time.perf_counter() - _b0

        _c0 = time.perf_counter() if PERF else 0.0
        casters = [] if getattr(state, "ball_hidden", False) else [(ball_pos[0], ball_pos[1], ball_pos[2], 98.0)]
        sh_jobs = [] if not casters else [[(self._sh_ball_vao, self._model_matrix(ball_pos, b_f, b_u).tobytes())]]
        caster_fwd = [(0.0, 0.0)]
        caster_f3 = [(0.0, 0.0, 0.0)]
        caster_u3 = [(0.0, 0.0, 1.0)]
        boosting = {}
        now = time.time()
        for i, car_state in enumerate(state.car_states):
            if car_state.is_demoed:
                self.car_ribbons[2 * i].points.clear(); self.car_ribbons[2 * i + 1].points.clear()
                continue
            car_pos = car_state.phys.get_pos(interp_ratio)
            car_forward = car_state.phys.get_forward(interp_ratio)
            car_up = car_state.phys.get_up(interp_ratio)
            car_vel = car_state.phys.get_vel(interp_ratio)
            team = int(car_state.team_num) & 1

            self.fx.car_poses[i] = (car_pos, car_forward, car_up)      # flip-reset disc follows the car
            self._update_flip_streaks(i, car_pos, car_forward, car_up, delta_time, car_state.phys.is_teleporting(),
                                      bool(car_state.on_ground))
            car_model = self._model_matrix(car_pos, car_forward, car_up).tobytes()
            self.prog_car["m_model"].write(car_model)
            self.prog_car["bodyCol"].value = TEAM_BODY[team]
            self.prog_car["wheelGlow"].value = 0.0
            self.prog_car["glowCol"].value = (0.55, 0.9, 1.0)
            self.prog_car["brakeLight"].value = 0.0
            self.render_target.use()
            self.car_vao.render(moderngl.TRIANGLES)
            # wheels: suspension travel + spin + front steering (carrig.py)
            job = [(self._sh_car_vao, car_model)]
            for k_, (wv, wm) in enumerate(zip(self.wheel_vaos, self.wheel_rig.matrices(
                    i, car_pos, car_forward, car_up, car_vel, bool(car_state.on_ground), delta_time))):
                self.prog_car["m_model"].write(wm)
                wv.render(moderngl.TRIANGLES)
                job.append((self._sh_wheel_vaos[k_], wm))

            if len(casters) < 9:
                f2 = Vector3((car_forward[0], car_forward[1], 0.0))
                f2 = f2 / max(f2.length, 1e-4)
                casters.append((car_pos[0], car_pos[1], car_pos[2], 72.0))
                sh_jobs.append(job)
                caster_fwd.append((f2[0], f2[1]))
                caster_f3.append((float(car_forward[0]), float(car_forward[1]), float(car_forward[2])))
                caster_u3.append((float(car_up[0]), float(car_up[1]), float(car_up[2])))

            if car_state.is_boosting:
                self.fx.boost(i, tuple(car_pos), tuple(car_forward), tuple(car_up), tuple(car_vel), team, delta_time)
                self._boost_last[i] = cur_time
            else:
                self.fx.boost_end(i)
            # SOUND hold: bots feather boost (0.1 s presses, ~0.1 s gaps); treat presses < 0.35 s apart
            # as one boost so the Alpha ignition + tail don't retrigger twice a second ("bubbling").
            # The flame above stays exact.
            if cur_time - self._boost_last.get(i, -1e9) <= self.BOOST_SOUND_HOLD:
                boosting[i] = (tuple(car_pos), i == spectated, float(car_vel.length))

            # supersonic trails from the rear wheels (RL "classic" trail) -- batched in fx.render
            # RL only draws the supersonic trail while the wheels are on a surface
            supersonic = car_vel.length >= self.SUPERSONIC_SPEED and bool(car_state.on_ground)
            if supersonic or self.car_ribbons[2 * i].points or self.car_ribbons[2 * i + 1].points:
                right = fastvec.cross(car_forward, car_up)
                for k, side in enumerate((-1.0, 1.0)):
                    rib = self.car_ribbons[2 * i + k]
                    emit = car_pos - car_forward * 36.0 + right * (side * 29.0) - car_up * 11.0
                    rib.update(supersonic, 0, emit, Vector3((0.0, 0.0, 0.0)), self.SS_TRAIL_LIFE, delta_time)
                    if car_state.phys.is_teleporting():
                        rib.points.clear()
                    if len(rib.points) > 1:
                        # RL's supersonic trail (the same for both teams): off each rear wheel a SHORT, thin, glowing
                        # violet streak -- hot pink-white where it leaves the wheel, deep violet -> blue as it thins
                        # out about a car length and a half behind -- with tiny bluish-white sparkles along it.
                        L_ = self.SS_TRAIL_LIFE
                        self.fx.add_tube(rib, L_, 6.5, (0.78, 0.32, 1.0, 1.0), white_from=0.0, white_len=28.0)
                        self.fx.sparkle((i, k), tuple(emit), delta_time)
        self.ctx.enable(moderngl.CULL_FACE)
        self.audio.update_boost(boosting)
        eng = None
        if spectated >= 0 and not state.car_states[spectated].is_demoed:
            sc_ = state.car_states[spectated]
            v_ = sc_.phys.get_vel(interp_ratio)
            f_ = sc_.phys.get_forward(interp_ratio)
            ctl = sc_.controls if getattr(sc_, "has_controls", False) else None
            eng = {"pos": tuple(sc_.phys.get_pos(interp_ratio)), "dt": delta_time,
                   "v_fwd": float(v_[0] * f_[0] + v_[1] * f_[1] + v_[2] * f_[2]), "on_ground": bool(sc_.on_ground),
                   "throttle": None if ctl is None else float(ctl.throttle),
                   "steer": None if ctl is None else float(ctl.steer), "boosting": spectated in boosting,
                   "speed": float(v_.length)}
        self.audio.update_engine(eng)
        if PERF:
            self._perf_cars += time.perf_counter() - _c0

        ###########################################

        _a0 = time.perf_counter() if PERF else 0.0
        cst = np.zeros((9, 4), "f4"); cfw = np.zeros((9, 2), "f4")
        cst[:len(casters)] = casters; cfw[:len(caster_fwd)] = caster_fwd
        cf3 = np.zeros((9, 3), "f4"); cu3 = np.zeros((9, 3), "f4"); cu3[:, 2] = 1.0
        cf3[:len(caster_f3)] = caster_f3; cu3[:len(caster_u3)] = caster_u3
        self.prog_rl_arena["casters"].write(cst.tobytes())
        if "casterFwd" in self.prog_rl_arena:          # (the arena shader may not use it)
            self.prog_rl_arena["casterFwd"].write(cfw.tobytes())
        self.prog_rl_arena["nCasters"].value = len(casters)
        self._render_shadow_atlas(casters, sh_jobs)
        ball_mark = self._ball_mark(state, ball_pos)
        self.prog_rl_arena["ballMark"].value = ball_mark
        self._update_sky_cube(tnow)
        self.render_target.use()
        self.prog_rl_arena["detailBias"].value = 2.0 if self.config.gfx_detail == "sharp" else 1.0
        self._blade_tex.use(location=5)
        self._grain_tex.use(location=7)
        if self._space_cube is not None:
            self._space_cube.use(location=6)
        if "arena" not in _SKIP:
            self._arena_parts[0].render(moderngl.TRIANGLES)
        self._render_grass(vp, camera_pos, tnow, cst, cfw, cf3, cu3, len(casters), ball_mark)
        # Draw order is for overdraw: everything opaque first, so the depth test rejects the hidden
        # parts of the stadium and the sky only shades the pixels nothing else covered.
        self.ctx.disable(moderngl.CULL_FACE)
        self._render_scenery(vp_bytes, cam_bytes, tnow)
        self.ctx.fbo.depth_mask = False
        self.prog_sky["invVP"].write(_inv_view_proj(proj, lookat))
        self.prog_sky["time"].value = tnow
        if "sky" not in _SKIP:
            # sky at depth 1.0 with <=: only where nothing was drawn. (It used to sit at 0.9999 with <,
            # which overwrote everything farther than ~10 000 uu -- the "stadium vanishes far away" bug.)
            self.ctx.depth_func = "<="
            self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
            self.ctx.depth_func = "<"
        self.ctx.fbo.depth_mask = True
        self.ctx.enable(moderngl.CULL_FACE)
        self._render_pad_ghosts(vp_bytes, cam_bytes)
        # translucent glass walls + ceiling (premultiplied alpha, no depth write) so the stadium,
        # skyline and anything behind a wall show through like in game
        self.ctx.fbo.depth_mask = False
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        if "walls" not in _SKIP:
            self._arena_parts[1].render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.fbo.depth_mask = True
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if PERF:
            self._perf_arena += time.perf_counter() - _a0
        _t0 = time.perf_counter() if PERF else 0.0

        ###########################################

        if len(state.render_state.lines) > 0:
            vertices_flat = np.array(state.render_state.lines).flatten()
            num_verts = vertices_flat.shape[0] // 3
            self.lines_vbo.write(vertices_flat.astype('f4'))
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.render_model(
                None, None, None,
                "render_lines", self.t_boost_glow, 1, Vector4((1, 1, 1, 1)),
                mode=moderngl.LINES,
                vert_amount=num_verts
            )
            self.ctx.enable(moderngl.DEPTH_TEST)

        # ---- game events -> sounds + effects, fired when interpolation reaches their packet ----
        self._drain_events(state, spectated)
        self.fx.update(delta_time)
        self.render_target.use()
        px_scale = height / (2.0 * math.tan(math.radians(camera_fov) / 2.0))
        cam_f = safe_normalize(Vector3(camera_target_pos) - Vector3(camera_pos))
        cam_r = safe_normalize(fastvec.cross(cam_f, (0.0, 0.0, 1.0)))
        self.fx.cam_right = np.asarray(cam_r, "f4")
        self.fx.cam_up = np.asarray(fastvec.cross(cam_r, cam_f), "f4")
        self._update_speed_lines(state, interp_ratio, spectated)
        if self.map_name == "paris" and rl_maps.THEMES["paris"].get("night", 1.0) > 0.5:
            top = np.array([0.0, 28500.0, 25200.0], "f4")
            for k in range(2):
                a = tnow * 0.35 + k * math.pi
                d = np.array([math.cos(a) * 0.75, math.sin(a) * 0.75, 0.45], "f4")
                d /= np.linalg.norm(d)
                self.fx.add_beam(tuple(top), tuple(top + d * 45000.0), 60.0, 3200.0, (0.62, 0.78, 1.0, 0.16),
                                 (0.62, 0.78, 1.0, 0.0))
        if self.map_name == "space":                   # the cruiser's engine exhaust plumes (additive, flickering)
            for ex, ez, r, ln in ((7200.0, 2600.0, 1250.0, 9000.0), (-7200.0, 2600.0, 1250.0, 9000.0),
                                  (0.0, -1700.0, 1500.0, 11000.0), (4600.0, -1500.0, 1100.0, 8000.0), (-4600.0, -1500.0, 1100.0, 8000.0)):
                fl = 0.85 + 0.15 * math.sin(tnow * 23.0 + ex)
                y0 = -24420.0 if abs(ex) > 7000 else -23200.0
                self.fx.add_beam((ex, y0, ez), (ex, y0 - ln * 0.55 * fl, ez), r * 1.1, r * 0.2,
                                 (0.75, 0.92, 1.0, 0.35 * fl), (0.4, 0.6, 1.0, 0.0))            # hot core
                self.fx.add_beam((ex, y0, ez), (ex, y0 - ln * fl, ez), r * 2.2, r * 0.8,
                                 (0.35, 0.6, 1.0, 0.12 * fl), (0.2, 0.35, 1.0, 0.0))            # soft outer glow
        self.fx.render(vp_bytes, px_scale, camera_pos)

        if PERF:
            self._perf_tail += time.perf_counter() - _t0

        self.prev_interp_ratio = interp_ratio
        return state, interp_ratio, spectated

    # ---- events -------------------------------------------------------------------------------- #

    def _drain_events(self, state, spectated):
        q = rl_events.event_queue
        now = time.time()
        delay = min(max(state.recv_interval, 0.0), 0.12)
        # Rendering was paused (window unfocused / minimised) while the sender kept going: the events of the pause
        # must not all fire in this one frame -- up to 0.6 s of hits, bounces and boosts stacked into one loud burst.
        last, self._drain_t = getattr(self, "_drain_t", now), now
        if now - last > 0.25:
            while q and q[0]["t"] < now - 0.1:
                q.popleft()
        while q and q[0]["t"] + delay <= now:
            ev = q.popleft()
            if now - ev["t"] > 0.6:
                continue                              # stale (window was inactive): drop silently
            try:
                self._handle_event(ev, spectated)
            except Exception as e:
                print("[events] handler error: {!r}".format(e))

    # Impact sounds (like RL): too-soft contacts are skipped, and the same sound repeating within its window only
    # plays again if clearly harder than the last one -- so a ball rubbing a post, a car grinding a wall or two
    # cars pushing each other don't machine-gun the sound. key -> (min strength, window s).
    SOUND_GATES = {"ball_hit": (130.0, 0.15), "post": (350.0, 0.50), "bounce": (300.0, 0.20),
                   "bump": (150.0, 0.40), "body": (300.0, 0.40), "land": (140.0, 0.25)}

    def _sound_gate(self, kind, key, strength, harder=1.6):
        min_s, window = self.SOUND_GATES[kind]
        now = time.time()
        gates = self.__dict__.setdefault("_snd_gates", {})
        last_t, last_s = gates.get(key, (-1e9, 0.0))
        if strength < min_s or (now - last_t < window and strength < last_s * harder):
            return False
        gates[key] = (now, strength)
        return True

    # RL ball trail: a ribbon following the ball centre in the colour of the team that touched it last, only
    # while the ball moves faster than 75 kph; a round tube, each point fading out over 600 ms, white for the
    # first 50 uu past the ball's surface. Once the ball slows under 75 kph the trail keeps being emitted for
    # 600 ms with a strength going 1 -> 0 (a reverse fade), so it tapers off instead of stopping dead.
    BALL_TRAIL_SPEED = 82.0 / 0.036                           # 82 kph in uu/s (1 uu = 1 cm)
    BALL_TRAIL_TAIL = 0.60           # after the ball drops under 82 kph it keeps emitting, fainter and fainter
    BALL_TRAIL_LIFE = 1.00
    BALL_TRAIL_ARM = 0.10            # s over the threshold, uninterrupted, before the trail starts
    BALL_TRAIL_GRACE = 0.12          # s below it that don't count as an interruption
    BALL_TRAIL_RADIUS = 21.0
    # 70% of the previous saturation ((0.30, 0.45, 1.0) / (1.0, 0.52, 0.12)): 70% of the way from grey, same luminance
    TEAM_TRAIL = ((0.347, 0.452, 0.837), (0.878, 0.542, 0.262))

    # Flip streaks: very faint, very thin white lines left by the car's four upper corners while it flips.
    FLIP_STREAK_EMIT = 0.60          # a dodge's rotation lasts ~0.6 s
    FLIP_STREAK_LIFE = 0.40          # each point fully faded after 400 ms

    def _update_flip_streaks(self, i, car_pos, car_forward, car_up, delta_time, teleported, on_surface):
        ribs = self._corner_ribs.get(i)
        # airborne flips only: nothing while the wheels touch the floor / a wall (wavedash, wall dash)
        # A landing ends the flip (a wavedash): a jump right after it must not draw streaks for the rest of
        # the 0.6 s window. "Landed" = wheels down AFTER the car was seen airborne in this flip (the rendered
        # state lags the event a little, so the take-off frames can still read on_ground).
        if time.time() < self._flip_until.get(i, 0.0):
            seen = self.__dict__.setdefault("_flip_airseen", {})
            if not on_surface:
                seen[i] = True
            elif seen.get(i):
                self._flip_until[i] = 0.0
        flipping = time.time() < self._flip_until.get(i, 0.0) and not on_surface
        if ribs is None:
            if not flipping:
                return
            ribs = self._corner_ribs[i] = [RibbonEmitter() for _ in self.car_streak_points]
        fx_, fy_, fz_ = float(car_forward[0]), float(car_forward[1]), float(car_forward[2])
        ux_, uy_, uz_ = float(car_up[0]), float(car_up[1]), float(car_up[2])
        lx_, ly_, lz_ = uy_ * fz_ - uz_ * fy_, uz_ * fx_ - ux_ * fz_, ux_ * fy_ - uy_ * fx_
        px_, py_, pz_ = float(car_pos[0]), float(car_pos[1]), float(car_pos[2])
        zero = (0.0, 0.0, 0.0)
        for rib, (cx, cy, cz) in zip(ribs, self.car_streak_points):
            p = (px_ + fx_ * cx + lx_ * cy + ux_ * cz, py_ + fy_ * cx + ly_ * cy + uy_ * cz,
                 pz_ + fz_ * cx + lz_ * cy + uz_ * cz)
            rib.update(flipping, 0, p, zero, self.FLIP_STREAK_LIFE, delta_time)
            if teleported:
                rib.points.clear()
            if len(rib.points) > 1:
                self.fx.add_trail(rib, self.FLIP_STREAK_LIFE, 0.9, (1.0, 1.0, 1.0, 0.12))
        if not flipping and not any(r.points for r in ribs):
            del self._corner_ribs[i]

    def _update_ball_trail(self, state, ball_phys, ball_pos, interp_ratio, delta_time):
        if not int(getattr(self.config, "gfx_ball_trail", 1)):
            if self.ball_trail.points:
                self.ball_trail.points.clear()
            self._ball_trail_on = False
            return
        team = rl_events.g_detector.last_touch_team
        hidden = getattr(state, "ball_hidden", False)
        usable = not hidden and team is not None and state.gamemode != "heatseeker"
        now = time.time()
        # Only after the ball has been over the threshold for BALL_TRAIL_ARM s in a row (a 50/50 spikes the speed for
        # a packet and the ball goes nowhere). Dips shorter than BALL_TRAIL_GRACE (a bounce, the blend between two
        # packets) don't reset the timer. The packet velocity, not the interpolated one (which dips mid-bounce).
        fast = usable and max(ball_phys.prev_vel.length, ball_phys.next_vel.length) > self.BALL_TRAIL_SPEED
        if fast:
            if getattr(self, "_trail_arm_t", None) is None:
                self._trail_arm_t = now
            self._trail_seen_t = now
        elif now - getattr(self, "_trail_seen_t", -1e9) > self.BALL_TRAIL_GRACE:
            self._trail_arm_t = None
        if fast and now - self._trail_arm_t >= self.BALL_TRAIL_ARM:
            self._trail_fast_t = now
        since = now - getattr(self, "_trail_fast_t", -1e9)
        k = max(0.0, 1.0 - since / self.BALL_TRAIL_TAIL) if usable else 0.0
        on = k > 0.0
        restart = on and not self._ball_trail_on
        if restart and self.ball_trail.points:
            self.ball_trail.points.clear()                     # restart: never bridge across the gap
        self._ball_trail_on = on
        self.ball_trail.update(on, 0, Vector3(ball_pos), Vector3((0.0, 0.0, 0.0)), self.BALL_TRAIL_LIFE, delta_time)
        # Touch-aware colour: each point keeps the colour it was emitted with; after a touch by the other team
        # the NEW points blend to its colour over ~0.1 s (the trail already drawn doesn't change retroactively).
        tt = getattr(self, "_trail_touch_team", None)
        tt = (int(team) & 1) if tt is None and team is not None else tt
        target = np.asarray(self.TEAM_TRAIL[tt if tt is not None else 0], "f4")
        col = getattr(self, "_trail_col", None)
        if col is None or restart or not self.ball_trail.points:
            col = target.copy()
        else:
            col = col + (target - col) * (1.0 - math.exp(-max(delta_time, 0.0) / 0.035))
        self._trail_col = col
        if on and self.ball_trail.points:
            self.ball_trail.points[0].k = k                    # newest point: emission strength
            self.ball_trail.points[0].col = (float(col[0]), float(col[1]), float(col[2]))
        if hidden or ball_phys.is_teleporting():
            self.ball_trail.points.clear()
        if len(self.ball_trail.points) > 1 and team is not None:
            self._trail_team = int(team) & 1
        if len(self.ball_trail.points) > 1:
            self.fx.add_tube3d(self.ball_trail, self.BALL_TRAIL_LIFE, self.BALL_TRAIL_RADIUS,
                             (*[float(c) for c in self._trail_col], 0.55),
                             white_from=91.25, white_len=50.0)

    def _handle_event(self, ev, spectated):
        a = self.audio
        k = ev["kind"]
        pos = ev["pos"]
        car = ev.get("car", -1)
        local = car == spectated and car >= 0
        sfx = "_local" if local else "_other"
        stall = bool(ev.get("stall"))              # a stall: flip input, no impulse and no rotation
        if k in ("jump", "doublejump") and car >= 0:
            # a jump that is not a flip (e.g. right after a wavedash, even before the landing was seen) ends any
            # flip streaks still pending for that car
            self._flip_until[car] = 0.0
        if k == "dodge" and car >= 0 and not stall:
            self._flip_until[car] = time.time() + self.FLIP_STREAK_EMIT
            self.__dict__.setdefault("_flip_airseen", {})[car] = False
        if k in ("jump", "doublejump", "dodge"):
            a.play(k + sfx, pos, 0.9, local)
            if not stall:
                self.fx.on_event(ev, spectated)            # the jump glow (jumps, double jumps, flips)
        elif k == "flipreset":
            a.play("flipreset" + sfx, pos, 1.0 if local else 0.85, local)
            self.fx.on_event(ev, spectated)
        elif k == "land":
            if not self._sound_gate("land", ("land", car), ev.get("strength", 0.0)):
                return
            g = min(1.0, max(0.25, (ev.get("strength", 0.0) - 140.0) / 900.0))
            a.play("land" + sfx, pos, g, local)
        elif k == "ball_hit":
            if ev.get("team") is not None:
                self._trail_touch_team = int(ev["team"]) & 1   # trail colour follows the touch as it is SHOWN
            # RL doesn't re-trigger the hit sound on every tiny dribble contact: soft touches are
            # skipped, and hits closer than 0.15 s play only if clearly harder than the last one.
            if not self._sound_gate("ball_hit", "ball_hit", ev.get("strength", 0.0)):
                return
            d = a.distance(pos)
            band = "close" if d < 1500 else ("mid" if d < 4000 else "far")
            g = min(1.0, 0.45 + ev.get("strength", 0.0) / 2500.0)
            a.play("ball_hit_" + ("close" if local else band), pos, g, local)
            cp = ev.get("contact")
            if cp is not None and not ev.get("underside", True):     # body touch (not the wheels): sparks
                self.fx.sparks(cp, np.subtract(cp, pos), ev.get("strength", 0.0))
        elif k == "ball_bounce":
            d = a.distance(pos)
            band = "close" if d < 1500 else ("mid" if d < 4000 else "far")
            s_ = ev.get("strength", 0.0)
            surf = ev.get("surface")
            kind = "floor" if surf == "floor" else ("post" if surf == "post" else "wall")
            if kind == "post":
                if not self._sound_gate("post", "post", s_):
                    return
                a.play("ball_post_{}".format(band), pos, min(1.0, max(0.2, s_ / 1500.0)))
                # the metallic ring on its own: in the premix it is buried under the impact thud
                a.play("ball_post_ping", pos, min(1.0, max(0.5, s_ / 1200.0)))
            else:
                if not self._sound_gate("bounce", ("bounce", kind), s_):
                    return
                a.play("ball_bounce_{}_{}".format(kind, band), pos, min(1.0, max(0.15, s_ / 1800.0)))
        elif k == "bump":
            s = ev.get("strength", 0.0)
            if not self._sound_gate("bump", ("bump",) + tuple(sorted((car, ev.get("other", -2)))), s):
                return
            stage = 0 if s < 500 else (1 if s < 900 else (2 if s < 1400 else 3))
            involved = spectated in (car, ev.get("other", -2))
            a.play("bump_{}".format(stage), pos, min(1.0, max(0.35, s / 1400.0)), involved)
            if ev.get("spark") and ev.get("contact") is not None:
                self.fx.sparks(ev["contact"], ev.get("normal", (0, 0, 1)), s)
        elif k == "demo":
            a.play("demo", pos, 1.0)
            a.play("demo_small" + sfx, pos, 0.8, local)          # the demolished car's own layer
            if spectated >= 0 and ev.get("by", -1) == spectated:
                a.play("demolish_stinger", None, 0.8, True)     # RL's "Demolition" stat jingle for YOUR demo
            self.fx.on_event(ev, spectated)
        elif k == "body":
            # car body (roof / side / nose) into the floor, a wall or the ceiling
            if not self._sound_gate("body", ("body", car), ev.get("strength", 0.0)):
                return
            g = min(1.0, max(0.25, (ev.get("strength", 0.0) - 200.0) / 1200.0))
            a.play("body" + sfx, pos, g, local)
            self.fx.sparks(pos, ev.get("normal", (0, 0, 1)), ev.get("strength", 0.0))
        elif k == "pad":
            if local:                                  # RL only plays pickups for your own car
                # big pad = Play_Boost_Pickup_Pad_Local, small pad = Play_Boost_Pickup_Pill_Local
                a.play("pad_pickup" if ev.get("big") else "pad_pickup_small", pos, 0.9, True)
        elif k == "save":
            self._cheer_t = [time.time(), time.time()]     # a save: both teams' fans jump on their seats
        elif k == "goal":
            self._cheer_t[int(ev.get("team", 0)) & 1] = time.time()   # a goal: the scoring team's fans
            self._goal_banner = (int(ev.get("team", 0)) & 1, time.time())
            a.play("goal_explosion_default", pos, 1.0, True)
            a.play("goal_explosion", pos, 0.8, True)             # the goal event layer on top
            a.play("goal_horn", None, 0.8, True)
            self.fx.on_event(ev, spectated)
        elif k == "supersonic":
            if local:
                a.play("supersonic", pos, 0.7, True)

    # ---- HUD pass (native resolution, on the window framebuffer) ------------------------------- #

    def render_hud(self, state, interp_ratio, spectated, total_time, width, height):
        ss_target = 0.0
        foe = None
        if spectated >= 0:
            spectated_car = state.car_states[spectated]
            if not spectated_car.is_demoed:
                vel_ = spectated_car.phys.get_vel(interp_ratio)
                speed = vel_.length
                if speed >= self.SUPERSONIC_SPEED:
                    ss_target = 1.0
                    foe = self._heading_point(vel_, width, height)
                self.render_boost_hud(width, height, spectated_car.boost_amount, spectated_car.team_num)
        self.render_pai_hud(width, height, getattr(state_manager, "gail_hud", None))
        self.render_scoreboard_hud(width, height, getattr(state_manager, "scoreboard", None))
        if self.show_phys_hud and self.window_mode:   # live window only: clips (offline render) never carry the physics overlay
            if self._phys_hud is None:
                from phys_hud import PhysHud
                self._phys_hud = PhysHud(self.ctx)
            self._phys_hud.render(self.render_target, state, interp_ratio, spectated, width, height)
        self.render_goal_banner(width, height, state)
        self.render_clip_indicator(width, height)

        v = self.config.volume.val / 100.0          # settings-panel volume slider drives the mixer
        for name in self.config.SOUND_MIX:          # per-category sliders
            self.audio.cat[name[4:]] = getattr(self.config, name).val / 100.0
        if abs(v - self.audio.volume) > 1e-4:
            self.audio.volume = v

        ui_text = ""
        ui_text += "Render FPS: {}".format(self.last_fps) + "\n"
        ui_text += "Map: {} (arrow keys)".format(rl_maps.TITLE.get(self.map_name, self.map_name)) + "\n"
        ui_text += "GPU: {}".format(self.gpu_name) + "\n"
        ui_text += "Connected: {}".format(state.recv_time > 0) + "\n"
        if state.recv_interval > 0:
            ui_text += "Network rate: {:.2f}fps".format(1 / state.recv_interval) + "\n"
        ui_text += "Ball speed: {:.2f}kph".format(state.ball_state.prev_vel.length * (9 / 250)) + "\n"
        self._poll_feed_answer()
        tm = getattr(self, "_toast_msg", None)
        if tm is not None and time.time() - tm[1] < 3.0 and tm[0]:
            ui_text += ">> " + tm[0] + "\n"
        ui_text += "Camera: {}  [A] auto  [P] closest  [Space] ball cam".format(
            "auto" if self.auto_cam_enabled() else "manual") + "\n"
        if self.audio.ok:
            ui_text += "Sound: {} ({:.0f}%)  [M] mute  [H] hide panel".format(
                "muted" if self.audio.muted else "on", self.audio.volume * 100) + "\n"
        try:
            from pose_editor import g_pose_editor
            ui_text += g_pose_editor.hud_text()
            ui_text += getattr(state_manager, "hud_text", "")   # PLAY driver overlay (reward/record)
        except Exception:
            pass
        u = get_ui()
        if u is not None:
            u.set_text(ui_text)
        self.last_ui_text = ui_text

    def render_flipreset_flash(self, width, height):
        """RL v2.66 flip-reset indicator, screen part: blue light blades sweep in from the four corners
        for ~0.45 s when the SPECTATED car gets a reset (the 3D shockwave is in fx.py)."""
        age = time.time() - self.fx.screen_flash
        if age > 0.45:
            return
        k = 1.0 - age / 0.45
        ortho = Matrix44.orthogonal_projection(0.0, width, height, 0.0, -1.0, 1.0)
        self._hud_ortho = ortho
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        m = min(width, height)
        L = m * (0.30 + 0.25 * (1.0 - k))                # blades grow inward while fading
        tris = []
        for cx, cy, sx, sy in ((0, 0, 1, 1), (width, 0, -1, 1), (0, height, 1, -1), (width, height, -1, -1)):
            for off, wid in ((-0.30, 0.020), (0.0, 0.035), (0.30, 0.020)):
                a = math.atan2(sy, sx) + off * 0.5
                dx, dy = math.cos(a), math.sin(a)
                px, py = -dy, dx
                w = m * wid
                ex, ey = cx + dx * L, cy + dy * L
                tris += [(cx + px * w, cy + py * w, 0.0, 0.35, 0.75, 1.0, 0.85 * k),
                         (cx - px * w, cy - py * w, 0.0, 0.35, 0.75, 1.0, 0.85 * k),
                         (ex, ey, 0.0, 0.8, 0.95, 1.0, 0.0)]
        arr = np.asarray(tris, "f4")
        self.hud_c_vbo.write(arr.tobytes(), 0)
        self._hud_c_prog['m_vp'].write(ortho.astype('f4'))
        self.render_target.use()
        self.hud_c_vao.render(moderngl.TRIANGLES, vertices=len(arr))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    # ---- input (called by the Qt widget) ------------------------------------------------------- #

    def handle_mouse_press(self, event):
        """-> True if the view changed (the widget repaints immediately)."""
        # While a play/pose tool is capturing input, RocketSimVis's own camera controls stand down.
        if state_manager.is_input_captured():
            return False
        if event.button() == Qt.LeftButton:
            if self.spectate_count == 0:
                return False
            self.spectate_idx += 1
            if self.spectate_idx >= self.spectate_count:
                self.spectate_idx = -1
            self._car_cam_offset_smooth = None
            self._car_cam_dir_smooth = None
            return True
        return False

    FEED_PORT = 9275     # the sender's side channel (focus heartbeats + these requests), see networking-format.md

    def _send_feed_request(self, msg, label):
        """Visualizer key -> the game sender: b"view:<n>" (team size) or b"det:<0|1>" (bot actions). The sender
        decides (e.g. only team sizes its bot's observation supports) and answers with "vis_msg" in a packet."""
        try:
            if getattr(self, "_feed_sock", None) is None:
                import socket as _s
                self._feed_sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            self._feed_sock.sendto(msg, ("127.0.0.1", self.FEED_PORT))
            self._feed_wait = time.time()
            self._toast("Requested " + label + "...")
        except OSError as e:
            self._toast("Could not reach the sender: {}".format(e))

    def _toast(self, msg):
        self._toast_msg = (msg, time.time())

    def _poll_feed_answer(self):
        vm = getattr(state_manager, "vis_msg", None)
        w = getattr(self, "_feed_wait", None)
        if vm is not None and w is not None and vm[1] >= w:
            self._feed_wait = None
            self._toast(vm[0])
        elif w is not None and time.time() - w > 1.5:
            self._feed_wait = None
            self._toast("No answer: this sender doesn't handle visualizer keys")

    def handle_key_press(self, event):
        # Ball-cam toggle (Space) — like Rocket League. Ignore auto-repeat so a hold = one toggle.
        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                cur = getattr(self, "cam_manual", None)
                self.cam_manual = False if cur is None else (not cur)
            return

        # Auto camera: follow whichever player has been closest to the ball (kickoff: a random one)
        if event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self._auto_cam_key = not self.auto_cam_enabled()
                self._vis_auto_candidate = -1
                self._kickoff_pick_done = False
            return

        # Team size of the spectated arena: '&' / 'e-acute' / '"' (AZERTY number row) or 1 / 2 / 3.
        # Bot actions: S = stochastic, D = deterministic (not while a pose editor / play tool owns ZQSD).
        # Sent to the game sender (UDP side channel), which decides and answers (see networking-format.md).
        size = {Qt.Key_Ampersand: 1, Qt.Key_1: 1, Qt.Key_Eacute: 2, Qt.Key_2: 2,
                Qt.Key_QuoteDbl: 3, Qt.Key_3: 3}.get(event.key())
        if size is not None and not event.isAutoRepeat():
            self._send_feed_request(b"view:%d" % size, "%dv%d" % (size, size))
            return
        if event.key() in (Qt.Key_S, Qt.Key_D) and not event.isAutoRepeat() and not (
                getattr(state_manager, "edit_mode", False) or state_manager.is_pose_edit_allowed()
                or state_manager.is_input_captured()):
            det = event.key() == Qt.Key_D
            self._send_feed_request(b"det:1" if det else b"det:0", "deterministic" if det else "stochastic")
            return

        # Save a clip of the last 12 gameplay seconds to mp4
        if event.key() == Qt.Key_C:
            self.clip_recorder.save_clip()
            return

        # Sound: M = mute toggle, [ / ] = volume down / up (persisted)
        if event.key() == Qt.Key_M and not event.isAutoRepeat():
            self.audio.toggle_mute()
            return
        if event.key() in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            d = 10.0 if event.key() == Qt.Key_BracketRight else -10.0
            self.config.volume.val = min(100.0, max(0.0, self.config.volume.val + d))
            self.config.save()
            return

        # Switch to player closest to ball
        if event.key() in self.MAP_KEYS and not event.isAutoRepeat():
            name = self.MAP_KEYS[event.key()]
            if name != self.map_name:
                self.set_map(name)
            return
        if event.key() == Qt.Key_P:
            if not (self.prev_state is None):
                closest_idx = -1
                closest_dist = 100_000
                for i in range(len(self.prev_state.car_states)):
                    player = self.prev_state.car_states[i] # type: CarState
                    dist_to_ball = (player.phys.next_pos - self.prev_state.ball_state.next_pos).length
                    if dist_to_ball < closest_dist:
                        closest_idx = i
                        closest_dist = dist_to_ball

                self.spectate_idx = closest_idx


def _is_foreground(window):
    """True only if the vis really is the OS foreground window. Qt's isActiveWindow() can already be True for a
    window Windows refused to bring forward (launched from the GUI: focus-stealing protection), which made the
    vis render + play sound + un-pause the feed behind the GUI at startup."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        u32 = ctypes.windll.user32
        u32.GetForegroundWindow.restype = ctypes.c_void_p
        u32.GetAncestor.restype = ctypes.c_void_p
        u32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        fg = u32.GetForegroundWindow()
        if not fg:
            return False
        hwnd = int(window.winId())
        return fg == hwnd or u32.GetAncestor(fg, 3) == hwnd       # 3 = GA_ROOTOWNER (popups, dialogs)
    except Exception:
        return True


def set_swap_interval(n):
    """VSync on/off at runtime on the CURRENT GL context (WGL_EXT_swap_control). -> success."""
    try:
        import ctypes
        gl = ctypes.WinDLL("opengl32")
        gl.wglGetProcAddress.restype = ctypes.c_void_p
        gl.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        addr = gl.wglGetProcAddress(b"wglSwapIntervalEXT")
        if not addr:
            return False
        return bool(ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)(addr)(int(n)))
    except Exception:
        return False


def _inv_view_proj(proj, lookat):
    """inverse(proj * lookat) in closed form (a rigid view matrix and a perspective projection), as float32 bytes:
    np.linalg.inv on the 4x4 cost ~0.14 ms per frame of numpy overhead."""
    P = np.asarray(proj, "f8").reshape(4, 4)
    L = np.asarray(lookat, "f8").reshape(4, 4)
    # row-vector convention (pyrr): v' = v @ M.  L = [[R, 0], [t, 1]] -> L^-1 = [[R^T, 0], [-t R^T, 1]]
    Li = np.zeros((4, 4))
    Rt = L[:3, :3].T
    Li[:3, :3] = Rt
    Li[3, :3] = -L[3, :3] @ Rt
    Li[3, 3] = 1.0
    # perspective: only [0,0], [1,1], [2,2], [2,3], [3,2] are non-zero
    a, b, c_, d, e = P[0, 0], P[1, 1], P[2, 2], P[2, 3], P[3, 2]
    Pi = np.zeros((4, 4))
    Pi[0, 0] = 1.0 / a
    Pi[1, 1] = 1.0 / b
    Pi[3, 2] = 1.0 / d
    Pi[2, 3] = 1.0 / e
    Pi[3, 3] = -c_ / (d * e)
    return (Pi @ Li).astype("f4").tobytes()


class _ProgPair:
    """Two programs that share the same uniforms (the arena's opaque and glass variants): prog["name"].value = v /
    .write(b) sets it on each program that has it (the compiler removes unused ones per variant)."""
    class _U:
        def __init__(self, us):
            self._us = us

        @property
        def value(self):
            return self._us[0].value

        @value.setter
        def value(self, v):
            for u in self._us:
                u.value = v

        def write(self, b):
            for u in self._us:
                u.write(b)

    def __init__(self, *progs):
        self.progs = progs
        self._cache = {}

    def __contains__(self, k):
        u = self._cache.get(k)
        if u is None:
            return any(k in p for p in self.progs)
        return True

    def __getitem__(self, k):
        u = self._cache.get(k)
        if u is None:
            us = [p[k] for p in self.progs if k in p]
            if not us:
                raise KeyError(k)
            u = self._cache[k] = _ProgPair._U(us)
        return u


class FramePacer:
    """VSync at the monitor's refresh when the machine keeps up, otherwise EVERY OTHER refresh (swap interval 2).

    Opt-in (RSV_PACER=1). On a 165 Hz screen a frame has 6.1 ms; when frames straddle that budget the display
    alternates between full and half rate. This locks to a steady half rate instead, and every PROBE_S seconds tries
    full rate again for a moment and keeps it if frames fit."""
    WINDOW = 90                  # frames per decision
    MISS = 0.08                  # > 8% of frames late = can't hold this rate
    PROBE_S = 20.0

    def __init__(self, refresh_hz):
        self.period = 1.0 / max(30.0, float(refresh_hz))
        self.interval = 1
        self.applied = None
        self.last = None
        self.late = self.n = 0
        self.probe_at = time.monotonic() + self.PROBE_S
        self.probing = False

    def reset(self):
        self.last = None
        self.late = self.n = 0

    def frame(self, enabled):
        """Call once per painted frame -> the swap interval to use."""
        if not enabled:
            self.interval, self.probing = 1, False
            self.reset()
            return 0
        now = time.monotonic()
        if self.last is not None:
            dt = now - self.last
            if dt > 0.25:                                # a pause (unfocused), not a frame
                self.reset()
            else:
                self.n += 1
                self.late += dt > (self.interval + 0.5) * self.period
        self.last = now
        if self.n >= self.WINDOW:
            miss = self.late / float(self.n)
            if self.interval == 1 and miss > self.MISS:
                self.interval = 2
                self.probe_at = now + (self.PROBE_S * 3 if self.probing else self.PROBE_S)
                print("[pacing] {:.0f}% late at {:.0f} fps -> {:.0f} fps".format(
                    100 * miss, 1 / self.period, 0.5 / self.period), flush=True)
            elif self.interval == 1 and self.probing:
                print("[pacing] back to {:.0f} fps".format(1 / self.period), flush=True)
            self.probing = False
            self.reset()
            self.last = now
        if self.interval == 2 and now >= self.probe_at:
            self.interval, self.probing = 1, True
            self.reset()
        return self.interval


class QRSVGLWidget(QtOpenGL.QGLWidget):
    """The window's GL surface: owns the Qt/OpenGL context and forwards everything to RSVRenderer."""

    def __init__(self, screen: QScreen):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setDoubleBuffer(True)
        # RSV_VSYNC=0 to bypass vsync (see git history: on a muxless-Optimus laptop an OpenGL app's own
        # vsync can stack with the dGPU->iGPU copy). Default 1 = tear-free.
        from config import Config as _Cfg
        self._vsync = int(_Cfg().gfx_vsync)          # Edit Settings > Graphics > VSync (live, see paintGL)
        fmt.setSwapInterval(self._vsync)
        # NO sample buffers on the window: MSAA is done in the renderer's own scene framebuffer.
        super(QRSVGLWidget, self).__init__(fmt, None)
        self.setMouseTracking(True)
        self.renderer = RSVRenderer()
        self.renderer.window_mode = True
        self.config = self.renderer.config

    def initializeGL(self):
        self.renderer.init_gl(moderngl.create_context())

    def paintGL(self):
        want = int(self.config.gfx_vsync)
        pacer = getattr(self, "pacer", None)
        if pacer is not None:
            # VSync on + "Monitor refresh": full or half refresh, whichever the machine holds (FramePacer)
            auto = want == 1 and self.config.gfx_fps == 0
            if pacer.n == 0:                             # the refresh of the screen the window is on now
                try:
                    hz = float(self.window().windowHandle().screen().refreshRate())
                    if hz >= 30.0 and abs(1.0 / hz - pacer.period) > 1e-4:
                        pacer.period = 1.0 / hz
                except Exception:
                    pass
            iv = pacer.frame(auto)
            if auto:
                want = iv
        if want != self._vsync:
            set_swap_interval(want)
            self._vsync = want
        self.renderer.paint(self.width(), self.height(), self.renderer.ctx.screen)

    def mousePressEvent(self, event):
        if self.renderer.handle_mouse_press(event):
            # Force an immediate repaint after re-aiming the camera so the click never leaves a stale frame.
            self.update()

    def keyPressEvent(self, event):
        # H toggles the whole top-left panel (stats + Edit Settings) while focused.
        if event.key() == Qt.Key_H and not event.isAutoRepeat():
            win = self.window()
            if hasattr(win, "toggle_panel"):
                win.toggle_panel()
            return
        self.renderer.handle_key_press(event)


g_socket_listener = None
def run_socket_thread(port):
    global g_socket_listener
    g_socket_listener = SocketListener()
    g_socket_listener.run(port)

def main():
    # Log any unhandled exception to a file (PyQt aborts silently on slot exceptions; pythonw has
    # no console). This makes RocketSimVis crashes diagnosable.
    import traceback as _tb
    _crash_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_crash.txt")

    def _excepthook(t, v, tb):
        try:
            with open(_crash_log, "a") as f:
                f.write("".join(_tb.format_exception(t, v, tb)) + "\n")
        except Exception:
            pass
        sys.__excepthook__(t, v, tb)
    sys.excepthook = _excepthook

    # RSV_PORT lets a second instance (tests, a side-by-side comparison) run without stealing the
    # training vis's socket.
    port = int(os.environ.get("RSV_PORT", "9273"))

    print("Starting RocketSimVis...")

    print("Starting socket thread...")
    socket_thread = threading.Thread(target=run_socket_thread, args=(int(port),))
    socket_thread.start()

    print("Starting visualizer window...")

    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    gl_widget = QRSVGLWidget(app.primaryScreen())
    window = QRSVWindow(gl_widget)
    window.showNormal()
    window._norm_geo = window.frameGeometry()   # seed the anti-auto-minimize geometry

    from PyQt5.QtCore import QTimer
    import socket as _socket
    _focus_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _was_active = [False]
    _active_t = [time.monotonic()]

    _side_next = [0.0]
    _fps_applied = [None]

    def _drive_render():
        # Edit Settings > Graphics > Frame rate cap (live)
        f = gl_widget.config.gfx_fps
        if f != _fps_applied[0]:
            _fps_applied[0] = f
            render_timer.setInterval(0 if f == -1 else (frame_ms if f == 0 else max(1, round(1000.0 / f))))
        # Only repaint the GL widget while the window is the ACTIVE window. Repainting an OpenGL
        # context for a backgrounded window crashes on some drivers — that was the vis "closing" on
        # click-away. While unfocused the window simply holds its last frame (it does NOT close).
        active = False
        try:
            active = window.isActiveWindow() and not window.isMinimized() and _is_foreground(window)
            # Suspend BEFORE repainting stops: Qt still repaints once on focus loss, which used to
            # restart the boost loop with no later frame to stop it.
            gl_widget.renderer.audio.set_active(active)
            if active:
                window.gl_widget.update()
        except Exception:
            pass
        _was_active[0] = active
        now_ = time.monotonic()
        if active:
            _active_t[0] = now_
        if now_ < _side_next[0]:
            return                       # clip poll + focus heartbeat at ~30 Hz, not every frame
        _side_next[0] = now_ + 0.03
        # The GUI's "Clip 12s" request must work while the GUI (not the vis) has focus: packets are
        # recorded by the socket thread regardless, so poll the trigger file here, not only in paint.
        try:
            gl_widget.renderer.clip_recorder.poll_signal()
        except Exception:
            pass
        # Report focus to a PLAY driver (if any) so it can HOLD playback while we're hidden.
        try:
            # "0" (pause the feed) only after 0.25 s unfocused in a row: while the window comes back to the front
            # Windows reports it as inactive for a few frames, and pausing the feed for each of those flickers
            # played the first moments after a tab-back in slow motion
            held = active or now_ - _active_t[0] < 0.25
            _focus_sock.sendto(b"1" if held else b"0", ("127.0.0.1", 9275))
        except Exception:
            pass

    render_timer = QTimer()
    render_timer.setTimerType(Qt.PreciseTimer)
    render_timer.timeout.connect(_drive_render)
    # Frame pacing: the monitor's refresh rate by default (vsync paces the rest). RSV_FPS=<n> caps it lower (e.g. to
    # leave more GPU to a training run, see tools/gpu_contention_bench.py); RSV_FRAME_MS overrides the interval.
    try:
        # fastest attached screen (the primary may be a 60 Hz external while the vis sits on a faster panel)
        refresh = max(float(sc.refreshRate()) for sc in app.screens()) or 60.0
    except Exception:
        refresh = 60.0
    # Default: uncapped = the monitor's refresh (vsync paces the rest). RSV_FPS=<n> caps it lower.
    fps_env = float(os.environ.get("RSV_FPS", "0") or 0)
    fps = refresh if fps_env <= 0 else min(refresh, fps_env)
    frame_ms = int(os.environ.get("RSV_FRAME_MS", max(1, int(1000.0 / max(1.0, fps)))))
    print("[timing] monitor {:.0f} Hz -> frame interval {} ms".format(refresh, frame_ms), flush=True)
    if os.environ.get("RSV_PACER") == "1":        # off by default: half-rate locking cost more fps than it saved in jitter
        gl_widget.pacer = FramePacer(refresh)
    # Frame-time spikes: (1) while training runs, its collection threads keep every core busy and the render thread
    # had to wait its turn at normal priority -> above normal (it uses the same CPU time, just on time); (2) every
    # startup object (meshes' Python wrappers, modules, caches) was rescanned by each full GC pass (~20 ms) -> frozen.
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000)       # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass
    import gc as _gc
    _gc.collect()
    _gc.freeze()
    render_timer.start(frame_ms)

    pose_timer = None
    try:
        from pose_editor import g_pose_editor
        pose_timer = QTimer()
        pose_timer.timeout.connect(g_pose_editor.poll_keys)
        pose_timer.start(8)   # ~120Hz so posing is smooth and test-mode input isn't undersampled
    except Exception:
        _tb_ = __import__("traceback").format_exc()
        try:
            open(_crash_log, "a").write("pose_editor timer setup failed:\n" + _tb_ + "\n")
        except Exception:
            pass

    app.exec_()

    print("Shutting down...")
    g_socket_listener.stop_async()
    exit()

if __name__ == "__main__":
    main()
