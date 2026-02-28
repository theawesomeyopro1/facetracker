import pygame
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time
import threading
import math
import traceback
from dataclasses import dataclass

# ── Download MediaPipe models if missing ───────────────────────────────────────
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
for fname, url in MODELS.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, fname)
        print(f"Done: {fname}")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

BaseOptions           = mp_tasks.BaseOptions
FaceLandmarker        = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

latest_face  = None
latest_hands = []
results_lock = threading.Lock()

def on_face(result, output_image, ts):
    global latest_face
    with results_lock:
        latest_face = result.face_landmarks[0] if result.face_landmarks else None

def on_hands(result, output_image, ts):
    global latest_hands
    with results_lock:
        latest_hands = result.hand_landmarks if result.hand_landmarks else []

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_face,
)
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_hands,
)

# ── Face & Hand landmark groups ───────────────────────────────────────────────
FACE_OVAL     = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE      = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
RIGHT_EYE     = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
NOSE          = [168,6,197,195,5,4,1,19,94,2,164,0,11,12,13,14,15,16,17,18]
LIPS_OUTER    = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]
FACE_GROUPS_SIMPLE = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, NOSE, LIPS_OUTER]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# ── Pygame setup ─────────────────────────────────────────────────────────────
pygame.init()
W, H = 640, 480
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Draw-a-Cube")
clock = pygame.time.Clock()
font  = pygame.font.SysFont("monospace", 13)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    pygame.quit()
    exit(1)

# ── Glow drawing helpers ──────────────────────────────────────────────────────
def glow_line(surf, color, p1, p2, w=2, layers=3):
    for i in range(layers,0,-1):
        pygame.draw.line(surf, color, p1, p2, w+i*2)

def glow_circle(surf, color, center, r, w=2, layers=3):
    for i in range(layers,0,-1):
        pygame.draw.circle(surf, color, center, r+i*2, w+i)

# ── Cube 3D ─────────────────────────────────────────────────────────────────
UNIT_VERTS = np.array([
    [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
    [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
], dtype=float)
CUBE_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def make_rotation(rx, ry, rz=0.0):
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def project_verts(verts, size, cx, cy):
    fov   = 600.0
    scale = fov / (fov + verts[:,2]*size*0.5 + 200)
    xs    = (verts[:,0]*size*scale + cx).astype(int)
    ys    = (verts[:,1]*size*scale + cy).astype(int)
    return list(zip(xs, ys))

class Cube:
    def __init__(self, cx, cy, size):
        self.cx, self.cy, self.size = float(cx), float(cy), float(size)
        self.rx, self.ry, self.rz = 0.3, 0.5, 0.0
        self.scale = 0.0
        self.color = (0,255,200)
        self.alive = True
        self.dying = False

    def start_close(self):
        self.dying = True

    def update(self, dt):
        if self.dying:
            self.scale = max(0.0, self.scale - dt*5)
            if self.scale < 0.01:
                self.alive = False
        else:
            self.scale = min(1.0, self.scale + dt*4)

    def draw(self, surf):
        s = self.size * self.scale
        R = make_rotation(self.rx, self.ry, self.rz)
        verts = UNIT_VERTS @ R.T
        pts = project_verts(verts, s, int(self.cx), int(self.cy))
        color = tuple(int(c*self.scale) for c in self.color)
        for a,b in CUBE_EDGES:
            glow_line(surf, color, pts[a], pts[b])

# ── Gesture helpers ───────────────────────────────────────────────────────────
PINCH_DIST = 40

def pinch_point(hpts):
    thumb = np.array(hpts[4])
    index = np.array(hpts[8])
    if np.linalg.norm(thumb-index) < PINCH_DIST:
        return tuple(((thumb+index)/2).astype(int))
    return None

def is_fist(hpts):
    wrist = np.array(hpts[0])
    tips  = np.array([hpts[i] for i in [4,8,12,16,20]])
    return np.mean(np.linalg.norm(tips - wrist, axis=1)) < 60

def hand_center(hpts):
    pts = np.array(hpts)
    return pts.mean(axis=0)

def smooth(prev, curr, alpha=0.5):
    if prev is None or len(prev)!=len(curr):
        return curr.copy()
    return alpha*curr + (1-alpha)*prev

# ── HUD ──────────────────────────────────────────────────────────────────────
def hud(surf, text, y, color=(60,220,150)):
    surf.blit(font.render(text, True, color), (10,y))

# ── State ────────────────────────────────────────────────────────────────────
@dataclass
class CubeState:
    state: str = 'idle'
    corner1: tuple = None
    corner_time: float = 0.0
    cube: Cube = None
    prev_one_hand: np.ndarray = None
    prev_two_angle: float = None
    prev_two_dist: float = None
    was_pinching: dict = None
    was_fist: dict = None
    pinch_cd: float = 0.0

state = CubeState(was_pinching={}, was_fist={})
smoothed_face = None
smoothed_hands = {}

# ── Main loop ────────────────────────────────────────────────────────────────
try:
    with FaceLandmarker.create_from_options(face_options) as face_lm, \
         HandLandmarker.create_from_options(hand_options) as hand_lm:

        print("Ready!")
        running = True
        prev_time = time.time()

        while running:
            now = time.time()
            dt = min(now-prev_time, 0.1)
            prev_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame,1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts    = int(now*1000)

            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                face_lm.detect_async(mp_img, ts)
                hand_lm.detect_async(mp_img, ts)
            except Exception:
                traceback.print_exc()

            surf = pygame.Surface((W,H), pygame.SRCALPHA)
            surf.fill((0,0,0,255))

            with results_lock:
                face_snap = latest_face
                hands_snap = list(latest_hands)

            # ── Face drawing ─────────────────────────
            if face_snap:
                raw = np.array([(lm.x*W, lm.y*H) for lm in face_snap], dtype=np.float32)
                smoothed_face = smooth(smoothed_face, raw, 0.35)
                pts = [(int(x),int(y)) for x,y in smoothed_face]
                for group in FACE_GROUPS_SIMPLE:
                    for i in range(0,len(group)-1,2):
                        a,b = group[i], group[i+1]
                        if a<len(pts) and b<len(pts):
                            pygame.draw.line(surf,(0,180,0),pts[a],pts[b],1)
                for x,y in pts:
                    pygame.draw.circle(surf,(0,220,0),(x,y),2)
            else:
                smoothed_face = None

            # ── Hands and gestures ─────────────────────────
            active_keys = set()
            hand_pts_list = []
            pinch_pts = []

            for hi, hand in enumerate(hands_snap):
                active_keys.add(hi)
                raw = np.array([(lm.x*W,lm.y*H) for lm in hand],dtype=np.float32)
                smoothed_hands[hi] = smooth(smoothed_hands.get(hi), raw, 0.5)
                hpts = [(int(x),int(y)) for x,y in smoothed_hands[hi]]
                hand_pts_list.append(hpts)
                pp = pinch_point(hpts)
                pinch_pts.append(pp)

                hcolor = (255,255,0) if pp else (200,140,0)
                lcolor = (180,180,0) if pp else (150,100,0)
                for a,b in HAND_CONNECTIONS:
                    if a<len(hpts) and b<len(hpts):
                        glow_line(surf,lcolor,hpts[a],hpts[b])
                for x,y in hpts:
                    glow_circle(surf,hcolor,(x,y),3)
                if pp:
                    glow_circle(surf,(255,255,255),pp,10,w=2,layers=4)

            for key in list(smoothed_hands.keys()):
                if key not in active_keys:
                    del smoothed_hands[key]

            # ── Pinch detection and state machine ─────────
            fresh_pinches = []
            for hi, hpts in enumerate(hand_pts_list):
                pp = pinch_pts[hi]
                is_p = pp is not None
                was_p = state.was_pinching.get(hi, False)
                if is_p and not was_p and now>state.pinch_cd:
                    fresh_pinches.append(pp)
                state.was_pinching[hi] = is_p
            for key in list(state.was_pinching.keys()):
                if key not in active_keys:
                    del state.was_pinching[key]

            # Handle cube states: idle → one_corner → alive
            if state.state=='idle':
                if fresh_pinches:
                    state.corner1 = fresh_pinches[0]
                    state.corner_time = now
                    state.state = 'one_corner'
                    state.pinch_cd = now+0.4

            elif state.state=='one_corner':
                glow_circle(surf,(255,100,0),state.corner1,12,w=2,layers=5)
                pulse = 0.5+0.5*math.sin((now-state.corner_time)*6)
                r = int(16 + pulse*8)
                glow_circle(surf,(255,150,50),state.corner1,r,w=1,layers=2)
                if fresh_pinches:
                    corner2 = fresh_pinches[0]
                    cx = (state.corner1[0]+corner2[0])/2
                    cy = (state.corner1[1]+corner2[1])/2
                    size = max(30, math.hypot(corner2[0]-state.corner1[0], corner2[1]-state.corner1[1])/2)
                    state.cube = Cube(cx,cy,size)
                    state.state = 'alive'
                    state.pinch_cd = now+0.5
                if now - state.corner_time > 4.0:
                    state.state = 'idle'
                    state.corner1 = None

            elif state.state=='alive':
                c = state.cube
                if c and c.alive:
                    c.update(dt)
                    fist_hands = [h for h in hand_pts_list if is_fist(h)]
                    if len(fist_hands)==1 and len(hand_pts_list)==1:
                        hc = hand_center(fist_hands[0])
                        if state.prev_one_hand is not None:
                            dx,dy = hc - state.prev_one_hand
                            c.ry += dx*0.01*dt*30
                            c.rx += dy*0.01*dt*30
                        state.prev_one_hand = hc
                    else:
                        state.prev_one_hand = None
                    # two hands twist/scale
                    if len(hand_pts_list)==2:
                        c0,c1 = hand_center(hand_pts_list[0]), hand_center(hand_pts_list[1])
                        angle = math.atan2(c1[1]-c0[1],c1[0]-c0[0])
                        dist  = float(np.linalg.norm(c1-c0))
                        if state.prev_two_angle is not None:
                            da = angle - state.prev_two_angle
                            if da>math.pi: da-=2*math.pi
                            if da<-math.pi: da+=2*math.pi
                            c.rz = getattr(c,'rz',0.0)+da
                            c.ry += da*0.5
                        if state.prev_two_dist is not None:
                            dd = dist - state.prev_two_dist
                            c.size = max(20,min(350,c.size+dd*0.5))
                        state.prev_two_angle = angle
                        state.prev_two_dist  = dist
                    else:
                        state.prev_two_angle = None
                        state.prev_two_dist  = None
                    # crush cube if all fists
                    if hand_pts_list and all(is_fist(h) for h in hand_pts_list):
                        c.start_close()
                    c.draw(surf)
                else:
                    state.state='idle'
                    state.cube=None
                    state.prev_one_hand=None
                    state.prev_two_angle=None
                    state.prev_two_dist=None

            # ── HUD ─────────────────────────────
            if state.state=='idle':
                hud(surf,"PINCH → place corner 1", 10)
            elif state.state=='one_corner':
                hud(surf,"PINCH again → place corner 2 & spawn cube", 10, (255,180,50))
            elif state.state=='alive':
                hud(surf,"FIST + drag   → rotate", 10)
                hud(surf,"2 HANDS twist → rotate / resize", 24)
                hud(surf,"FIST both     → crush cube", 38)

            # ── Render to screen ──────────────
            screen.blit(surf, (0,0))
            pygame.display.update()
            clock.tick_busy_loop(60)

except Exception:
    traceback.print_exc()
finally:
    cap.release()
    pygame.quit()
