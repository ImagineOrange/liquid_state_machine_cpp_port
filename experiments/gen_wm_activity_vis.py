#!/usr/bin/env python3
"""
Generate interactive 3D WM activity visualization with spike/adaptation toggle.

Reads the WM compound trial from example_rasters.json (which contains both
spike events and per-ms adaptation snapshots) and builds a self-contained HTML.

The "Spikes / Adaptation" toggle switches between:
  - Spike mode: neurons flash on spike, decay afterward (same as original)
  - Adaptation mode: neurons glow with luminance proportional to their
    adaptation conductance (dark=low, bright=high), updated every ms

Usage:
  python experiments/gen_wm_activity_vis.py
"""
import json
import numpy as np
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
RASTER_JSON = CPP_DIR / 'results' / 'mechanistic_interp' / 'example_rasters.json'
OUT_PATH = CPP_DIR / 'results' / 'activity_vis' / 'lsm_wm_activity.html'


def mel_bin_to_rgb(mel_bin, n_bins=128):
    t = mel_bin / max(n_bins - 1, 1)
    hue = t * 270.0
    s, l = 0.85, 0.55
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((hue / 60.0) % 2 - 1.0))
    m = l - c / 2.0
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    else:
        r, g, b = x, 0, c
    return (r + m, g + m, b + m)


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def main():
    if not RASTER_JSON.exists():
        print(f'{RASTER_JSON} not found. Run: ./cls_sweep --mech-raster --n-workers 8')
        return

    print(f'Loading {RASTER_JSON}...')
    with open(RASTER_JSON) as f:
        rdata = json.load(f)

    # Use the WM optimum entry
    wm_entry = next(e for e in rdata['rasters'] if e['label'] == 'wm_optimum')
    if 'adapt_snapshots' not in wm_entry or not wm_entry['adapt_snapshots']:
        print('No adapt_snapshots — regenerate with updated ./cls_sweep --mech-raster')
        return

    n_res = wm_entry['n_reservoir']
    n_adapt_snaps = wm_entry['n_adapt_snapshots']
    adapt_flat = np.array(wm_entry['adapt_snapshots'])
    adapt_matrix = adapt_flat.reshape(n_adapt_snaps, n_res)  # time x neuron

    # Load network snapshot for positions, zones, etc.
    snap = np.load(SNAPSHOT)
    n_neurons = int(snap['n_neurons'])
    positions = snap['positions']
    sphere_radius = float(snap['sphere_radius'])
    is_inhibitory = snap['is_inhibitory'].astype(bool)
    input_neuron_indices = set(snap['input_neuron_indices'].tolist())
    input_zone_indices = set(snap['input_zone_indices'].tolist())
    reservoir_zone_indices = list(snap['reservoir_zone_indices'])
    input_neuron_mapping = snap['input_neuron_mapping']

    # Build neuron -> primary mel bin from mapping
    neuron_primary_mel = {}
    for mel_bin in range(input_neuron_mapping.shape[0]):
        for k in range(input_neuron_mapping.shape[1]):
            nid = int(input_neuron_mapping[mel_bin, k])
            if nid >= 0:
                if nid not in neuron_primary_mel:
                    neuron_primary_mel[nid] = mel_bin

    # Per-neuron color + zone assignment
    n_mel_bins = 128
    neuron_colors = []
    zone_idx = []
    input_zone_unassigned = input_zone_indices - input_neuron_indices

    for i in range(n_neurons):
        if i in input_neuron_indices:
            zone_idx.append(0)
            mel = neuron_primary_mel.get(i, 64)
            r, g, b = mel_bin_to_rgb(mel, n_mel_bins)
            neuron_colors.append({'r': round(r, 3), 'g': round(g, 3), 'b': round(b, 3)})
        elif i in input_zone_unassigned:
            zone_idx.append(1)
            neuron_colors.append({'r': 1.0, 'g': 0.84, 'b': 0.25})
        elif is_inhibitory[i]:
            zone_idx.append(3)
            neuron_colors.append({'r': 0.2, 'g': 0.2, 'b': 1.0})
        else:
            zone_idx.append(2)
            neuron_colors.append({'r': 1.0, 'g': 0.2, 'b': 0.2})

    # Raster ordering
    zone_neuron_lists = [[], [], [], []]
    for i in range(n_neurons):
        zone_neuron_lists[zone_idx[i]].append(i)
    zone_neuron_lists[0].sort(key=lambda nid: neuron_primary_mel.get(nid, 64))

    raster_order = []
    zone_boundaries = []
    for zl in zone_neuron_lists:
        raster_order.extend(zl)
        zone_boundaries.append(len(raster_order))
    neuron_to_raster = {}
    for pos, neuron in enumerate(raster_order):
        neuron_to_raster[neuron] = pos

    # Build reservoir neuron index -> position mapping
    # reservoir_zone_indices[p] = global neuron ID for reservoir neuron p
    res_global_ids = list(reservoir_zone_indices)

    # Parse spike events — use all-neuron spikes (global IDs, includes input neurons)
    if 'all_spike_times_ms' in wm_entry:
        spike_times = np.array(wm_entry['all_spike_times_ms'])
        spike_global = np.array(wm_entry['all_spike_neuron_ids'])
    else:
        # Fallback: reservoir-only spikes
        spike_times = np.array(wm_entry['spike_times_ms'])
        spike_pos = np.array(wm_entry['spike_neuron_pos'])
        spike_global = np.array([res_global_ids[p] for p in spike_pos])

    # Time parameters
    dt = rdata['dt']  # 0.1 ms
    trial_end_ms = rdata['stim_b_end_ms'] + rdata['post_stim_ms']
    total_steps = int(trial_end_ms / dt)

    # Build per-step spike lists (all neurons, global IDs)
    activity_record = [[] for _ in range(total_steps)]
    for t_ms, gid in zip(spike_times, spike_global):
        step = int(round(t_ms / dt))
        if 0 <= step < total_steps:
            activity_record[step].append(int(gid))

    # Downsample to animation frames
    max_frames = 2000
    if total_steps > max_frames:
        frame_indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)
        sampled_activity = [activity_record[i] for i in frame_indices]
    else:
        frame_indices = np.arange(total_steps)
        sampled_activity = activity_record
    num_anim_frames = len(sampled_activity)

    # Build adaptation data: for each animation frame, map to the nearest ms snapshot
    # adapt_matrix is [n_adapt_snaps x n_res], indexed by ms
    # We need per-frame, per-neuron adaptation values for ALL neurons (input neurons = 0)
    # Downsample adapt to match animation frames
    adapt_per_frame = []
    for fi in frame_indices:
        t_ms = fi * dt
        snap_idx = min(int(t_ms), n_adapt_snaps - 1)
        # Build full-neuron array (input neurons get 0)
        vals = [0.0] * n_neurons
        for p in range(n_res):
            gid = res_global_ids[p]
            vals[gid] = float(adapt_matrix[snap_idx, p])
        adapt_per_frame.append(vals)

    # Use 99th percentile for normalization (max is an outlier spike)
    adapt_max = float(np.percentile(adapt_matrix[adapt_matrix > 0], 99))
    if adapt_max < 1e-6:
        adapt_max = 1.0

    # Epoch markers
    a_end_ms = rdata['stim_a_end_ms']
    b_start_ms = rdata['stim_b_offset_ms']
    b_end_ms = rdata['stim_b_end_ms']

    a_end_frame = int(a_end_ms / dt)
    b_start_frame = int(b_start_ms / dt)
    b_end_frame = int(b_end_ms / dt)

    # Serialize data
    positions_json = json.dumps([[round(float(positions[i][0]), 4),
                                   round(float(positions[i][1]), 4),
                                   round(float(positions[i][2]), 4)]
                                  for i in range(n_neurons)])
    neuron_colors_json = json.dumps(neuron_colors)
    zone_idx_json = json.dumps(zone_idx)
    is_inhibitory_json = json.dumps(is_inhibitory.tolist())
    spike_data_json = json.dumps(sampled_activity)
    frame_indices_json = json.dumps(frame_indices.tolist())
    zone_boundaries_json = json.dumps(zone_boundaries)
    neuron_to_raster_json = json.dumps(neuron_to_raster)

    # Compact adaptation data: store as flat Float32-ish array per frame
    # To keep file size manageable, quantize to 2 decimal places
    adapt_data_json = json.dumps([[round(v, 2) for v in frame]
                                   for frame in adapt_per_frame])

    zone_hex = ['#00e676', '#ffd740', '#ff3232', '#3232ff']
    zone_names = ['Input (tonotopic)', 'Input zone', 'Reservoir E', 'Reservoir I']

    mel_legend_bins = list(range(0, 128, 16)) + [127]
    mel_legend_colors = [rgb_to_hex(*mel_bin_to_rgb(b, 128)) for b in mel_legend_bins]

    adapt_inc = wm_entry['adapt_inc']
    adapt_tau = wm_entry['adapt_tau']
    g_tonic = wm_entry['g_tonic']
    digit_a = rdata['digit_a']
    digit_b = rdata['digit_b']

    info_label = f"WM Optimum — Δ<sub>a</sub>={adapt_inc:.4f}, τ<sub>a</sub>={adapt_tau:.0f} ms"
    info_sub = (f"Digit A={digit_a}, Digit B={digit_b} · "
                f"g<sub>tonic</sub>={g_tonic:.2f} nS (rate-matched)")

    tonic_legend_html = (
        '<div style="margin-top:10px;padding-top:6px;border-top:1px solid #333;">'
        '<div style="font-size:10px;color:#999;letter-spacing:0.3px;">'
        f'<span style="color:#b39ddb;">&#9644;</span> Tonic: {g_tonic:.2f} nS'
        '</div>'
        '<div style="font-size:9px;color:#666;margin-top:2px;">Rate-matching conductance</div>'
        '</div>'
    )

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LSM WM Activity — Spikes + Adaptation</title>
<style>
  body {{ margin:0; overflow:hidden; background:#0a0a0f; font-family:Arial,sans-serif; }}
  #main {{ display:flex; height:calc(100vh - 52px); }}
  #three-container {{ flex:65; position:relative; }}
  #raster-container {{ flex:35; position:relative; background:#0d0d18;
      border-left:1px solid #333; }}
  #rasterCanvas {{ display:block; width:100%; height:100%; }}
  #controls {{
      height:52px; display:flex; align-items:center; justify-content:center;
      gap:14px; background:rgba(0,0,0,0.6); padding:0 16px;
  }}
  #controls button {{
      padding:7px 14px; font-size:13px; cursor:pointer;
      background:#333; color:#fff; border:1px solid #555; border-radius:4px;
  }}
  #controls button:hover {{ background:#444; }}
  #controls button.active {{ background:#1a6b1a; border-color:#2a2; }}
  #slider {{ width:280px; cursor:pointer; }}
  #timeDisplay {{ color:#fff; font-family:monospace; min-width:80px; font-size:13px; }}
  #info {{
      position:absolute; top:10px; left:10px; color:#fff; font-size:13px;
      z-index:100; background:rgba(0,0,0,0.5); padding:8px 12px; border-radius:5px;
  }}
  #legend {{
      position:absolute; top:10px; right:10px; color:#fff; font-size:11px;
      z-index:100; background:rgba(0,0,0,0.5); padding:8px 12px; border-radius:5px;
  }}
  .leg-item {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
  .leg-dot {{ width:10px; height:10px; border-radius:50%; }}
  #mel-colorbar {{
      margin-top:8px; display:flex; height:12px; border-radius:3px; overflow:hidden;
  }}
  #mel-colorbar div {{ flex:1; }}
  .mel-labels {{ display:flex; justify-content:space-between; font-size:9px; color:#888; margin-top:2px; }}
  #adapt-legend {{ display:none; margin-top:8px; padding-top:6px; border-top:1px solid #333; }}
  #adapt-colorbar {{
      height:12px; border-radius:3px; margin-top:4px;
      background: linear-gradient(to right, #000000, #550000, #ff0000, #ff8800, #ffff00, #ffffff);
  }}
  .adapt-labels {{ display:flex; justify-content:space-between; font-size:9px; color:#888; margin-top:2px; }}
</style>
</head>
<body>
<div id="main">
  <div id="three-container">
    <div id="info">{info_label}<br>
      <span style="font-size:11px;color:#aaa;">{info_sub}</span><br>
      <span style="font-size:11px;color:#aaa;">Drag to orbit &middot; scroll to zoom</span></div>
    <div id="legend">
      <div style="font-size:12px;margin-bottom:4px;color:#ccc;">Input neurons (mel bin)</div>
      <div id="mel-colorbar">
        {"".join(f'<div style="background:{c}"></div>' for c in mel_legend_colors)}
      </div>
      <div class="mel-labels"><span>0 (low)</span><span>64</span><span>127 (high)</span></div>
      <div style="margin-top:8px;">
        <div class="leg-item"><div class="leg-dot" style="background:#ffd740"></div>Input zone</div>
        <div class="leg-item"><div class="leg-dot" style="background:#ff3232"></div>Reservoir E</div>
        <div class="leg-item"><div class="leg-dot" style="background:#3232ff"></div>Reservoir I</div>
      </div>
      {tonic_legend_html}
      <div id="adapt-legend">
        <div style="font-size:10px;color:#ccc;">Adaptation conductance</div>
        <div id="adapt-colorbar"></div>
        <div class="adapt-labels"><span>0</span><span>{adapt_max/2:.1f}</span><span>{adapt_max:.1f}</span></div>
      </div>
    </div>
  </div>
  <div id="raster-container">
    <canvas id="rasterCanvas"></canvas>
  </div>
</div>
<div id="controls">
  <button id="playPauseBtn">Pause</button>
  <button id="rotateBtn">Stop Rotate</button>
  <button id="modeBtn">Spikes</button>
  <button id="soundBtn">Sound Off</button>
  <input type="range" id="slider" min="0" max="{num_anim_frames - 1}" value="0">
  <span id="timeDisplay">0.0 ms</span>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const neuronPositions = {positions_json};
const neuronColors    = {neuron_colors_json};
const neuronZoneIdx   = {zone_idx_json};
const isInhibitory    = {is_inhibitory_json};
const spikeData       = {spike_data_json};
const adaptData       = {adapt_data_json};
const frameIndices    = {frame_indices_json};
const zoneBoundaries  = {zone_boundaries_json};
const neuronToRaster  = {neuron_to_raster_json};
const zoneHex         = {json.dumps(zone_hex)};
const zoneNames       = {json.dumps(zone_names)};
const neuronCount     = {n_neurons};
const sphereRadius    = {sphere_radius};
const decayFactor     = 0.8;
const dt              = {dt};
const totalAnimFrames = {num_anim_frames};
const adaptMax        = {adapt_max};

// Epoch boundaries in simulation steps
const epochSteps = [
  {{step: {a_end_frame}, label: 'A end'}},
  {{step: {b_start_frame}, label: 'B start'}},
  {{step: {b_end_frame}, label: 'B end'}},
];

// View mode: 'spikes' or 'adapt'
let viewMode = 'spikes';

// THREE.JS SETUP
const threeContainer = document.getElementById('three-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
const camera = new THREE.PerspectiveCamera(
    60, threeContainer.clientWidth / threeContainer.clientHeight, 0.1, 1000);
camera.position.set(sphereRadius*1.44, sphereRadius*1.44, sphereRadius*1.2);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
threeContainer.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = sphereRadius*1.5;
controls.maxDistance = sphereRadius*10;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.69;

// Wireframe sphere
const sphGeom = new THREE.SphereGeometry(sphereRadius, 32, 32);
const sphMat  = new THREE.MeshBasicMaterial({{
    color:0x888888, transparent:true, opacity:0.05, wireframe:true }});
scene.add(new THREE.Mesh(sphGeom, sphMat));

// Background neuron points
const bgGeom = new THREE.BufferGeometry();
const bgPos  = new Float32Array(neuronCount*3);
const bgCol  = new Float32Array(neuronCount*3);
for(let i=0;i<neuronCount;i++){{
  bgPos[i*3]=neuronPositions[i][0]; bgPos[i*3+1]=neuronPositions[i][1]; bgPos[i*3+2]=neuronPositions[i][2];
  const c=neuronColors[i];
  bgCol[i*3]=c.r*0.25; bgCol[i*3+1]=c.g*0.25; bgCol[i*3+2]=c.b*0.25;
}}
bgGeom.setAttribute('position',new THREE.BufferAttribute(bgPos,3));
bgGeom.setAttribute('color',new THREE.BufferAttribute(bgCol,3));
const bgPoints = new THREE.Points(bgGeom, new THREE.PointsMaterial({{
  size:0.15, vertexColors:true, transparent:true, opacity:0.3, sizeAttenuation:true
}}));
scene.add(bgPoints);

// Active neuron points (shader for glow)
const actGeom = new THREE.BufferGeometry();
const actPos  = new Float32Array(neuronCount*3);
const actCol  = new Float32Array(neuronCount*3);
const actSize = new Float32Array(neuronCount);
for(let i=0;i<neuronCount;i++){{
  actPos[i*3]=neuronPositions[i][0]; actPos[i*3+1]=neuronPositions[i][1]; actPos[i*3+2]=neuronPositions[i][2];
}}
actGeom.setAttribute('position',new THREE.BufferAttribute(actPos,3));
actGeom.setAttribute('color',new THREE.BufferAttribute(actCol,3));
actGeom.setAttribute('size',new THREE.BufferAttribute(actSize,1));

function createCircleTex(){{
  const c=document.createElement('canvas'); c.width=64; c.height=64;
  const x=c.getContext('2d');
  const g=x.createRadialGradient(32,32,0,32,32,32);
  g.addColorStop(0,'rgba(255,255,255,1)');
  g.addColorStop(0.3,'rgba(255,255,255,0.8)');
  g.addColorStop(1,'rgba(255,255,255,0)');
  x.fillStyle=g; x.fillRect(0,0,64,64);
  return new THREE.CanvasTexture(c);
}}

const actMat = new THREE.ShaderMaterial({{
  uniforms:{{ pointTexture:{{value:createCircleTex()}} }},
  vertexShader:`
    attribute float size;
    varying vec3 vColor;
    void main(){{
      vColor=color;
      vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=size*(300.0/-mv.z);
      gl_Position=projectionMatrix*mv;
    }}`,
  fragmentShader:`
    uniform sampler2D pointTexture;
    varying vec3 vColor;
    void main(){{
      vec4 t=texture2D(pointTexture,gl_PointCoord);
      if(t.a<0.1) discard;
      gl_FragColor=vec4(vColor,t.a);
    }}`,
  blending:THREE.AdditiveBlending,
  depthTest:true, depthWrite:false, transparent:true, vertexColors:true
}});
scene.add(new THREE.Points(actGeom, actMat));

// Adaptation colormap: hot (black -> red -> orange -> yellow -> white)
function adaptColor(v) {{
  // v in [0, 1], clamped
  v = Math.min(v, 1.0);
  let r, g, b;
  if (v < 0.33) {{
    const t = v / 0.33;
    r = t; g = 0; b = 0;
  }} else if (v < 0.66) {{
    const t = (v - 0.33) / 0.33;
    r = 1.0; g = t; b = 0;
  }} else {{
    const t = (v - 0.66) / 0.34;
    r = 1.0; g = 1.0; b = t;
  }}
  return {{ r, g, b }};
}}

// RASTER CANVAS
const rasterCanvas = document.getElementById('rasterCanvas');
const rasterCtx    = rasterCanvas.getContext('2d');
let rasterBuffer   = document.createElement('canvas');
let rasterAdaptBuffer = document.createElement('canvas');

function neuronHex(idx){{
  const c = neuronColors[idx];
  const toHex = v => {{const h=Math.round(v*255).toString(16); return h.length<2?'0'+h:h;}};
  return '#'+toHex(c.r)+toHex(c.g)+toHex(c.b);
}}

function prerenderRaster(){{
  const rc = document.getElementById('raster-container');
  const W = rc.clientWidth, H = rc.clientHeight;
  rasterCanvas.width = W; rasterCanvas.height = H;

  // --- Spike raster buffer ---
  rasterBuffer.width = W; rasterBuffer.height = H;
  const ctx = rasterBuffer.getContext('2d');
  ctx.fillStyle='#0d0d18'; ctx.fillRect(0,0,W,H);
  const margin=55, plotW=W-margin, plotH=H-4;
  ctx.font='10px monospace';

  for(let z=0;z<4;z++){{
    const startY=(z===0?0:zoneBoundaries[z-1])/neuronCount*plotH+2;
    const endY=zoneBoundaries[z]/neuronCount*plotH+2;
    ctx.strokeStyle='#2a2a3a'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(margin,endY); ctx.lineTo(W,endY); ctx.stroke();
    ctx.fillStyle=zoneHex[z];
    ctx.fillText(zoneNames[z], 2, (startY+endY)/2+3);
  }}

  // Epoch markers
  const lastFrame = frameIndices[totalAnimFrames-1]+1;
  for(const ep of epochSteps){{
    const mx = margin + (ep.step/lastFrame)*plotW;
    ctx.strokeStyle='rgba(255,255,255,0.2)'; ctx.lineWidth=1;
    ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(mx,0); ctx.lineTo(mx,H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='rgba(255,255,255,0.3)'; ctx.font='9px monospace';
    ctx.fillText(ep.label, mx+3, H-6);
  }}

  for(let f=0;f<totalAnimFrames;f++){{
    const spikes=spikeData[f];
    const x=margin+(f/totalAnimFrames)*plotW;
    for(let s=0;s<spikes.length;s++){{
      const idx=spikes[s];
      const rasterPos=neuronToRaster[idx];
      const y=(rasterPos/neuronCount)*plotH+2;
      ctx.fillStyle = neuronZoneIdx[idx]===0 ? neuronHex(idx) : zoneHex[neuronZoneIdx[idx]];
      ctx.fillRect(x,y,1.5,1);
    }}
  }}

  // --- Adaptation raster buffer ---
  rasterAdaptBuffer.width = W; rasterAdaptBuffer.height = H;
  const actx = rasterAdaptBuffer.getContext('2d');
  actx.fillStyle='#0d0d18'; actx.fillRect(0,0,W,H);

  for(let z=0;z<4;z++){{
    const startY=(z===0?0:zoneBoundaries[z-1])/neuronCount*plotH+2;
    const endY=zoneBoundaries[z]/neuronCount*plotH+2;
    actx.strokeStyle='#2a2a3a'; actx.lineWidth=0.5;
    actx.beginPath(); actx.moveTo(margin,endY); actx.lineTo(W,endY); actx.stroke();
    actx.fillStyle='#666';
    actx.font='10px monospace';
    actx.fillText(zoneNames[z], 2, (startY+endY)/2+3);
  }}

  for(const ep of epochSteps){{
    const mx = margin + (ep.step/lastFrame)*plotW;
    actx.strokeStyle='rgba(255,255,255,0.2)'; actx.lineWidth=1;
    actx.setLineDash([4,3]);
    actx.beginPath(); actx.moveTo(mx,0); actx.lineTo(mx,H); actx.stroke();
    actx.setLineDash([]);
    actx.fillStyle='rgba(255,255,255,0.3)'; actx.font='9px monospace';
    actx.fillText(ep.label, mx+3, H-6);
  }}

  // Draw adaptation heatmap on raster
  for(let f=0;f<totalAnimFrames;f++){{
    const avals=adaptData[f];
    const x=margin+(f/totalAnimFrames)*plotW;
    for(let i=0;i<neuronCount;i++){{
      const v = Math.min(avals[i] / adaptMax, 1.0);
      if(v < 0.01) continue;
      const rasterPos=neuronToRaster[i];
      const y=(rasterPos/neuronCount)*plotH+2;
      const c = adaptColor(v);
      const ri=Math.round(c.r*255), gi=Math.round(c.g*255), bi=Math.round(c.b*255);
      actx.fillStyle='rgb('+ri+','+gi+','+bi+')';
      actx.fillRect(x,y,1.5,1);
    }}
  }}
}}

function drawRasterFrame(frameIdx){{
  const W=rasterCanvas.width, H=rasterCanvas.height;
  const buf = viewMode==='adapt' ? rasterAdaptBuffer : rasterBuffer;
  rasterCtx.drawImage(buf,0,0);
  const margin=55, plotW=W-margin;
  const x=margin+(frameIdx/totalAnimFrames)*plotW;
  rasterCtx.strokeStyle='rgba(255,255,255,0.75)';
  rasterCtx.lineWidth=1.5;
  rasterCtx.beginPath(); rasterCtx.moveTo(x,0); rasterCtx.lineTo(x,H); rasterCtx.stroke();
  const timeMs=frameIndices[frameIdx]*dt;
  rasterCtx.fillStyle='#fff'; rasterCtx.font='11px monospace';
  rasterCtx.fillText(timeMs.toFixed(0)+' ms', Math.min(x+4,W-50), 14);
}}

// ANIMATION
let currentFrame=0, isPlaying=true, lastFrameTime=0;
const frameDelay=33;
const intensity=new Float32Array(neuronCount);
const slider=document.getElementById('slider');
const timeDisplay=document.getElementById('timeDisplay');

function updateNeurons(frameIdx){{
  const ca=actGeom.attributes.color, sa=actGeom.attributes.size;

  if(viewMode === 'spikes'){{
    for(let i=0;i<neuronCount;i++) intensity[i]*=decayFactor;
    const spikes=spikeData[frameIdx];
    for(let s=0;s<spikes.length;s++) intensity[spikes[s]]=1.0;
    for(let i=0;i<neuronCount;i++){{
      const v=intensity[i];
      if(v>0.02){{
        const c=neuronColors[i];
        ca.array[i*3]=c.r*v; ca.array[i*3+1]=c.g*v; ca.array[i*3+2]=c.b*v;
        sa.array[i]=0.3+v*0.7;
      }} else {{
        ca.array[i*3]=0; ca.array[i*3+1]=0; ca.array[i*3+2]=0; sa.array[i]=0;
      }}
    }}
  }} else {{
    // Adaptation mode
    const avals=adaptData[frameIdx];
    for(let i=0;i<neuronCount;i++){{
      const v = Math.min(avals[i] / adaptMax, 1.0);
      if(v > 0.005){{
        const c = adaptColor(v);
        ca.array[i*3]=c.r; ca.array[i*3+1]=c.g; ca.array[i*3+2]=c.b;
        sa.array[i]=0.35+v*0.65;
      }} else {{
        // Dim baseline dot for spatial context
        ca.array[i*3]=0.05; ca.array[i*3+1]=0.05; ca.array[i*3+2]=0.05; sa.array[i]=0.12;
      }}
    }}
  }}

  ca.needsUpdate=true; sa.needsUpdate=true;
  slider.value=frameIdx;
  timeDisplay.textContent=(frameIndices[frameIdx]*dt).toFixed(1)+' ms';
  drawRasterFrame(frameIdx);

  if(viewMode==='spikes' && soundEnabled){{
    const spikes=spikeData[frameIdx];
    if(spikes.length>0){{
      let eC=0,iC=0;
      for(let s=0;s<spikes.length;s++){{ if(isInhibitory[spikes[s]]) iC++; else eC++; }}
      playSpikeAudio(eC,iC);
    }}
  }}
}}

// Audio
let audioContext=null, soundEnabled=false;
function initAudio(){{ if(!audioContext) audioContext=new(window.AudioContext||window.webkitAudioContext)(); }}
function playSpikeAudio(eC,iC){{
  if(!soundEnabled||!audioContext) return;
  const now=audioContext.currentTime;
  if(eC>0){{
    const o=audioContext.createOscillator(),g=audioContext.createGain();
    o.frequency.value=784+Math.random()*200; o.type='sine';
    g.gain.setValueAtTime(Math.min(0.05,0.01*Math.log2(eC+1)),now);
    g.gain.exponentialRampToValueAtTime(0.001,now+0.18);
    o.connect(g); g.connect(audioContext.destination); o.start(now); o.stop(now+0.18);
  }}
  if(iC>0){{
    const o=audioContext.createOscillator(),g=audioContext.createGain();
    o.frequency.value=523+Math.random()*120; o.type='sine';
    g.gain.setValueAtTime(Math.min(0.035,0.008*Math.log2(iC+1)),now);
    g.gain.exponentialRampToValueAtTime(0.001,now+0.18);
    o.connect(g); g.connect(audioContext.destination); o.start(now); o.stop(now+0.18);
  }}
}}

// Controls
document.getElementById('playPauseBtn').addEventListener('click',()=>{{
  isPlaying=!isPlaying;
  document.getElementById('playPauseBtn').textContent=isPlaying?'Pause':'Play';
}});
document.getElementById('rotateBtn').addEventListener('click',()=>{{
  controls.autoRotate=!controls.autoRotate;
  document.getElementById('rotateBtn').textContent=controls.autoRotate?'Stop Rotate':'Rotate';
}});
document.getElementById('modeBtn').addEventListener('click',()=>{{
  viewMode = viewMode==='spikes' ? 'adapt' : 'spikes';
  const btn = document.getElementById('modeBtn');
  btn.textContent = viewMode==='spikes' ? 'Spikes' : 'Adaptation';
  btn.classList.toggle('active', viewMode==='adapt');
  document.getElementById('adapt-legend').style.display = viewMode==='adapt' ? 'block' : 'none';
  // Reset intensity for clean transition
  for(let i=0;i<neuronCount;i++) intensity[i]=0;
  updateNeurons(currentFrame);
}});
document.getElementById('soundBtn').addEventListener('click',()=>{{
  initAudio(); soundEnabled=!soundEnabled;
  document.getElementById('soundBtn').textContent=soundEnabled?'Sound On':'Sound Off';
}});
slider.addEventListener('input',(e)=>{{
  currentFrame=parseInt(e.target.value); updateNeurons(currentFrame);
}});

function animate(time){{
  requestAnimationFrame(animate);
  if(isPlaying && time-lastFrameTime>frameDelay){{
    currentFrame=(currentFrame+1)%totalAnimFrames;
    updateNeurons(currentFrame);
    lastFrameTime=time;
  }}
  controls.update();
  renderer.render(scene,camera);
}}

window.addEventListener('resize',()=>{{
  const tw=threeContainer.clientWidth, th=threeContainer.clientHeight;
  camera.aspect=tw/th; camera.updateProjectionMatrix();
  renderer.setSize(tw,th); prerenderRaster(); drawRasterFrame(currentFrame);
}});

prerenderRaster(); updateNeurons(0); animate(0);
</script>
</body>
</html>'''

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    print(f'Saved: {OUT_PATH}')
    print(f'\nOpen in browser:\n  file://{OUT_PATH.resolve()}')


if __name__ == '__main__':
    main()
