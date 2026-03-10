#!/usr/bin/env python3
"""
Generate interactive 3D activity visualization (Three.js) with raster panel.

Runs the C++ raster dump with 0 input adaptation, then builds a self-contained
HTML with:
  - Left: 3D sphere with neurons colored by zone / mel bin (input neurons
    colored by their primary mel frequency bin on a perceptual rainbow scale)
  - Right: spike raster grouped by zone

Usage:
  python experiments/gen_activity_vis.py [--sample <name>] [--output <path>]
"""
import argparse
import subprocess
import json
import numpy as np
import pandas as pd
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent
CPP_BIN = CPP_DIR / 'cls_sweep'
SNAPSHOT = CPP_DIR / 'network_snapshot.npz'
DATA_DIR = CPP_DIR / 'data'
BSA_DIR = DATA_DIR / 'spike_trains_bsa'
OUT_DIR = CPP_DIR / 'results' / 'activity_vis'

DEFAULT_SAMPLE = 'spike_train_0_george_0.npz'

# Simulation params: Best classification cell (Branch B, inc_idx=10, tau_idx=14)
# 96.33% accuracy — rate-matched with tonic conductance
STIM_CURRENT = 0.0518
INPUT_TAU_E = 1.05
INPUT_ADAPT_INC = 0.0
ADAPT_INC = 0.0707       # reservoir adaptation increment (nS)
ADAPT_TAU = 5000.0        # reservoir adaptation time constant (ms)
TONIC_CONDUCTANCE = 1.40625  # rate-matching tonic inhibitory conductance (nS)


def run_raster_dump(sample_path, dump_dir):
    """Run the C++ binary to generate spike dump CSVs."""
    cmd = [
        str(CPP_BIN), '--snapshot', str(SNAPSHOT),
        '--trace-file', str(sample_path),
        '--raster-dump', str(dump_dir),
        '--stim-current', str(STIM_CURRENT),
        '--input-tau-e', str(INPUT_TAU_E),
        '--input-adapt-inc', str(INPUT_ADAPT_INC),
        '--adapt-inc', str(ADAPT_INC),
        '--adapt-tau', str(ADAPT_TAU),
        '--tonic-conductance', str(TONIC_CONDUCTANCE),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def mel_bin_to_rgb(mel_bin, n_bins=128):
    """Map a mel bin index [0, n_bins) to a perceptual rainbow color.

    Low bins (bass) → warm reds/oranges, high bins (treble) → cool blues/violets.
    Uses HSL with hue sweeping 0° → 270° so we avoid wrapping back to red.
    """
    t = mel_bin / max(n_bins - 1, 1)  # 0..1
    hue = t * 270.0  # degrees
    # Convert HSL (hue, 0.85, 0.55) → RGB
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


def build_html(snapshot_path, dump_dir, save_path, max_frames=2000, decay_factor=0.8):
    """Build the interactive HTML visualization."""
    # Load network snapshot
    snap = np.load(snapshot_path)
    n_neurons = int(snap['n_neurons'])
    positions = snap['positions']  # (n, 3)
    sphere_radius = float(snap['sphere_radius'])
    is_inhibitory = snap['is_inhibitory'].astype(bool)
    input_neuron_indices = set(snap['input_neuron_indices'].tolist())
    input_zone_indices = set(snap['input_zone_indices'].tolist())
    reservoir_indices = set(snap['reservoir_zone_indices'].tolist())
    input_neuron_mapping = snap['input_neuron_mapping']  # (128, K)

    # Build neuron → primary mel bin (weighted-average center, from neurons.csv)
    neurons_csv = pd.read_csv(dump_dir / 'neurons.csv')
    neuron_primary_mel = {}
    for _, row in neurons_csv.iterrows():
        nid = int(row['neuron_id'])
        bins = [int(b) for b in str(row['freq_bins']).split(';')]
        weights = [float(w) for w in str(row['weights']).split(';')]
        # Weighted average mel bin
        total_w = sum(weights)
        if total_w > 0:
            avg_bin = sum(b * w for b, w in zip(bins, weights)) / total_w
        else:
            avg_bin = bins[0]
        neuron_primary_mel[nid] = avg_bin

    # Per-neuron color assignment
    # Input neurons: colored by mel bin (rainbow)
    # Input zone (unassigned): dim yellow
    # Reservoir excitatory: red
    # Reservoir inhibitory: blue
    n_mel_bins = 128
    neuron_colors = []  # list of {r, g, b}
    zone_idx = []       # per-neuron zone index (for raster grouping)
    # Zone 0 = input (assigned), 1 = input zone unassigned, 2 = reservoir E, 3 = reservoir I
    # But for input neurons we'll use per-mel-bin colors instead of a single zone color

    input_zone_unassigned = input_zone_indices - input_neuron_indices

    for i in range(n_neurons):
        if i in input_neuron_indices:
            zone_idx.append(0)
            mel = neuron_primary_mel.get(i, 64)
            r, g, b = mel_bin_to_rgb(mel, n_mel_bins)
            neuron_colors.append({'r': r, 'g': g, 'b': b})
        elif i in input_zone_unassigned:
            zone_idx.append(1)
            neuron_colors.append({'r': 1.0, 'g': 0.84, 'b': 0.25})
        elif is_inhibitory[i]:
            zone_idx.append(3)
            neuron_colors.append({'r': 0.2, 'g': 0.2, 'b': 1.0})
        else:
            zone_idx.append(2)
            neuron_colors.append({'r': 1.0, 'g': 0.2, 'b': 0.2})

    # Raster ordering: group by zone, within input zone sort by mel bin
    zone_neuron_lists = [[], [], [], []]
    for i in range(n_neurons):
        zone_neuron_lists[zone_idx[i]].append(i)

    # Sort input neurons by mel bin for the raster
    zone_neuron_lists[0].sort(key=lambda nid: neuron_primary_mel.get(nid, 64))

    raster_order = []
    zone_boundaries = []
    for zl in zone_neuron_lists:
        raster_order.extend(zl)
        zone_boundaries.append(len(raster_order))
    neuron_to_raster = {}
    for pos, neuron in enumerate(raster_order):
        neuron_to_raster[neuron] = pos

    # Load spike data
    meta = json.loads((dump_dir / 'meta.json').read_text())
    dt = meta['dt']
    audio_duration_ms = meta['audio_duration_ms']
    total_ms = meta['total_ms']
    warmup_ms = meta['warmup_ms']

    spikes_df = pd.read_csv(dump_dir / 'spikes.csv')
    total_steps = int(total_ms / dt)

    # Build per-step spike lists
    activity_record = [[] for _ in range(total_steps)]
    for _, row in spikes_df.iterrows():
        step = int(round(row['time_ms'] / dt))
        if 0 <= step < total_steps:
            activity_record[step].append(int(row['neuron_id']))

    # Downsample frames for animation
    if total_steps > max_frames:
        frame_indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)
        sampled_activity = [activity_record[i] for i in frame_indices]
    else:
        frame_indices = np.arange(total_steps)
        sampled_activity = activity_record

    num_anim_frames = len(sampled_activity)

    # Epoch markers
    stim_end_frame = int(audio_duration_ms / dt)

    # Build JSON data
    positions_json = json.dumps([[float(positions[i][0]), float(positions[i][1]),
                                  float(positions[i][2])] for i in range(n_neurons)])
    neuron_colors_json = json.dumps(neuron_colors)
    zone_idx_json = json.dumps(zone_idx)
    is_inhibitory_json = json.dumps(is_inhibitory.tolist())
    spike_data_json = json.dumps(sampled_activity)
    frame_indices_json = json.dumps(frame_indices.tolist())
    zone_boundaries_json = json.dumps(zone_boundaries)
    neuron_to_raster_json = json.dumps(neuron_to_raster)

    zone_hex = ['#00e676', '#ffd740', '#ff3232', '#3232ff']
    zone_names = ['Input (tonotopic)', 'Input zone', 'Reservoir E', 'Reservoir I']
    zone_hex_json = json.dumps(zone_hex)
    zone_names_json = json.dumps(zone_names)

    digit = meta.get('digit', '?')
    filename = meta.get('filename', '')
    adapt_inc = meta.get('adapt_inc', 0.0)
    adapt_tau = meta.get('adapt_tau', 0.0)
    g_tonic = meta.get('tonic_conductance', 0.0)

    if adapt_inc > 0:
        info_label = f"Best Classification Cell — Δ<sub>a</sub>={adapt_inc:.4f} nS, τ<sub>a</sub>={adapt_tau:.0f} ms"
    else:
        info_label = f"LHS-021 — No Adaptation"
    info_sub = f"Digit {digit} — {filename}"
    if g_tonic > 0:
        info_sub += f" · g<sub>tonic</sub>={g_tonic:.2f} nS (rate-matched)"

    # Build mel colorbar data for legend (sample of mel bins)
    mel_legend_bins = list(range(0, 128, 16)) + [127]
    mel_legend_colors = [rgb_to_hex(*mel_bin_to_rgb(b, 128)) for b in mel_legend_bins]

    # Build tonic conductance legend snippet (empty string if no tonic conductance)
    if g_tonic > 0:
        tonic_legend_html = (
            '<div style="margin-top:10px;padding-top:6px;border-top:1px solid #333;">'
            '<div style="font-size:10px;color:#999;letter-spacing:0.3px;">'
            f'<span style="color:#b39ddb;">&#9644;</span> Tonic inhibition: {g_tonic:.2f} nS'
            '</div>'
            '<div style="font-size:9px;color:#666;margin-top:2px;">Background conductance for rate-matching</div>'
            '</div>'
        )
    else:
        tonic_legend_html = ''

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LSM Activity — 3D + Raster</title>
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
  #slider {{ width:320px; cursor:pointer; }}
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
        <div class="leg-item"><div class="leg-dot" style="background:#ffd740"></div>Input zone (unassigned)</div>
        <div class="leg-item"><div class="leg-dot" style="background:#ff3232"></div>Reservoir E</div>
        <div class="leg-item"><div class="leg-dot" style="background:#3232ff"></div>Reservoir I</div>
      </div>
      {tonic_legend_html}
    </div>
  </div>
  <div id="raster-container">
    <canvas id="rasterCanvas"></canvas>
  </div>
</div>
<div id="controls">
  <button id="playPauseBtn">Pause</button>
  <button id="rotateBtn">Stop Rotate</button>
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
const frameIndices    = {frame_indices_json};
const zoneBoundaries  = {zone_boundaries_json};
const neuronToRaster  = {neuron_to_raster_json};
const zoneHex         = {zone_hex_json};
const zoneNames       = {zone_names_json};
const neuronCount     = {n_neurons};
const sphereRadius    = {sphere_radius};
const decayFactor     = {decay_factor};
const dt              = {dt};
const totalAnimFrames = {num_anim_frames};
const stimEndFrame    = {stim_end_frame};

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

// Background neuron points (dim, per-neuron color)
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
scene.add(new THREE.Points(bgGeom, new THREE.PointsMaterial({{
  size:0.15, vertexColors:true, transparent:true, opacity:0.3, sizeAttenuation:true
}})));

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

// RASTER CANVAS
const rasterCanvas = document.getElementById('rasterCanvas');
const rasterCtx    = rasterCanvas.getContext('2d');
let rasterBuffer   = document.createElement('canvas');

// Per-neuron hex color for raster dots (input neurons get mel-bin color)
function neuronHex(idx){{
  const c = neuronColors[idx];
  const toHex = v => {{const h=Math.round(v*255).toString(16); return h.length<2?'0'+h:h;}};
  return '#'+toHex(c.r)+toHex(c.g)+toHex(c.b);
}}

function prerenderRaster(){{
  const rc = document.getElementById('raster-container');
  const W = rc.clientWidth, H = rc.clientHeight;
  rasterCanvas.width = W; rasterCanvas.height = H;
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

  // Stimulus end marker
  const lastFrame = frameIndices[totalAnimFrames-1]+1;
  const mx = margin + (stimEndFrame/lastFrame)*plotW;
  ctx.strokeStyle='rgba(255,100,100,0.4)'; ctx.lineWidth=1;
  ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(mx,0); ctx.lineTo(mx,H); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='rgba(255,100,100,0.4)'; ctx.font='9px monospace';
  ctx.fillText('stim end', mx+3, H-6);

  for(let f=0;f<totalAnimFrames;f++){{
    const spikes=spikeData[f];
    const x=margin+(f/totalAnimFrames)*plotW;
    for(let s=0;s<spikes.length;s++){{
      const idx=spikes[s];
      const rasterPos=neuronToRaster[idx];
      const y=(rasterPos/neuronCount)*plotH+2;
      // Use per-neuron color for input neurons, zone color for others
      ctx.fillStyle = neuronZoneIdx[idx]===0 ? neuronHex(idx) : zoneHex[neuronZoneIdx[idx]];
      ctx.fillRect(x,y,1.5,1);
    }}
  }}
}}

function drawRasterFrame(frameIdx){{
  const W=rasterCanvas.width, H=rasterCanvas.height;
  rasterCtx.drawImage(rasterBuffer,0,0);
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
  for(let i=0;i<neuronCount;i++) intensity[i]*=decayFactor;
  const spikes=spikeData[frameIdx];
  for(let s=0;s<spikes.length;s++) intensity[spikes[s]]=1.0;
  const ca=actGeom.attributes.color, sa=actGeom.attributes.size;
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
  ca.needsUpdate=true; sa.needsUpdate=true;
  slider.value=frameIdx;
  timeDisplay.textContent=(frameIndices[frameIdx]*dt).toFixed(1)+' ms';
  drawRasterFrame(frameIdx);
  if(soundEnabled && spikes.length>0){{
    let eC=0,iC=0;
    for(let s=0;s<spikes.length;s++){{ if(isInhibitory[spikes[s]]) iC++; else eC++; }}
    playSpikeAudio(eC,iC);
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

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D activity visualization')
    parser.add_argument('--sample', default=DEFAULT_SAMPLE,
                        help='BSA spike train filename (default: %(default)s)')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: results/activity_vis/lsm_activity.html)')
    parser.add_argument('--dump-dir', default=None,
                        help='Spike dump directory (skip sim if it already has data)')
    args = parser.parse_args()

    sample_path = BSA_DIR / args.sample
    output_path = Path(args.output) if args.output else OUT_DIR / 'lsm_activity.html'
    dump_dir = Path(args.dump_dir) if args.dump_dir else OUT_DIR / 'dump'
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Skip simulation if dump already exists
    if (dump_dir / 'spikes.csv').exists() and (dump_dir / 'meta.json').exists():
        print(f"Using existing dump in {dump_dir}")
    else:
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample not found: {sample_path}")
        run_raster_dump(sample_path, dump_dir)

    build_html(SNAPSHOT, dump_dir, output_path)
    print(f"\nDone! Open in browser:\n  file://{output_path.resolve()}")


if __name__ == '__main__':
    main()
