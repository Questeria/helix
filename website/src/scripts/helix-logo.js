// =====================================================================
// Helix logo — 3D double helix, carbon ↔ silicon
// Vertical spiral. Carbon strand: matte graphite. Silicon strand:
// polished iridescent wafer. Rungs interleave; group rotates slowly.
// =====================================================================

import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export function mountHelix(container, opts = {}) {
  const {
    height = container.clientHeight || 540,
    width  = container.clientWidth  || 540,
    autoRotate = true,
  } = opts;

  // ---------- Scene ----------
  const scene = new THREE.Scene();
  scene.background = null;

  // ---------- Camera ----------
  const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
  camera.position.set(0, 0, 14.5);
  camera.lookAt(0, 0, 0);

  // ---------- Renderer ----------
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  container.appendChild(renderer.domElement);

  // Environment map — critical for metallic silicon to actually reflect something
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  // ---------- Lighting ----------
  // Keep ambient very low so the strand reads near-black in shadow.
  // The rainbow ribbon will only appear where direct lights hit.
  const hemi = new THREE.HemisphereLight(0xfaf4e6, 0x18171f, 0.10);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(4, 6, 6);
  scene.add(key);

  const rim = new THREE.DirectionalLight(0xffffff, 2.0); // neutral rim — was violet, was leaving a purple streak on the strand
  rim.position.set(-5, 2, -3);
  scene.add(rim);

  const silSpot = new THREE.PointLight(0xffffff, 1.6, 30, 1.5);
  silSpot.position.set(3, 4, 4);
  scene.add(silSpot);

  // (cyan fill removed — it was washing the strand in soft light
  // and preventing the discrete hotspot look.)

  // ---------- Helix geometry ----------
  // Two strand curves: helix1 at phase 0, helix2 at phase π.
  // Elongated, more open spiral so the carbon/silicon read clearly.
  const TURNS    = 1.5;
  const RADIUS   = 1.20;
  const HEIGHT   = 6.6;
  const SAMPLES  = 280;
  const TUBE_R   = 0.115;

  function strandPoints(phase) {
    const pts = [];
    for (let i = 0; i <= SAMPLES; i++) {
      const t = i / SAMPLES;
      const theta = t * TURNS * Math.PI * 2 + phase;
      const y = (t - 0.5) * HEIGHT;
      pts.push(new THREE.Vector3(
        Math.cos(theta) * RADIUS,
        y,
        Math.sin(theta) * RADIUS,
      ));
    }
    return pts;
  }

  const curveA = new THREE.CatmullRomCurve3(strandPoints(0));
  const curveB = new THREE.CatmullRomCurve3(strandPoints(Math.PI));

  const tubeGeoA = new THREE.TubeGeometry(curveA, SAMPLES, TUBE_R, 18, false);
  const tubeGeoB = new THREE.TubeGeometry(curveB, SAMPLES, TUBE_R, 18, false);

  // ---------- Materials ----------
  // Rainbow = where the bright white reflections land. Remove the UV spot
  // mask entirely. Instead: paint a HIGH-FREQUENCY thickness map that
  // cycles the full visible spectrum many times along the strand. Wherever
  // a specular highlight lands, the thickness underneath spans the full
  // 380–700nm range → the highlight contains the complete rainbow.
  const TMAP_W = 4096, TMAP_H = 16;

  // Mutable so the slider can regenerate the texture on demand.
  let thicknessCycles = 3;
  let thicknessSeparation = 0.55; // 0 = continuous rainbow, 1 = pure gap. Bigger = bigger dark gap between bands.
  let rainbowBrightness = 2.0; // peak iridescence in the bands
  let rainbowGlow = 0.06;      // emissive boost on the rainbow bands

  function makeThicknessMap(cycles, separation) {
    const c = document.createElement('canvas');
    c.width = TMAP_W; c.height = TMAP_H;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(TMAP_W, TMAP_H);
    const data = img.data;
    // Each cycle: thickness DWELLS at 0 for the first portion of the band
    // (red end), transitions through the middle, then DWELLS at 1 for the
    // last portion (blue end). This makes the strand spend most of each
    // visible band at the spectrum extremes — red and blue dominate, the
    // green/yellow middle is just a thin seam between.
    const halfBand = Math.max(0.02, 0.5 * (1 - separation));
    const bandStart = 0.5 - halfBand;
    const bandWidth = 2 * halfBand;
    // Inside each band: u = 0..1
    //   u in [0, 0.40]   : v = 0       (red plateau)
    //   u in [0.40, 0.60]: v = smoothstep transition through middle
    //   u in [0.60, 1.0] : v = 1       (blue plateau)
    const plateauL = 0.40;
    const plateauR = 0.60;
    for (let x = 0; x < TMAP_W; x++) {
      const t = (x / TMAP_W) * cycles;
      const frac = t - Math.floor(t);
      let v;
      if (frac < bandStart || frac > bandStart + bandWidth) {
        v = 0;
      } else {
        const u = (frac - bandStart) / bandWidth;
        if (u < plateauL) {
          v = 0;
        } else if (u > plateauR) {
          v = 1;
        } else {
          const k = (u - plateauL) / (plateauR - plateauL);
          v = k * k * (3 - 2 * k); // smoothstep
        }
      }
      const byte = Math.floor(v * 255);
      for (let y = 0; y < TMAP_H; y++) {
        const i = (y * TMAP_W + x) * 4;
        data[i] = byte; data[i + 1] = byte; data[i + 2] = byte; data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  // Iridescence MASK — same cycles, but a DUTY-CYCLE pulse so iridescence
  // is ON in bright bands (rainbow section) and OFF in dark gaps. The
  // separation slider controls how narrow each band is. The brightness
  // slider controls the peak iridescence inside each band.
  function makeIridescenceMap(cycles, separation, brightness) {
    const c = document.createElement('canvas');
    c.width = TMAP_W; c.height = TMAP_H;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(TMAP_W, TMAP_H);
    const data = img.data;
    // ON window centered at frac=0.5, width 2*halfBand. separation=0 → full
    // duty (no gap); separation=1 → nearly all gap. Same halfBand as the
    // thickness map, so the bright band aligns exactly with the sweep.
    const halfBand = Math.max(0.02, 0.5 * (1 - separation));
    const peak = Math.min(1, brightness);
    const bandStart = 0.5 - halfBand;
    const bandWidth = 2 * halfBand;
    // Edge fade keeps the band boundaries soft (no hard cutoff into the
    // gap). 8% of the band width on each side fades in/out smoothly.
    const fade = 0.08;
    for (let x = 0; x < TMAP_W; x++) {
      const t = (x / TMAP_W) * cycles;
      const frac = t - Math.floor(t);
      let v;
      if (frac < bandStart || frac > bandStart + bandWidth) {
        v = 0;
      } else {
        // u = 0..1 across the band. The thickness map sweeps 0..1 over the
        // same range, so u=0 is one spectrum extreme (red), u=1 is the other
        // (violet/blue). U-shaped envelope: bright at the extremes, dim in
        // the middle. This makes the red & blue ends stand out and pushes
        // the green/yellow middle into the background.
        const u = (frac - bandStart) / bandWidth;
        // Match the thickness-map plateaus: bright on the red plateau
        // (u in [0, 0.4]) and the blue plateau (u in [0.6, 1]), dim in the
        // middle (u in [0.4, 0.6]) where green/yellow lives.
        const plateauL = 0.40;
        const plateauR = 0.60;
        const dipDepth = 0.75; // 0 = no dim, 1 = fully off in middle
        let env;
        if (u < plateauL || u > plateauR) {
          env = 1.0; // plateau — full brightness for red / blue
        } else {
          const k = (u - plateauL) / (plateauR - plateauL); // 0..1
          // Smooth dip: high at the plateau edges, lowest at center (k=0.5).
          const bell = 4 * k * (1 - k); // 0..1..0
          env = 1.0 - dipDepth * bell;
        }
        // Soft fade at the very edges of the band so it blends into the gap.
        const edgeL = Math.min(1, u / fade);
        const edgeR = Math.min(1, (1 - u) / fade);
        const edge = Math.min(edgeL, edgeR);
        const edgeS = edge * edge * (3 - 2 * edge); // smoothstep
        v = peak * env * edgeS;
      }
      const byte = Math.floor(v * 255);
      for (let y = 0; y < TMAP_H; y++) {
        const i = (y * TMAP_W + x) * 4;
        data[i] = byte; data[i + 1] = byte; data[i + 2] = byte; data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  // COLORED emissive map — same band envelope as the iridescence mask, but
  // each band is painted with the actual rainbow spectrum (hue cycles 0→360
  // across each band). Used as emissiveMap so glow lights up in the
  // bands' real colors, not white.
  function hslToRgb(h, s, l) {
    h = ((h % 1) + 1) % 1;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(-1, Math.min(Math.min(k - 3, 9 - k), 1));
    };
    return [f(0), f(8), f(4)];
  }
  function makeRainbowEmissiveMap(cycles, separation, brightness) {
    const c = document.createElement('canvas');
    c.width = TMAP_W; c.height = TMAP_H;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(TMAP_W, TMAP_H);
    const data = img.data;
    const halfBand = 0.5 * (1 - separation * 0.8);
    const peak = Math.min(1, brightness);
    for (let x = 0; x < TMAP_W; x++) {
      const t = (x / TMAP_W) * cycles;
      const frac = t - Math.floor(t);
      const dist = Math.abs(frac - 0.5);
      let v;
      if (dist > halfBand) {
        v = 0;
      } else {
        const k = 1 - dist / halfBand;
        v = peak * (k * k * (3 - 2 * k));
      }
      // Hue sweeps full spectrum across each band (frac 0..1 → hue 0..1).
      // Shift so band center (frac=0.5) lands roughly mid-spectrum.
      const hue = frac;
      const [r, g, b] = hslToRgb(hue, 1.0, 0.55);
      const rb = Math.floor(r * v * 255);
      const gb = Math.floor(g * v * 255);
      const bb = Math.floor(b * v * 255);
      for (let y = 0; y < TMAP_H; y++) {
        const i = (y * TMAP_W + x) * 4;
        data[i] = rb; data[i + 1] = gb; data[i + 2] = bb; data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }
  const carbonMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x141114),
    metalness: 0.45,
    roughness: 0.62,
    clearcoat: 0.55,
    clearcoatRoughness: 0.45,
    sheen: 0.5,
    sheenRoughness: 0.7,
    sheenColor: new THREE.Color(0x3a3038),
  });

  // Silicon — strand B: minimal working material. Restoring after a recent
  // iteration produced a silently-broken shader compile. Build back up from
  // here once we've confirmed the helix is rendering again.
  // Trick: drop envMapIntensity to a sliver so the metal doesn't reflect
  // the ambient environment (which spreads light evenly). What's left is
  // the bright DIRECT lights — key, rim, spot — which only hit narrow
  // bands. Iridescence rides those highlights, so the rainbow only fires
  // where direct light actually lands.
  const siliconMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color().setRGB(0.82, 0.82, 0.82),
    metalness: 1.0,
    roughness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.0,
    envMapIntensity: 1.62,
    iridescence: 0.0,
    iridescenceIOR: 3.0,
    iridescenceThicknessRange: [180, 380],
    iridescenceThicknessMap: makeThicknessMap(thicknessCycles, thicknessSeparation),
    iridescenceMap: makeIridescenceMap(thicknessCycles, thicknessSeparation, rainbowBrightness),
    sheen: 0.26,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(1, 1, 1),
    // Glow is computed in shader (world-space). Keep emissive black so the
    // built-in emissive path contributes nothing — our onBeforeCompile adds
    // the world-fixed colored glow on top.
    emissive: new THREE.Color(0, 0, 0),
    emissiveIntensity: 1.0,
  });

  // ---- World-space glow uniform (driven by the Glow slider) ----
  const glowUniform = { value: rainbowGlow };
  const spotSharpUniform = { value: 42.0 };
  const hueStartUniform = { value: 0.00 };
  const hueEndUniform  = { value: 1.00 };
  const spotW1Uniform = { value: 1.72 };
  const spotW2Uniform = { value: 0.71 };
  const spotW3Uniform = { value: 1.32 };
  const spotW4Uniform = { value: 1.07 };
  const spotW5Uniform = { value: 1.14 };
  const spotW6Uniform = { value: 1.24 };
  const spotW7Uniform = { value: 1.46 };
  const spotSepUniform = { value: 0.36 };
  siliconMat.onBeforeCompile = (shader) => {
    shader.uniforms.uGlow = glowUniform;
    shader.uniforms.uSpotSharp = spotSharpUniform;
    shader.uniforms.uHueStart = hueStartUniform;
    shader.uniforms.uHueEnd   = hueEndUniform;
    shader.uniforms.uW1 = spotW1Uniform;
    shader.uniforms.uW2 = spotW2Uniform;
    shader.uniforms.uW3 = spotW3Uniform;
    shader.uniforms.uW4 = spotW4Uniform;
    shader.uniforms.uW5 = spotW5Uniform;
    shader.uniforms.uW6 = spotW6Uniform;
    shader.uniforms.uW7 = spotW7Uniform;
    shader.uniforms.uSep = spotSepUniform;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vGlowN;')
      .replace(
        '#include <worldpos_vertex>',
        '#include <worldpos_vertex>\nvGlowN = normalize(mat3(modelMatrix) * objectNormal);'
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vGlowN;
        uniform float uGlow;
        uniform float uSpotSharp;
        uniform float uHueStart;
        uniform float uHueEnd;
        uniform float uW1;
        uniform float uW2;
        uniform float uW3;
        uniform float uW4;
        uniform float uW5;
        uniform float uW6;
        uniform float uW7;
        uniform float uSep;
        vec3 _rotY(vec3 v, float a) {
          float c = cos(a), s = sin(a);
          return vec3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
        }
        vec3 _hsl2rgb(float h, float s, float l) {
          vec3 k = mod(vec3(0.0, 8.0, 4.0) + h * 12.0, 12.0);
          float a = s * min(l, 1.0 - l);
          return vec3(
            l - a * clamp(min(min(k.x - 3.0, 9.0 - k.x), 1.0), -1.0, 1.0),
            l - a * clamp(min(min(k.y - 3.0, 9.0 - k.y), 1.0), -1.0, 1.0),
            l - a * clamp(min(min(k.z - 3.0, 9.0 - k.z), 1.0), -1.0, 1.0)
          );
        }
      `)
      .replace(
        '#include <emissivemap_fragment>',
        `
        // World-fixed glow: spots are anchored to fixed directions in world
        // space. As the helix rotates, the strand's surface normal passes
        // through these directions, lighting up only the points that face
        // them — like real specular highlights from off-screen lights.
        vec3 n = normalize(vGlowN);
        // Seven world-fixed rainbow spots, evenly distributed in world space.
        // Hues sweep linearly from dark red (0.00) through to purple (0.78),
        // so each direction lights a distinct band of the visible spectrum.
        // Seven spots distributed across the visible FRONT hemisphere so
        // every color lights surfaces the camera can actually see. Earlier
        // versions placed half the spots on the back of the helix, where
        // the blue/purple ones never lit any visible normal.
        vec3 g1 = normalize(vec3(-1.00,  0.25,  0.00)); // red,    far left
        vec3 g2 = normalize(vec3(-0.87,  0.25,  0.50)); // orange, front-left
        vec3 g3 = normalize(vec3(-0.50,  0.30,  0.87)); // yellow, near-front-left
        vec3 g4 = normalize(vec3( 0.00,  0.35,  1.00)); // green,  dead front
        vec3 g5 = normalize(vec3( 0.50,  0.30,  0.87)); // cyan,   near-front-right
        vec3 g6 = normalize(vec3( 0.87,  0.25,  0.50)); // blue,   front-right
        vec3 g7 = normalize(vec3( 1.00,  0.25,  0.00)); // purple, far right
        // Separation: rotate each spot around Y, pivoting on the middle
        // spot (g4). At uSep=1, no rotation -> the default spread. At uSep<1,
        // spots bunch toward g4; at uSep>1, they fan out wider.
        float _sep = (uSep - 1.0) * 0.6;
        g1 = _rotY(g1, -3.0 * _sep);
        g2 = _rotY(g2, -2.0 * _sep);
        g3 = _rotY(g3, -1.0 * _sep);
        g5 = _rotY(g5,  1.0 * _sep);
        g6 = _rotY(g6,  2.0 * _sep);
        g7 = _rotY(g7,  3.0 * _sep);
        float sharp = uSpotSharp;
        float w1 = pow(max(dot(n, g1), 0.0), sharp);
        float w2 = pow(max(dot(n, g2), 0.0), sharp);
        float w3 = pow(max(dot(n, g3), 0.0), sharp);
        float w4 = pow(max(dot(n, g4), 0.0), sharp);
        float w5 = pow(max(dot(n, g5), 0.0), sharp);
        float w6 = pow(max(dot(n, g6), 0.0), sharp);
        float w7 = pow(max(dot(n, g7), 0.0), sharp);
        // Hues spaced equally across the picked spectrum range.
        // Blues / purples get higher L to compensate for the eye's lower\n        // sensitivity to short wavelengths — without it, boosting cyan/blue/\n        // purple strengths is barely visible against the carbon backdrop.
        vec3 c1 = _hsl2rgb(mix(uHueStart, uHueEnd, 0.0),     1.0, 0.50);
        vec3 c2 = _hsl2rgb(mix(uHueStart, uHueEnd, 1.0/6.0), 1.0, 0.55);
        vec3 c3 = _hsl2rgb(mix(uHueStart, uHueEnd, 2.0/6.0), 1.0, 0.55);
        vec3 c4 = _hsl2rgb(mix(uHueStart, uHueEnd, 3.0/6.0), 1.0, 0.55);
        vec3 c5 = _hsl2rgb(mix(uHueStart, uHueEnd, 4.0/6.0), 1.0, 0.70);
        vec3 c6 = _hsl2rgb(mix(uHueStart, uHueEnd, 5.0/6.0), 1.0, 0.72);
        vec3 c7 = _hsl2rgb(mix(uHueStart, uHueEnd, 1.0),     1.0, 0.65);
        // Square the per-spot weights so slider changes are dramatic:
        // 0.5 -> 0.25 (4x reduction), 1 -> 1 (neutral), 2 -> 4 (4x boost),
        // 3 -> 9 (9x boost). 0 still kills the spot completely.
        vec3 glowSum = c1*w1*(uW1*uW1) + c2*w2*(uW2*uW2) + c3*w3*(uW3*uW3) + c4*w4*(uW4*uW4) + c5*w5*(uW5*uW5) + c6*w6*(uW6*uW6) + c7*w7*(uW7*uW7);
        totalEmissiveRadiance += glowSum * uGlow * 6.0;
        `
      );
  };

  const strandA = new THREE.Mesh(tubeGeoA, carbonMat);   // carbon strand
  const strandB = new THREE.Mesh(tubeGeoB, siliconMat);  // silicon strand

  // ---------- Strand top caps (flat discs) ----------
  // Only cap the TOP of each strand. The BOTTOM embeds into the pedestal
  // for a seamless join — no visible disc on the base.
  const flatCapGeo = new THREE.CylinderGeometry(TUBE_R * 1.02, TUBE_R * 1.02, 0.04, 28);
  function makeFlatCap(curve, t, mat) {
    const p = curve.getPoint(t);
    const tan = curve.getTangent(t).normalize();
    const cap = new THREE.Mesh(flatCapGeo, mat);
    cap.position.copy(p);
    cap.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tan);
    return cap;
  }
  const topCapA = makeFlatCap(curveA, 1, carbonMat);
  const topCapB = makeFlatCap(curveB, 1, siliconMat);
  const endCapsA = [topCapA];
  const endCapsB = [topCapB];

  // ---------- Rungs ----------
  // Each rung is half carbon / half silicon.
  // Two cylinder halves meet at the midpoint between strands.
  const rungGroup = new THREE.Group();
  const RUNG_COUNT = 13;
  const halfRungGeo = new THREE.CylinderGeometry(0.045, 0.045, 1, 14, 1);
  // Tiny brushed metallic ferrule at the seam — sells the join.
  const seamGeo = new THREE.CylinderGeometry(0.054, 0.054, 0.04, 16, 1);
  // Glowing ferrule at the seam — where the carbon and silicon halves meet,
  // a tiny rainbow-cycling bead lights up like a fused junction.
  const seamMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0xffffff),
    transparent: false,
    toneMapped: false,
  });
  // Additive halo sprite around each seam for the bloom-like glow.
  const seamHaloTex = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0.00, 'rgba(255,255,255,1)');
    grad.addColorStop(0.20, 'rgba(255,255,255,0.85)');
    grad.addColorStop(0.50, 'rgba(255,255,255,0.25)');
    grad.addColorStop(1.00, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  })();
  const capGeoA = new THREE.SphereGeometry(0.11, 16, 12);
  const capGeoB = new THREE.SphereGeometry(0.11, 16, 12);

  const rungs = []; // for animation

  for (let i = 0; i < RUNG_COUNT; i++) {
    const t = (i + 0.5) / RUNG_COUNT;
    const pA = curveA.getPoint(t);
    const pB = curveB.getPoint(t);

    const dir   = pB.clone().sub(pA);
    const len   = dir.length();
    const dirN  = dir.clone().normalize();
    const halfL = len / 2;
    const mid   = pA.clone().add(pB).multiplyScalar(0.5);
    const quat  = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirN);

    // Carbon half — pA → mid
    const halfA = new THREE.Mesh(halfRungGeo, carbonMat);
    halfA.scale.y = halfL;
    halfA.quaternion.copy(quat);
    halfA.position.copy(pA.clone().add(mid).multiplyScalar(0.5));
    rungGroup.add(halfA);

    // Silicon half — mid → pB
    const halfB = new THREE.Mesh(halfRungGeo, siliconMat);
    halfB.scale.y = halfL;
    halfB.quaternion.copy(quat);
    halfB.position.copy(mid.clone().add(pB).multiplyScalar(0.5));
    rungGroup.add(halfB);

    // Seam ferrule at the midpoint
    const seam = new THREE.Mesh(seamGeo, seamMat);
    seam.quaternion.copy(quat);
    seam.position.copy(mid);
    rungGroup.add(seam);

    // Glow halo — additive sprite around the seam.
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: seamHaloTex,
      color: new THREE.Color(0xffffff),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    }));
    halo.scale.set(0.55, 0.55, 1);
    halo.position.copy(mid);
    rungGroup.add(halo);

    // End caps inherit each strand's material
    const capA = new THREE.Mesh(capGeoA, carbonMat);
    capA.position.copy(pA);
    rungGroup.add(capA);

    const capB = new THREE.Mesh(capGeoB, siliconMat);
    capB.position.copy(pB);
    rungGroup.add(capB);

    rungs.push({ halfA, halfB, capA, capB, seam, halo, t });
  }

  // ---------- Group everything ----------
  // root  — tilts (parallax X)
  //   helix — spins on Y (auto + parallax Y)
  //   base  — static pedestal
  const root = new THREE.Group();
  scene.add(root);

  const helix = new THREE.Group();
  helix.add(strandA);
  helix.add(strandB);
  endCapsA.forEach(m => helix.add(m));
  endCapsB.forEach(m => helix.add(m));
  helix.add(rungGroup);
  root.add(helix);

  // ---------- Pedestal base ----------
  // Strand bottom is embedded into the pedestal so the join is seamless —
  // raise base so its top surface sits ABOVE the strand bottom by ~0.10.
  const baseGroup = new THREE.Group();
  const PED_R   = 1.85;
  const PED_H   = 0.26;
  // strand bottom = -HEIGHT/2; we want pedestal_top = strand_bottom + 0.12
  const baseY   = -HEIGHT / 2 + 0.12 - PED_H / 2;

  // (No lower flange — pedestal sits flush.)

  const pedestalGeo = new THREE.CylinderGeometry(PED_R - 0.04, PED_R + 0.06, PED_H, 96);
  const pedestalMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x111114),
    metalness: 0.7,
    roughness: 0.22,
    clearcoat: 1.0,
    clearcoatRoughness: 0.10,
  });
  const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
  pedestal.position.y = baseY;
  baseGroup.add(pedestal);

  // Top inset disc — slightly recessed, polished metallic
  const topDiscGeo = new THREE.CylinderGeometry(PED_R - 0.18, PED_R - 0.18, 0.02, 96);
  const topDiscMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0x1c1c22),
    metalness: 0.95, roughness: 0.18,
    clearcoat: 1.0, clearcoatRoughness: 0.05,
    envMapIntensity: 1.6,
  });
  const topDisc = new THREE.Mesh(topDiscGeo, topDiscMat);
  topDisc.position.y = baseY + PED_H / 2 - 0.005;
  baseGroup.add(topDisc);

  // Inset top bevel ring — catches light
  const bevelGeo = new THREE.TorusGeometry(PED_R - 0.06, 0.012, 16, 96);
  const bevelMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0x6a6675),
    transparent: true, opacity: 0.85,
  });
  const bevel = new THREE.Mesh(bevelGeo, bevelMat);
  bevel.rotation.x = Math.PI / 2;
  bevel.position.y = baseY + PED_H / 2 - 0.005;
  baseGroup.add(bevel);

  // Violet glow ring at base — echoes the iridescence
  const glowGeo = new THREE.TorusGeometry(PED_R - 0.22, 0.018, 16, 96);
  const glowMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0x9c6cff),
    transparent: true, opacity: 0.55,
  });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.rotation.x = Math.PI / 2;
  glow.position.y = baseY + PED_H / 2 - 0.02;
  baseGroup.add(glow);

  // (Strands embed directly into the pedestal — no surface studs.)

  root.add(baseGroup);

  // Initial sculptural tilt on the root (so base + helix read together)
  root.rotation.x = -0.18;
  root.position.y = 0;

  // ---------- Pointer parallax ----------
  // helix.Y spins (auto + horizontal pointer); root.X tilts (vertical pointer)
  let targetRY = 0;
  let targetRX = -0.18;
  container.addEventListener("pointermove", (e) => {
    const r = container.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width  - 0.5) * 2; // -1..1
    const ny = ((e.clientY - r.top)  / r.height - 0.5) * 2;
    targetRY = nx * 0.35;
    targetRX = -0.18 + ny * 0.14;
  });
  container.addEventListener("pointerleave", () => {
    targetRY = 0;
    targetRX = -0.18;
  });

  // ---------- Resize ----------
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);

  // ---------- Animate ----------
  const clock = new THREE.Clock();
  let baseSpin = 0;
  let rafId = 0;
  let paused = false;

  function tick() {
    const dt = clock.getDelta();
    const t  = clock.getElapsedTime();

    // Constant slow spin
    if (autoRotate) baseSpin += dt * 0.26;

    // helix spins on Y; base spins with it; root tilts on X
    const targetY = targetRY + baseSpin;
    helix.rotation.y     += (targetY - helix.rotation.y) * 0.08;
    baseGroup.rotation.y += (targetY - baseGroup.rotation.y) * 0.08;
    root.rotation.x      += (targetRX - root.rotation.x) * 0.08;

    // Subtle base glow pulse
    glowMat.opacity = 0.45 + 0.18 * Math.sin(t * 1.4);

    // Subtle pulse on rungs — running wave from bottom to top (both halves)
    for (let i = 0; i < rungs.length; i++) {
      const r = rungs[i];
      const pulse = 0.5 + 0.5 * Math.sin(t * 1.6 - r.t * Math.PI * 4);
      const s = 1 + pulse * 0.04;
      r.halfA.scale.x = s; r.halfA.scale.z = s;
      r.halfB.scale.x = s; r.halfB.scale.z = s;

      // Seam glow — cycles through hues over time and along the strand,
      // with a synchronized brightness pulse so each junction breathes.
      const hue = (t * 0.08 + r.t * 1.0) % 1.0;
      const breath = 0.85 + 0.5 * pulse;
      r.halo.material.color.setHSL(hue, 0.75, 0.60);
      r.halo.material.opacity = 0.42 * breath;
      r.halo.scale.set(0.44 * breath, 0.44 * breath, 1);
      r.seam.material.color.setHSL(hue, 0.65, 0.78);
    }

    renderer.render(scene, camera);
    rafId = requestAnimationFrame(tick);
  }
  tick();

  return {
    setTheme(isDark) {
      // Subtle material shift — carbon gets darker in light mode, silicon stays the star;
      // in dark mode silicon glows more.
      if (isDark) {
        renderer.toneMappingExposure = 1.15;
        rim.intensity = 1.4;
      } else {
        renderer.toneMappingExposure = 1.0;
        rim.intensity = 1.0;
      }
    },
    setRainbow(v) {
      const k = Math.max(0, Math.min(1, v));
      const curve = Math.pow(k, 1.6);
      siliconMat.iridescence = curve;
      siliconMat.iridescenceIOR = 1.3 + curve * 1.3;
      siliconMat.needsUpdate = true;
    },
    setSheen(params) {
      // Generic setter. Pass any subset of {iridescence, iridescenceIOR,
      // thicknessMin, thicknessMax, envMapIntensity, roughness,
      // clearcoatRoughness, baseLightness, cycles}.
      if (params.iridescence != null)      siliconMat.iridescence       = +params.iridescence;
      if (params.iridescenceIOR != null)   siliconMat.iridescenceIOR    = +params.iridescenceIOR;
      if (params.envMapIntensity != null)  siliconMat.envMapIntensity   = +params.envMapIntensity;
      if (params.roughness != null)        siliconMat.roughness         = +params.roughness;
      if (params.clearcoatRoughness != null) siliconMat.clearcoatRoughness = +params.clearcoatRoughness;
      if (params.thicknessMin != null || params.thicknessMax != null) {
        const [oldA, oldB] = siliconMat.iridescenceThicknessRange;
        siliconMat.iridescenceThicknessRange = [
          params.thicknessMin != null ? +params.thicknessMin : oldA,
          params.thicknessMax != null ? +params.thicknessMax : oldB,
        ];
      }
      if (params.baseLightness != null) {
        const L = Math.max(0, Math.min(1, +params.baseLightness));
        siliconMat.color.setRGB(L, L, L);
      }
      if (params.cycles != null || params.separation != null || params.brightness != null) {
        if (params.cycles != null) thicknessCycles = Math.max(1, Math.min(200, Math.round(+params.cycles)));
        if (params.separation != null) thicknessSeparation = Math.max(0, Math.min(1, +params.separation));
        if (params.brightness != null) rainbowBrightness = Math.max(0, Math.min(2, +params.brightness));
        const oldT = siliconMat.iridescenceThicknessMap;
        const oldI = siliconMat.iridescenceMap;
        siliconMat.iridescenceThicknessMap = makeThicknessMap(thicknessCycles, thicknessSeparation);
        siliconMat.iridescenceMap = makeIridescenceMap(thicknessCycles, thicknessSeparation, rainbowBrightness);
        if (oldT && oldT.dispose) oldT.dispose();
        if (oldI && oldI.dispose) oldI.dispose();
      }
      if (params.glow != null) {
        rainbowGlow = Math.max(0, Math.min(3, +params.glow));
        glowUniform.value = rainbowGlow;
      }
      if (params.spotSharp != null) {
        spotSharpUniform.value = Math.max(1, Math.min(80, +params.spotSharp));
      }
      if (params.hueStart != null) {
        hueStartUniform.value = Math.max(0, Math.min(1, +params.hueStart));
      }
      if (params.hueEnd != null) {
        hueEndUniform.value = Math.max(0, Math.min(1, +params.hueEnd));
      }
      if (params.spotSep != null) {
        spotSepUniform.value = Math.max(0, Math.min(2.5, +params.spotSep));
      }
      const wUni = [spotW1Uniform, spotW2Uniform, spotW3Uniform, spotW4Uniform, spotW5Uniform, spotW6Uniform, spotW7Uniform];
      for (let i = 1; i <= 7; i++) {
        const k = 'spotW' + i;
        if (params[k] != null) {
          wUni[i - 1].value = Math.max(0, Math.min(3, +params[k]));
        }
      }
      // (emissive no longer driven by brightness — brightness is now the
      // peak of the iridescence map, so only the rainbow bands light up.)
      // Master sheen strength — multiplies iridescence and bumps env reflectivity
      // so a single slider can pull the whole rainbow up/down.
      if (params.sheenStrength != null) {
        const s = Math.max(0, Math.min(2, +params.sheenStrength));
        // Iridescence (helix-following rainbow) is permanently off — only
        // the world-fixed colored glow is used now.
        siliconMat.iridescence = 0;
        // Don't touch envMapIntensity here — the envMapIntensity slider
        // owns that value so seeding both at once doesn't fight.
        siliconMat.sheen = Math.max(0, s - 1);   // 0 below 1, ramps to 1 above
        siliconMat.sheenRoughness = 0.5;
        siliconMat.sheenColor.setRGB(1, 1, 1);
      }
      siliconMat.needsUpdate = true;
    },
    setPaused(v) {
      // Stops/starts the render loop. Used while the canvas is offscreen —
      // pure perf, zero visual change (the clock gap is swallowed on resume).
      const pNew = !!v;
      if (pNew === paused) return;
      paused = pNew;
      if (paused) {
        cancelAnimationFrame(rafId);
      } else {
        clock.getDelta();
        rafId = requestAnimationFrame(tick);
      }
    },
    dispose() {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      renderer.dispose();
      tubeGeoA.dispose(); tubeGeoB.dispose();
      halfRungGeo.dispose(); seamGeo.dispose();
      capGeoA.dispose(); capGeoB.dispose();
      flatCapGeo.dispose();
      carbonMat.dispose(); siliconMat.dispose(); seamMat.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    },
  };
}
