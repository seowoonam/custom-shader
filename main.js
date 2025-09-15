// WebGL2 Halftone with custom shape texture + lil-gui controls

const dpr = Math.min(window.devicePixelRatio || 1, 2);
const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true, alpha: true });
if (!gl) {
  alert('WebGL2 not supported.');
  throw new Error('WebGL2 not supported');
}

// Shaders
const vertSrc = `#version 300 es
layout(location = 0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const fragSrc = `#version 300 es
precision highp float;
out vec4 outColor;

uniform vec2 u_resolution;
uniform float u_cellSize;     // in pixels
uniform float u_angle;        // radians
uniform float u_minScale;     // 0..1
uniform float u_maxScale;     // 0..1
uniform bool u_invert;

uniform sampler2D u_shapeTex; // expects alpha shape, transparent bg
uniform sampler2D u_sourceTex; // optional image to derive brightness
uniform bool u_hasSource;
uniform vec2 u_sourceSize;     // source image size in pixels

uniform vec3 u_colorLight;    // paper (unused when transparent bg)
uniform vec3 u_colorDark;     // ink
uniform float u_maskThreshold; // 0..1 alpha threshold for masking (used as epsilon)
uniform float u_curve;         // dot growth curve exponent
uniform float u_maxSizePx;     // absolute max size in pixels (cap scale)
uniform float u_paperOpacity;  // 0..1 background paper opacity
uniform float u_edgeFalloffPx; // pixels of shrink near edge
uniform bool u_colorFromSource; // if true, use source color for ink
uniform float u_density;        // stamps per base cell (1.0 = default)
uniform bool u_usePalette;      // if true, pick from palette by luminance
uniform vec3 u_palette[3];      // exactly three ink colors
uniform int u_paletteCount;
uniform int u_scaleSteps;       // quantize scale into steps (>1 enables)
uniform bool u_pixelateEnable;  // quantize shape UVs at small scales
uniform float u_pixelBlocksMin; // few blocks at small scale (big pixels)
uniform float u_pixelBlocksMax; // many blocks at large scale (fine)

float luminance(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec2 rotate(vec2 p, float a) {
  float s = sin(a), c = cos(a);
  return vec2(c*p.x - s*p.y, s*p.x + c*p.y);
}

void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = fragCoord / u_resolution; // 0..1

  // Brightness and mask are sampled at the cell center below

  // Rotate space around center to align the grid
  vec2 centered = fragCoord - 0.5 * u_resolution;
  vec2 rotated = rotate(centered, -u_angle) + 0.5 * u_resolution;

  // Compute cell index and world-space cell center
  float effCell = u_cellSize / max(u_density, 1e-4);
  vec2 grid = rotated / effCell;
  vec2 cellIndex = floor(grid);
  vec2 cellCenterRot = (cellIndex + 0.5) * effCell;
  vec2 cellCenterWorld = rotate(cellCenterRot - 0.5 * u_resolution, u_angle) + 0.5 * u_resolution;
  vec2 uvCenter = cellCenterWorld / u_resolution;

  // Per-cell local coords [0,1)
  vec2 cellUV = fract(grid);
  vec2 local = cellUV - 0.5; // center at 0

  // Sample brightness and mask at the cell center
  float scale;
  float srcA;
  {
    if (u_hasSource) {
      // Map canvas UV -> source UV (letterbox contain)
      vec2 canvasPx = uvCenter * u_resolution;
      float canvasAspect = u_resolution.x / u_resolution.y;
      float sourceAspect = u_sourceSize.x / max(1.0, u_sourceSize.y);
      vec2 contentSizePx;
      vec2 offsetPx;
      if (canvasAspect > sourceAspect) {
        contentSizePx = vec2(u_resolution.y * sourceAspect, u_resolution.y);
        offsetPx = vec2(0.5 * (u_resolution.x - contentSizePx.x), 0.0);
      } else {
        contentSizePx = vec2(u_resolution.x, u_resolution.x / sourceAspect);
        offsetPx = vec2(0.0, 0.5 * (u_resolution.y - contentSizePx.y));
      }
      vec2 srcUV = (canvasPx - offsetPx) / contentSizePx;
      vec2 uvCenterFlip = vec2(srcUV.x, 1.0 - srcUV.y);
      vec4 srcC = texture(u_sourceTex, uvCenterFlip);
      float inRect = float(all(greaterThanEqual(srcUV, vec2(0.0))) && all(lessThanEqual(srcUV, vec2(1.0))));
      srcA = srcC.a * inRect;
      float grad = length(vec2(dFdx(srcA), dFdy(srcA)));
      float distPx = (srcA - 0.5) / max(1e-5, grad);
      float edgeFactor = clamp(distPx / max(1e-3, u_edgeFalloffPx), 0.0, 1.0);
      float baseB = luminance(srcC.rgb);
      float bMixed = mix(1.0, baseB, srcA);
      if (u_invert) bMixed = 1.0 - bMixed;
      float t = pow(1.0 - bMixed, max(0.001, u_curve));
      scale = mix(u_minScale, u_maxScale, t) * edgeFactor;
    } else {
      vec2 p = uv * 2.0 - 1.0;
      float r = length(p);
      float bf = clamp(1.0 - r, 0.0, 1.0);
      if (u_invert) bf = 1.0 - bf;
      float t = pow(1.0 - bf, max(0.001, u_curve));
      scale = mix(u_minScale, u_maxScale, t);
      srcA = 1.0;
    }
  }
  // Clamp maximum apparent size in pixels (based on effective cell size)
  scale = min(scale, u_maxSizePx / max(1.0, effCell));
  // Optional quantization of scale to reduce tiny incremental changes
  if (u_scaleSteps > 1) {
    float steps = float(u_scaleSteps);
    scale = floor(scale * steps) / steps;
  }
  if (srcA <= u_maskThreshold || scale <= 0.01) {
    outColor = vec4(0.0);
    return;
  }

  // Sample the shape texture scaled about the cell center
  // Accumulate overlapping stamps from 3x3 neighbor cells
  float shapeAlpha = 0.0;
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      vec2 neighborLocal = local + vec2(float(i), float(j));
      vec2 shapeUV = neighborLocal / scale + 0.5;
      if (u_pixelateEnable) {
        float norm = clamp((scale - u_minScale) / max(1e-6, (u_maxScale - u_minScale)), 0.0, 1.0);
        float blocks = mix(u_pixelBlocksMin, u_pixelBlocksMax, norm);
        blocks = max(1.0, blocks);
        shapeUV = (floor(shapeUV * blocks) + 0.5) / blocks;
      }
      if (all(greaterThanEqual(shapeUV, vec2(0.0))) && all(lessThanEqual(shapeUV, vec2(1.0)))) {
        vec2 shapeUVFlip = vec2(shapeUV.x, 1.0 - shapeUV.y);
        shapeAlpha = max(shapeAlpha, texture(u_shapeTex, shapeUVFlip).a);
      }
    }
  }

  // Choose ink color
  vec3 inkRgb = u_colorDark;
  if (u_colorFromSource && u_hasSource) {
    // Reconstruct the same source sample used for center evaluation
    // Note: uvCenterFlip, srcUV, and srcC are defined earlier; replicate color safely
    vec2 canvasPx2 = uvCenter * u_resolution;
    float canvasAspect2 = u_resolution.x / u_resolution.y;
    float sourceAspect2 = u_sourceSize.x / max(1.0, u_sourceSize.y);
    vec2 contentSizePx2;
    vec2 offsetPx2;
    if (canvasAspect2 > sourceAspect2) {
      contentSizePx2 = vec2(u_resolution.y * sourceAspect2, u_resolution.y);
      offsetPx2 = vec2(0.5 * (u_resolution.x - contentSizePx2.x), 0.0);
    } else {
      contentSizePx2 = vec2(u_resolution.x, u_resolution.x / sourceAspect2);
      offsetPx2 = vec2(0.0, 0.5 * (u_resolution.y - contentSizePx2.y));
    }
    vec2 srcUV2 = (canvasPx2 - offsetPx2) / contentSizePx2;
    vec2 uvCenterFlip2 = vec2(srcUV2.x, 1.0 - srcUV2.y);
    vec4 srcC2 = texture(u_sourceTex, uvCenterFlip2);
    float inRect2 = float(all(greaterThanEqual(srcUV2, vec2(0.0))) && all(lessThanEqual(srcUV2, vec2(1.0))));
    vec3 srcRgbMixed = mix(vec3(1.0), srcC2.rgb, srcC2.a * inRect2);
    inkRgb = srcRgbMixed;
  } else if (u_usePalette && u_paletteCount > 0) {
    // Choose palette color by (non-inverted) luminance at cell center
    vec2 canvasPx2 = uvCenter * u_resolution;
    float canvasAspect2 = u_resolution.x / u_resolution.y;
    float sourceAspect2 = u_sourceSize.x / max(1.0, u_sourceSize.y);
    vec2 contentSizePx2;
    vec2 offsetPx2;
    if (canvasAspect2 > sourceAspect2) {
      contentSizePx2 = vec2(u_resolution.y * sourceAspect2, u_resolution.y);
      offsetPx2 = vec2(0.5 * (u_resolution.x - contentSizePx2.x), 0.0);
    } else {
      contentSizePx2 = vec2(u_resolution.x, u_resolution.x / sourceAspect2);
      offsetPx2 = vec2(0.0, 0.5 * (u_resolution.y - contentSizePx2.y));
    }
    vec2 srcUV2 = (canvasPx2 - offsetPx2) / contentSizePx2;
    vec2 uvCenterFlip2 = vec2(srcUV2.x, 1.0 - srcUV2.y);
    vec4 srcC2 = texture(u_sourceTex, uvCenterFlip2);
    float inRect2 = float(all(greaterThanEqual(srcUV2, vec2(0.0))) && all(lessThanEqual(srcUV2, vec2(1.0))));
    vec3 srcRgbMixed2 = mix(vec3(1.0), srcC2.rgb, srcC2.a * inRect2);
    float lum = luminance(srcRgbMixed2);
    // Map luminance to one of N buckets (N = u_paletteCount, max 3)
    float step = 1.0 / float(u_paletteCount);
    int idx = int(clamp(floor((1.0 - lum) / step), 0.0, float(u_paletteCount - 1)));
    inkRgb = u_palette[idx];
  }

  // Composite paper under ink with straight-alpha
  float aInk = shapeAlpha;
  float aPaper = clamp(u_paperOpacity, 0.0, 1.0) * (1.0 - aInk);
  float outA = aInk + aPaper;
  vec3 outRgb = inkRgb * aInk + u_colorLight * aPaper;
  outColor = vec4(outRgb, outA);
}`;

function createShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Shader compile error: ' + info);
  }
  return shader;
}

function createProgram(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error('Program link error: ' + info);
  }
  return prog;
}

// Geometry: fullscreen big triangle
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(
  gl.ARRAY_BUFFER,
  new Float32Array([
    -1, -1,
     3, -1,
    -1,  3,
  ]),
  gl.STATIC_DRAW
);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// Program
const vs = createShader(gl, gl.VERTEX_SHADER, vertSrc);
const fs = createShader(gl, gl.FRAGMENT_SHADER, fragSrc);
const program = createProgram(gl, vs, fs);

// Uniform locations
const loc = {
  u_resolution: gl.getUniformLocation(program, 'u_resolution'),
  u_cellSize: gl.getUniformLocation(program, 'u_cellSize'),
  u_angle: gl.getUniformLocation(program, 'u_angle'),
  u_minScale: gl.getUniformLocation(program, 'u_minScale'),
  u_maxScale: gl.getUniformLocation(program, 'u_maxScale'),
  u_invert: gl.getUniformLocation(program, 'u_invert'),
  u_shapeTex: gl.getUniformLocation(program, 'u_shapeTex'),
  u_sourceTex: gl.getUniformLocation(program, 'u_sourceTex'),
  u_hasSource: gl.getUniformLocation(program, 'u_hasSource'),
  u_sourceSize: gl.getUniformLocation(program, 'u_sourceSize'),
  u_colorLight: gl.getUniformLocation(program, 'u_colorLight'),
  u_colorDark: gl.getUniformLocation(program, 'u_colorDark'),
  u_maskThreshold: gl.getUniformLocation(program, 'u_maskThreshold'),
  u_curve: gl.getUniformLocation(program, 'u_curve'),
  u_maxSizePx: gl.getUniformLocation(program, 'u_maxSizePx'),
  u_paperOpacity: gl.getUniformLocation(program, 'u_paperOpacity'),
  u_edgeFalloffPx: gl.getUniformLocation(program, 'u_edgeFalloffPx'),
  u_colorFromSource: gl.getUniformLocation(program, 'u_colorFromSource'),
  u_density: gl.getUniformLocation(program, 'u_density'),
  u_usePalette: gl.getUniformLocation(program, 'u_usePalette'),
  u_palette: gl.getUniformLocation(program, 'u_palette'),
  u_paletteCount: gl.getUniformLocation(program, 'u_paletteCount'),
  u_scaleSteps: gl.getUniformLocation(program, 'u_scaleSteps'),
  u_pixelateEnable: gl.getUniformLocation(program, 'u_pixelateEnable'),
  u_pixelBlocksMin: gl.getUniformLocation(program, 'u_pixelBlocksMin'),
  u_pixelBlocksMax: gl.getUniformLocation(program, 'u_pixelBlocksMax'),
};

// Textures
function createTexture(minFilter = gl.LINEAR, magFilter = gl.LINEAR) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

const shapeTex = createTexture(gl.NEAREST, gl.NEAREST);
const sourceTex = createTexture(gl.LINEAR, gl.LINEAR);

// Default shape: generated crisp circle alpha
function generateDefaultShape(size = 128) {
  const cvs = document.createElement('canvas');
  cvs.width = size;
  cvs.height = size;
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#000';
  const r = size * 0.4;
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, r, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  return cvs;
}

function uploadCanvasToTexture(tex, canvasEl) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvasEl);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadImageBitmapToTexture(tex, imageBitmap) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, imageBitmap.width, imageBitmap.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, imageBitmap);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

// Initialize textures
uploadCanvasToTexture(shapeTex, generateDefaultShape(256));

// Source fallback: soft gradient
function generateDefaultSource(size = 1024) {
  const cvs = document.createElement('canvas');
  cvs.width = size;
  cvs.height = size;
  const ctx = cvs.getContext('2d');
  const grd = ctx.createLinearGradient(0, 0, size, size);
  grd.addColorStop(0, '#ffffff');
  grd.addColorStop(1, '#000000');
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, size, size);
  return cvs;
}
uploadCanvasToTexture(sourceTex, generateDefaultSource(1024));

// State and GUI
const settings = {
  cellSize: 10,
  angleDeg: 0,
  minScale: 0.1,
  maxScale: 1.0,
  invert: false,
  paper: '#ffffff',
  ink: '#111111',
  hasSource: false,
  curve: 1.0,
  maxSizePx: 64,
  paperOpacity: 0.0,
  edgeFalloffPx: 8,
  colorFromSource: false,
  density: 1.0,
  usePalette: false,
  palette1: '#000000',
  palette2: '#ff0000',
  palette3: '#ffffff',
  scaleSteps: 6,
  pixelateEnable: true,
  pixelBlocksMin: 3,
  pixelBlocksMax: 12,
};

// GUI
const guiWrap = document.getElementById('gui');
const gui = new window.lil.GUI({ container: guiWrap, title: 'Halftone' });
gui.add(settings, 'cellSize', 4, 128, 1).name('Cell Size (px)');
gui.add(settings, 'angleDeg', -90, 90, 1).name('Angle (deg)');
gui.add(settings, 'minScale', 0.0, 1.0, 0.01).name('Min Scale');
gui.add(settings, 'maxScale', 0.0, 1.5, 0.01).name('Max Scale');
gui.add(settings, 'invert').name('Invert Brightness');
gui.addColor(settings, 'paper').name('Paper Color');
gui.addColor(settings, 'ink').name('Ink Color');
gui.add(settings, 'curve', 0.2, 4.0, 0.05).name('Scale Curve');
gui.add(settings, 'maxSizePx', 2, 256, 1).name('Max Size (px)');
gui.add(settings, 'paperOpacity', 0.0, 1.0, 0.01).name('Paper Opacity');
gui.add(settings, 'edgeFalloffPx', 0, 64, 1).name('Edge Falloff (px)');
gui.add(settings, 'colorFromSource').name('Color Mode: Source');
gui.add(settings, 'density', 0.5, 4.0, 0.1).name('Density (stamps/cell)');
gui.add(settings, 'usePalette').name('Palette Mode');
gui.addColor(settings, 'palette1').name('Palette Color 1');
gui.addColor(settings, 'palette2').name('Palette Color 2');
gui.addColor(settings, 'palette3').name('Palette Color 3');
gui.add(settings, 'scaleSteps', 1, 12, 1).name('Scale Steps');
gui.add(settings, 'pixelateEnable').name('Pixelate Interim');
gui.add(settings, 'pixelBlocksMin', 1, 16, 1).name('Pixel Blocks Min');
gui.add(settings, 'pixelBlocksMax', 2, 32, 1).name('Pixel Blocks Max');

// File inputs
const fileShape = document.getElementById('file-shape');
const fileImage = document.getElementById('file-image');
document.getElementById('btn-shape').addEventListener('click', () => fileShape.click());
document.getElementById('btn-image').addEventListener('click', () => fileImage.click());
document.getElementById('btn-save').addEventListener('click', () => savePNG());

fileShape.addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    if (f.type === 'image/svg+xml') {
      const svgText = await f.text();
      const bitmap = await rasterizeSVG(svgText, 512);
      uploadImageBitmapToTexture(shapeTex, bitmap);
      bitmap.close?.();
    } else {
      const bmp = await createImageBitmap(f, { imageOrientation: 'from-image', premultiplyAlpha: 'none' });
      uploadImageBitmapToTexture(shapeTex, bmp);
      bmp.close?.();
    }
  } catch (err) {
    console.error(err);
    alert('Failed to load shape (SVG/PNG).');
  } finally {
    e.target.value = '';
  }
});

fileImage.addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    if (f.type === 'image/svg+xml') {
      const svgText = await f.text();
      const bitmap = await rasterizeSVG(svgText, 2048);
      uploadImageBitmapToTexture(sourceTex, bitmap);
      settings.hasSource = true;
      window.__sourceSize = { x: bitmap.width, y: bitmap.height };
      bitmap.close?.();
    } else {
      const bmp = await createImageBitmap(f, { imageOrientation: 'from-image', premultiplyAlpha: 'none' });
      uploadImageBitmapToTexture(sourceTex, bmp);
      settings.hasSource = true;
      window.__sourceSize = { x: bmp.width, y: bmp.height };
      bmp.close?.();
    }
  } catch (err) {
    console.error(err);
    alert('Failed to load source image.');
  } finally {
    e.target.value = '';
  }
});

function hexToRgb(hex) {
  const s = String(hex).replace('#', '');
  const bigint = parseInt(s.length === 3 ? s.split('').map((c)=>c+c).join('') : s, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return [r / 255, g / 255, b / 255];
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
}

window.addEventListener('resize', resize);
resize();

function render() {
  resize();
  gl.clearColor(0.0, 0.0, 0.0, 0.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(program);
  gl.bindVertexArray(vao);

  gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);
  gl.uniform1f(loc.u_cellSize, Math.max(1, settings.cellSize) * dpr);
  gl.uniform1f(loc.u_angle, (settings.angleDeg * Math.PI) / 180);
  gl.uniform1f(loc.u_minScale, Math.min(settings.minScale, settings.maxScale));
  gl.uniform1f(loc.u_maxScale, Math.max(settings.minScale, settings.maxScale));
  gl.uniform1i(loc.u_invert, settings.invert ? 1 : 0);
  gl.uniform1f(loc.u_maskThreshold, 0.5);
  gl.uniform1f(loc.u_curve, settings.curve);
  gl.uniform1f(loc.u_maxSizePx, settings.maxSizePx * dpr);
  gl.uniform1f(loc.u_edgeFalloffPx, settings.edgeFalloffPx * dpr);
  gl.uniform1i(loc.u_colorFromSource, settings.colorFromSource ? 1 : 0);
  gl.uniform1f(loc.u_density, settings.density);
  gl.uniform1i(loc.u_usePalette, settings.usePalette ? 1 : 0);
  const pal = [settings.palette1, settings.palette2, settings.palette3].map(hexToRgb);
  gl.uniform3fv(loc.u_palette, new Float32Array(pal.flat()));
  gl.uniform1i(loc.u_paletteCount, 3);
  gl.uniform1i(loc.u_scaleSteps, Math.max(1, Math.floor(settings.scaleSteps)));
  gl.uniform1i(loc.u_pixelateEnable, settings.pixelateEnable ? 1 : 0);
  gl.uniform1f(loc.u_pixelBlocksMin, settings.pixelBlocksMin);
  gl.uniform1f(loc.u_pixelBlocksMax, Math.max(settings.pixelBlocksMin, settings.pixelBlocksMax));

  const paper = hexToRgb(settings.paper);
  const ink = hexToRgb(settings.ink);
  gl.uniform3f(loc.u_colorLight, paper[0], paper[1], paper[2]);
  gl.uniform3f(loc.u_colorDark, ink[0], ink[1], ink[2]);
  gl.uniform1f(loc.u_paperOpacity, settings.paperOpacity);

  // Bind textures
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, shapeTex);
  gl.uniform1i(loc.u_shapeTex, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, sourceTex);
  gl.uniform1i(loc.u_sourceTex, 1);
  gl.uniform1i(loc.u_hasSource, settings.hasSource ? 1 : 0);
  const s = window.__sourceSize || { x: 0, y: 0 };
  gl.uniform2f(loc.u_sourceSize, s.x, s.y);

  gl.drawArrays(gl.TRIANGLES, 0, 3);

  requestAnimationFrame(render);
}
requestAnimationFrame(render);

function savePNG() {
  try {
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'halftone.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
    console.error(e);
    alert('Unable to save PNG.');
  }
}

async function rasterizeSVG(svgText, targetSize = 512) {
  // Parse width/height or viewBox for sizing
  const parser = new DOMParser();
  const doc = parser.parseFromString(svgText, 'image/svg+xml');
  const svgEl = doc.documentElement;
  let vb = svgEl.getAttribute('viewBox');
  let w = parseFloat(svgEl.getAttribute('width') || '0');
  let h = parseFloat(svgEl.getAttribute('height') || '0');
  if (vb) {
    const parts = vb.split(/\s+/).map(Number);
    if (parts.length === 4) {
      w = parts[2];
      h = parts[3];
    }
  }
  if (!(w > 0 && h > 0)) {
    w = h = targetSize;
  }
  const scale = targetSize / Math.max(w, h);
  const outW = Math.max(1, Math.round(w * scale));
  const outH = Math.max(1, Math.round(h * scale));

  // Create a blob URL for the possibly sanitized SVG
  const serializer = new XMLSerializer();
  const clean = serializer.serializeToString(doc);
  const blob = new Blob([clean], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.decoding = 'async';
  img.crossOrigin = 'anonymous';
  await new Promise((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (e) => reject(e);
    img.src = url;
  });
  URL.revokeObjectURL(url);

  // Draw to an offscreen canvas with transparent background
  const cvs = document.createElement('canvas');
  cvs.width = outW;
  cvs.height = outH;
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, outW, outH);
  ctx.drawImage(img, 0, 0, outW, outH);

  // Turn into ImageBitmap (no premultiply, no flip)
  const bitmap = await createImageBitmap(cvs, { imageOrientation: 'none', premultiplyAlpha: 'none' });
  return bitmap;
}


