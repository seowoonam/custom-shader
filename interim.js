const dpr = Math.min(window.devicePixelRatio || 1, 2);
const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true, alpha: true });
if (!gl) { alert('WebGL2 not supported'); throw new Error('WebGL2 not supported'); }

const vertSrc = `#version 300 es
layout(location = 0) in vec2 a_position;
void main(){ gl_Position = vec4(a_position,0.0,1.0); }`;

const fragSrc = `#version 300 es
precision highp float;
out vec4 outColor;

uniform vec2 u_resolution;
uniform sampler2D u_shapeTex;
uniform float u_scale;            // 0..1 relative size
uniform bool u_pixelateEnable;
uniform float u_blocks;           // max number of blocks per axis at smallest scale
uniform int u_steps;              // scale quantization steps
uniform vec3 u_ink;
uniform vec3 u_paper;
uniform float u_paperOpacity;
uniform int u_smallestMode;       // 0 square, 1 circle
uniform float u_cornerRadius;     // 0..0.5 rounded corner amount

float sdRoundedBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + vec2(r);
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main(){
  vec2 uv = gl_FragCoord.xy / u_resolution;

  float scale = u_scale;
  if (u_steps > 1) {
    scale = floor(scale * float(u_steps)) / float(u_steps);
  }
  vec2 local = (uv - 0.5) / max(1e-6, scale) + 0.5;
  vec2 localOrig = local; // preserve continuous sampling for large scales

  // Sample original at continuous coords
  float aTex = 0.0;
  if (all(greaterThanEqual(localOrig, vec2(0.0))) && all(lessThanEqual(localOrig, vec2(1.0)))) {
    vec2 flip = vec2(localOrig.x, 1.0 - localOrig.y);
    aTex = texture(u_shapeTex, flip).a;
  }

  // Primitive alpha for smallest size
  vec2 lp0 = localOrig - 0.5;
  float aSquare0;
  {
    float rPrim = clamp(u_cornerRadius, 0.0, 0.5);
    // rounded box SDF: inside <= 0
    vec2 q = abs(lp0) - vec2(0.5 - rPrim);
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - rPrim;
    aSquare0 = float(d <= 0.0);
  }
  float aCircle0 = float(length(lp0) <= 0.5);
  float aPrim = (u_smallestMode == 1) ? aCircle0 : aSquare0;
  float tMorph = clamp(scale, 0.0, 1.0);
  float aSmooth = mix(aPrim, aTex, tMorph);

  // Pixelated interim
  float aPixelated = aSmooth;
  if (u_pixelateEnable) {
    float pixelStrength = 1.0 - tMorph; // more pixelation when smaller
    float blocks = mix(1.0, max(1.0, u_blocks), pixelStrength);
    vec2 pix = localOrig * blocks;
    vec2 pixCenter = (floor(pix) + 0.5) / blocks;
    vec2 inPix = fract(pix) - 0.5; // -0.5..0.5

    // pixel mask
    float aPixMask;
    if (u_smallestMode == 1) {
      float rPix = clamp(u_cornerRadius, 0.0, 0.5) * pixelStrength;
      float d = sdRoundedBox(inPix, vec2(0.5), rPix);
      aPixMask = float(d <= 0.0);
    } else {
      aPixMask = float(all(lessThanEqual(abs(inPix), vec2(0.5))));
    }

    // sample original at pixel center
    float aCenter = 0.0;
    if (all(greaterThanEqual(pixCenter, vec2(0.0))) && all(lessThanEqual(pixCenter, vec2(1.0)))) {
      vec2 flipC = vec2(pixCenter.x, 1.0 - pixCenter.y);
      aCenter = texture(u_shapeTex, flipC).a;
    }
    aPixelated = aCenter > 0.5 ? aPixMask : 0.0;
  }

  float a = mix(aSmooth, aPixelated, u_pixelateEnable ? (1.0 - tMorph) : 0.0);

  // Opaque output
  vec3 rgb = mix(u_paper, u_ink, step(0.5, a));
  outColor = vec4(rgb, 1.0);
}`;

function createShader(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){throw new Error(gl.getShaderInfoLog(s));} return s; }
function createProgram(vs,fs){ const p=gl.createProgram(); gl.attachShader(p,vs); gl.attachShader(p,fs); gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)){throw new Error(gl.getProgramInfoLog(p));} return p; }

const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
gl.bindVertexArray(null);

const program = createProgram(createShader(gl.VERTEX_SHADER,vertSrc), createShader(gl.FRAGMENT_SHADER,fragSrc));
const loc = {
  u_resolution: gl.getUniformLocation(program,'u_resolution'),
  u_shapeTex: gl.getUniformLocation(program,'u_shapeTex'),
  u_scale: gl.getUniformLocation(program,'u_scale'),
  u_pixelateEnable: gl.getUniformLocation(program,'u_pixelateEnable'),
  u_blocks: gl.getUniformLocation(program,'u_blocks'),
  u_steps: gl.getUniformLocation(program,'u_steps'),
  u_ink: gl.getUniformLocation(program,'u_ink'),
  u_paper: gl.getUniformLocation(program,'u_paper'),
  u_paperOpacity: gl.getUniformLocation(program,'u_paperOpacity'),
  u_smallestMode: gl.getUniformLocation(program,'u_smallestMode'),
  u_cornerRadius: gl.getUniformLocation(program,'u_cornerRadius'),
};

function createTexture(){ const t=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,t); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE); gl.bindTexture(gl.TEXTURE_2D,null); return t; }
const shapeTex = createTexture();

function uploadImageBitmapToTexture(tex, imageBitmap){ gl.bindTexture(gl.TEXTURE_2D, tex); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false); gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,imageBitmap.width,imageBitmap.height,0,gl.RGBA,gl.UNSIGNED_BYTE,imageBitmap); gl.bindTexture(gl.TEXTURE_2D,null); }

async function rasterizeSVG(svgText, targetSize = 512){
  const parser = new DOMParser();
  const doc = parser.parseFromString(svgText,'image/svg+xml');
  const svg = doc.documentElement;
  let vb = svg.getAttribute('viewBox');
  let w = parseFloat(svg.getAttribute('width')||'0');
  let h = parseFloat(svg.getAttribute('height')||'0');
  if (vb){ const p=vb.split(/\s+/).map(Number); if(p.length===4){ w=p[2]; h=p[3]; } }
  if(!(w>0&&h>0)){ w=h=targetSize; }
  const scale = targetSize/Math.max(w,h);
  const outW = Math.max(1, Math.round(w*scale));
  const outH = Math.max(1, Math.round(h*scale));
  const ser = new XMLSerializer();
  const blob = new Blob([ser.serializeToString(doc)],{type:'image/svg+xml'});
  const url = URL.createObjectURL(blob);
  const img = new Image(); img.decoding='async'; img.crossOrigin='anonymous';
  await new Promise((res,rej)=>{ img.onload=()=>res(); img.onerror=(e)=>rej(e); img.src=url; });
  URL.revokeObjectURL(url);
  const cvs = document.createElement('canvas'); cvs.width=outW; cvs.height=outH; const ctx=cvs.getContext('2d'); ctx.clearRect(0,0,outW,outH); ctx.drawImage(img,0,0,outW,outH);
  return await createImageBitmap(cvs,{imageOrientation:'none', premultiplyAlpha:'none'});
}

const settings = {
  scale: 0.6,
  pixelateEnable: true,
  blocks: 8,
  steps: 6,
  ink: '#c6555d',
  paper: '#ffffff',
  paperOpacity: 0.0,
  smallestMode: 1,
  cornerRadius: 0.2,
};

const gui = new window.lil.GUI({ container: document.getElementById('gui'), title: 'Interim' });
gui.add(settings,'scale',0.05,1.0,0.01).name('Scale');
gui.add(settings,'steps',1,12,1).name('Scale Steps');
gui.add(settings,'pixelateEnable').name('Pixelate');
gui.add(settings,'blocks',1,32,1).name('Blocks');
gui.addColor(settings,'ink').name('Ink Color');
gui.addColor(settings,'paper').name('Paper Color');
gui.add(settings,'paperOpacity',0,1,0.01).name('Paper Opacity');
gui.add(settings,'smallestMode',{Square:0, Circle:1}).name('Smallest Shape');
gui.add(settings,'cornerRadius',0,0.5,0.01).name('Corner Radius');

const fileShape = document.getElementById('file-shape');
document.getElementById('btn-shape').addEventListener('click',()=>fileShape.click());
document.getElementById('btn-save').addEventListener('click',()=>savePNG());
document.getElementById('btn-save-grid').addEventListener('click',()=>saveGrid());

fileShape.addEventListener('change', async (e)=>{
  const f=e.target.files&&e.target.files[0]; if(!f) return;
  try{
    if(f.type==='image/svg+xml'){ const svg=await f.text(); const bmp=await rasterizeSVG(svg,1024); uploadImageBitmapToTexture(shapeTex,bmp); bmp.close?.(); }
    else{ const bmp=await createImageBitmap(f,{imageOrientation:'from-image', premultiplyAlpha:'none'}); uploadImageBitmapToTexture(shapeTex,bmp); bmp.close?.(); }
  }catch(err){ console.error(err); alert('Failed to load shape'); } finally { e.target.value=''; }
});

function hexToRgb(hex){ const s=String(hex).replace('#',''); const v=parseInt(s.length===3?s.split('').map(c=>c+c).join(''):s,16); return [(v>>16&255)/255,(v>>8&255)/255,(v&255)/255]; }

function resize(){ const r=canvas.getBoundingClientRect(); const w=Math.max(1,Math.floor(r.width*dpr)); const h=Math.max(1,Math.floor(r.height*dpr)); if(canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; } gl.viewport(0,0,canvas.width,canvas.height); }
window.addEventListener('resize', resize); resize();

function render(){
  resize();
  gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program); gl.bindVertexArray(vao);
  gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);
  gl.uniform1f(loc.u_scale, settings.scale);
  gl.uniform1i(loc.u_pixelateEnable, settings.pixelateEnable?1:0);
  gl.uniform1f(loc.u_blocks, settings.blocks);
  gl.uniform1i(loc.u_steps, settings.steps|0);
  gl.uniform1i(loc.u_smallestMode, settings.smallestMode|0);
  gl.uniform1f(loc.u_cornerRadius, settings.cornerRadius);
  const ink=hexToRgb(settings.ink), paper=hexToRgb(settings.paper);
  gl.uniform3f(loc.u_ink, ink[0], ink[1], ink[2]);
  gl.uniform3f(loc.u_paper, paper[0], paper[1], paper[2]);
  gl.uniform1f(loc.u_paperOpacity, settings.paperOpacity);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, shapeTex); gl.uniform1i(loc.u_shapeTex, 0);
  gl.drawArrays(gl.TRIANGLES,0,3);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

function savePNG(){ const url=canvas.toDataURL('image/png'); const a=document.createElement('a'); a.href=url; a.download='interim.png'; document.body.appendChild(a); a.click(); a.remove(); }

async function saveGrid(){
  const frames = 12; // 12 steps
  const cols = 6, rows = Math.ceil(frames/cols);
  const thumb = 256 * dpr; // size per cell in output
  const out = document.createElement('canvas');
  out.width = cols * thumb; out.height = rows * thumb;
  const ctx = out.getContext('2d');
  const original = { scale: settings.scale };
  for(let i=0;i<frames;i++){
    settings.scale = 1.0 - i/(frames-1);
    await new Promise(r=>requestAnimationFrame(r));
    ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, (i%cols)*thumb, Math.floor(i/cols)*thumb, thumb, thumb);
  }
  settings.scale = original.scale;
  const url = out.toDataURL('image/png');
  const a=document.createElement('a'); a.href=url; a.download='interim_grid.png'; document.body.appendChild(a); a.click(); a.remove();
}


