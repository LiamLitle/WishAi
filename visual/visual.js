// visual.js — WishAI Embedding Visualizer (Three.js WebGL)
// Zoom : molette | Pan/Orbit : cliquer-glisser | Hover : nom du token

'use strict';

const container = document.getElementById('canvas-container');
const tooltip   = document.getElementById('tooltip');
const nodata    = document.getElementById('nodata');

// ── État global ───────────────────────────────────────────────
let emb          = null;
let searchQuery  = '';
let activeFilter = 'all';   // 'all' | 'words' | 'numbers' | 'special' | 'endword'
let showLines    = false;
let isSeparated  = true;

// ── Three.js ──────────────────────────────────────────────────
let scene, camera, renderer, controls, raycaster, mouse;
let points, lines;
let hoveredIdx = null;

// Hover sprite (pour afficher l'anneau blanc autour du point survolé)
let hoverMesh;

initThree();
animate();

function initThree() {
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x101a24, 0.002);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.z = 200;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x101a24, 1);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableRotate = false; // Disable 3D rotation
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN
    };

    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 2.0; // Rayon de tolérance pour le survol
    mouse = new THREE.Vector2();

    // Mesh pour le hover
    const hoverGeo = new THREE.RingGeometry(1.5, 2.5, 16);
    const hoverMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.8 });
    hoverMesh = new THREE.Mesh(hoverGeo, hoverMat);
    hoverMesh.visible = false;
    scene.add(hoverMesh);

    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseleave', () => { hoveredIdx = null; tooltip.style.display = 'none'; hoverMesh.visible = false; });
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ── Chargement JSON ───────────────────────────────────────────
async function fetchJSON(path) {
    try {
        const r = await fetch(path, { cache: 'no-store' });
        if (!r.ok) return null;
        return await r.json();
    } catch { return null; }
}

async function autoLoad() {
    let data = null;
    const active = await fetchJSON('../model/active.json');
    if (active?.model)
        data = await fetchJSON(`../model/${active.model}/embeddings.json`);
    if (data) { setData(data); return; }
    nodata.classList.remove('hidden');
}

function setData(data) {
    emb = data;
    emb.baseX = new Float32Array(emb.x);
    emb.baseY = new Float32Array(emb.y);
    
    nodata.classList.add('hidden');
    buildThreeScene();
    applySeparation();
    fitToScreen();
    updateInfoBar();
}

function tokenCategory(tok) {
    if (!tok || tok.trim() === '' || tok === '\n' || tok === '\r\n') return 'empty';
    if (tok.endsWith('</w>'))                                         return 'endword';
    if (/^[0-9]/.test(tok))                                          return 'numbers';
    if (/^[a-zA-ZÀ-ÿÀ-ž]/.test(tok))                                 return 'words';
    return 'special';
}

const COLORS = {
    empty:   new THREE.Color('#1a2530'),
    words:   new THREE.Color('#5c7b99'),
    numbers: new THREE.Color('#cca040'),
    special: new THREE.Color('#b05060'),
    endword: new THREE.Color('#3a5068'),
    match:   new THREE.Color('#e07820'),
    fade:    new THREE.Color('#2a3a4a')
};

function tokenColor(cat) {
    return COLORS[cat] ?? COLORS.special;
}

function passesFilter(tok) {
    if (activeFilter === 'all') return true;
    return tokenCategory(tok) === activeFilter;
}

// Génère une texture de cercle avec glow pour les points
function createPointTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    
    // Glow
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.2, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.2)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

function buildThreeScene() {
    if (points) scene.remove(points);
    if (lines) scene.remove(lines);
    
    const n = emb.x.length;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    
    for (let i = 0; i < n; i++) {
        const tok = emb.tokens[i];
        const cat = tokenCategory(tok);
        const c = tokenColor(cat);
        colors[i * 3]     = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
        sizes[i] = cat === 'endword' ? 0.75 : 1.0;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // ShaderMaterial pour avoir une taille variable par point
    const material = new THREE.ShaderMaterial({
        uniforms: {
            pointTexture: { value: createPointTexture() },
            baseSize: { value: 60.0 }
        },
        vertexShader: `
            attribute float size;
            attribute vec3 color;
            varying vec3 vColor;
            uniform float baseSize;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * baseSize * (100.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            uniform sampler2D pointTexture;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0) * texture2D(pointTexture, gl_PointCoord);
            }
        `,
        blending: THREE.AdditiveBlending,
        depthTest: true,
        depthWrite: false,
        transparent: true,
    });

    points = new THREE.Points(geometry, material);
    scene.add(points);
    
    updateColors(); // Applique le filtrage
}

function applySeparation() {
    if (!emb || !points) return;
    const n = emb.baseX.length;
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    for (let i = 0; i < n; i++) {
        if (emb.baseX[i] < mnX) mnX = emb.baseX[i];
        if (emb.baseX[i] > mxX) mxX = emb.baseX[i];
        if (emb.baseY[i] < mnY) mnY = emb.baseY[i];
        if (emb.baseY[i] > mxY) mxY = emb.baseY[i];
    }
    
    // Échelle pour écarter les groupes si l'espace est très petit (-0.2 à 0.2)
    const spaceWidth = (mxX - mnX) || 1;
    let offsetX = spaceWidth * 1.5;
    let offsetY = (mxY - mnY) * 1.5 || 1;
    
    // Si l'espace est minuscule, on écarte plus
    if (offsetX < 10) { offsetX *= 10; offsetY *= 10; }

    const positions = points.geometry.attributes.position.array;

    for (let i = 0; i < n; i++) {
        let dx = 0, dy = 0;
        if (isSeparated) {
            const cat = tokenCategory(emb.tokens[i]);
            if (cat === 'numbers') { dx = -offsetX; dy = -offsetY; }
            else if (cat === 'special') { dx = offsetX; dy = -offsetY; }
            else if (cat === 'endword') { dy = offsetY; }
        }
        emb.x[i] = emb.baseX[i] + dx;
        emb.y[i] = emb.baseY[i] + dy;
        
        // Scale global pour étendre un peu si c'est de l'ancien modèle sans Z
        const mul = (mxX - mnX) < 1.0 ? 50 : 1; 

        positions[i * 3]     = emb.x[i] * mul;
        positions[i * 3 + 1] = emb.y[i] * mul;
        positions[i * 3 + 2] = 0; // Fixé à Z=0 pour un mode 2D strict
    }
    points.geometry.attributes.position.needsUpdate = true;
    
    buildLines(); // Met à jour les lignes avec les nouvelles coordonnées
}

function updateColors() {
    if (!emb || !points) return;
    const n = emb.x.length;
    const colors = points.geometry.attributes.color.array;
    const sq = searchQuery.toLowerCase();
    const hasSearch = sq.length > 0;
    const hasFilter = activeFilter !== 'all';
    
    for (let i = 0; i < n; i++) {
        const tok = emb.tokens[i];
        const isMatch = hasSearch && tok.toLowerCase().includes(sq);
        const passFilter = passesFilter(tok);
        
        let c;
        if ((hasSearch && !isMatch) || (hasFilter && !passFilter)) {
            c = COLORS.fade;
        } else {
            c = isMatch ? COLORS.match : tokenColor(tokenCategory(tok));
        }
        
        colors[i * 3]     = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
    }
    points.geometry.attributes.color.needsUpdate = true;
    buildLines();
}

function buildLines() {
    if (lines) {
        scene.remove(lines);
        lines.geometry.dispose();
        lines.material.dispose();
        lines = null;
    }
    
    if (!showLines || !emb) return;
    
    const n = emb.x.length;
    const lineVertices = [];
    const lineColors = [];
    
    const sq = searchQuery.toLowerCase();
    const hasSearch = sq.length > 0;
    const hasFilter = activeFilter !== 'all';
    const positions = points.geometry.attributes.position.array;
    
    // Pour ne pas faire exploser le CPU, on ne traite que les 2000 premiers
    // ou on pourrait utiliser un octree. Pour aller vite, on limite la boucle.
    const maxNodes = Math.min(n, 1500); 
    
    for (let i = 0; i < maxNodes; i++) {
        const tok1 = emb.tokens[i];
        const cat = tokenCategory(tok1);
        if ((hasFilter && !passesFilter(tok1)) || (hasSearch && !tok1.toLowerCase().includes(sq))) continue;
        
        const x1 = positions[i*3], y1 = positions[i*3+1], z1 = 0;
        const c = tokenColor(cat);
        
        let nearest = -1, bestD2 = Infinity;
        for (let j = 0; j < maxNodes; j++) {
            if (i === j) continue;
            const tok2 = emb.tokens[j];
            if (tokenCategory(tok2) !== cat) continue;
            
            const dx = x1 - positions[j*3];
            const dy = y1 - positions[j*3+1];
            const d2 = dx*dx + dy*dy;
            if (d2 < bestD2) {
                bestD2 = d2;
                nearest = j;
            }
        }
        
        if (nearest !== -1) {
            lineVertices.push(x1, y1, z1);
            lineVertices.push(positions[nearest*3], positions[nearest*3+1], positions[nearest*3+2]);
            lineColors.push(c.r, c.g, c.b, c.r, c.g, c.b);
        }
    }
    
    if (lineVertices.length > 0) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
        const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.2 });
        lines = new THREE.LineSegments(geo, mat);
        scene.add(lines);
    }
}

function fitToScreen() {
    if (!points) return;
    points.geometry.computeBoundingSphere();
    const radius = points.geometry.boundingSphere.radius;
    const center = points.geometry.boundingSphere.center;
    
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + radius * 1.5 + 50);
    
    // Ajuste le fog
    scene.fog.density = 1.2 / (radius || 100);
}

// ── Interaction ───────────────────────────────────────────────
function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    if (!points) return;
    
    // Raycaster a besoin d'être mis à l'échelle si la caméra bouge
    // Threshold dynamique basé sur la distance
    raycaster.params.Points.threshold = camera.position.distanceTo(controls.target) * 0.02;
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);
    
    if (intersects.length > 0) {
        const intersect = intersects[0];
        hoveredIdx = intersect.index;
        
        const pos = points.geometry.attributes.position;
        hoverMesh.position.set(pos.getX(hoveredIdx), pos.getY(hoveredIdx), pos.getZ(hoveredIdx));
        hoverMesh.quaternion.copy(camera.quaternion); // Face caméra
        const s = camera.position.distanceTo(hoverMesh.position) * 0.01;
        hoverMesh.scale.set(s, s, s);
        hoverMesh.visible = true;
        
        showTooltip(event.clientX, event.clientY, hoveredIdx);
    } else {
        hoveredIdx = null;
        hoverMesh.visible = false;
        tooltip.style.display = 'none';
    }
}

function showTooltip(sx, sy, idx) {
    if (idx === null) { tooltip.style.display = 'none'; return; }
    const tok = emb.tokens[idx];
    const cat = tokenCategory(tok);
    const catLabel = { words: 'mot', numbers: 'chiffre', special: 'spécial',
                       endword: 'fin de mot', empty: 'vide' }[cat] ?? cat;
    tooltip.innerHTML =
        `<b>"${tok.replace(/\n/g, '↵').replace(/</g, '&lt;')}"</b>` +
        `<span class="tip-id"> id ${idx}</span>` +
        `<span class="tip-cat">${catLabel}</span>`;
    tooltip.style.display = 'flex';
    tooltip.style.left = Math.min(sx + 16, window.innerWidth  - 220) + 'px';
    tooltip.style.top  = Math.max(sy - 12, 10) + 'px';
}

// ── Boucle de rendu ───────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// ── Filtres & Boutons ─────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.dataset.action === 'separate') {
            isSeparated = !isSeparated;
            btn.classList.toggle('active', isSeparated);
            applySeparation();
            fitToScreen();
            return;
        }
        if (btn.dataset.lines) {
            showLines = !showLines;
            btn.classList.toggle('active', showLines);
            buildLines();
            return;
        }
        
        document.querySelectorAll('.filter-btn:not([data-lines]):not([data-action])').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        updateColors();
    });
});

document.getElementById('btn-reset-drag').addEventListener('click', () => {
    controls.reset();
    fitToScreen();
});

document.getElementById('search').addEventListener('input', e => {
    searchQuery = e.target.value;
    updateColors();
});

function updateInfoBar() {
    if (!emb) return;
    document.getElementById('info-step').textContent  = `Étape : ${emb.step ?? '—'}`;
    document.getElementById('info-loss').textContent  = `val_loss : ${emb.val_loss ?? '—'}`;
    document.getElementById('info-count').textContent = `${emb.vocab_size ?? '—'} tokens`;
}

document.getElementById('btn-reload').addEventListener('click', autoLoad);

document.getElementById('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
        try { setData(JSON.parse(ev.target.result)); }
        catch { alert('Fichier JSON invalide.'); }
    };
    reader.readAsText(file);
    e.target.value = '';
});

setInterval(autoLoad, 30_000);
autoLoad(); // Charge au démarrage
