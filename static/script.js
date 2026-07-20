/* ═══════════════════════════════════════════════════════════════════════════
   DHUN AI — Frontend Application
   Handles: Face auth, music library, player, song generation, admin panel,
            PWA install, service worker, notifications.
═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const API = '/api';
const FACE_MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights';
const BLINK_EAR_THRESH  = 0.24;   // Eye Aspect Ratio threshold for blink
const FACE_DETECT_OPTS  = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });

// ── App State ────────────────────────────────────────────────────────────────
const S = {
  user:       null,    // { face_id, token }
  songs:      [],
  curSong:    null,
  audio:      null,
  isPlaying:  false,
  modelsOk:   false,
  adminToken: null,
  camStream:  null,
  detectLoop: null,
  songToDelete: null,
  searchQ:    '',
  filterGenre:'',
  filterMood: '',
  page:       1,
  totalPages: 1,
  installEvt: null,    // PWA BeforeInstallPromptEvent
  genCharts:  {},      // Chart.js instances
  pendingRegistrationEmbedding: null,
};

// ══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
  registerServiceWorker();

  // Auto-load model progress bar animation
  animateLoaderBar();

  try {
    // Preload face-api.js models in background while showing loader
    await loadFaceModels();

    // Try restoring session from localStorage
    const { token, faceId } = loadToken();
    if (token && faceId) {
      try {
        const profile = await callApi('GET', '/auth/verify', null, token);
        S.user = { token, face_id: faceId, display_name: profile.display_name };
        enterApp();
        return;
      } catch {
        clearToken();
      }
    }

    // Show auth screen
    showScreen('auth-screen');
    hideScreen('loading-screen');
  } catch (err) {
    console.error('Init error:', err);
    showScreen('auth-screen');
    hideScreen('loading-screen');
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// PWA — Service Worker + Install
// ══════════════════════════════════════════════════════════════════════════════
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
      .then(reg => console.log('SW registered:', reg.scope))
      .catch(err => console.warn('SW registration failed:', err));
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// LOADER
// ══════════════════════════════════════════════════════════════════════════════
function animateLoaderBar() {
  const bar = $('model-load-bar');
  if (!bar) return;
  let w = 0;
  const iv = setInterval(() => {
    w = Math.min(w + Math.random() * 12, 85);
    bar.style.width = w + '%';
    if (w >= 85) clearInterval(iv);
  }, 200);
}

function finishLoader() {
  const bar = $('model-load-bar');
  if (bar) bar.style.width = '100%';
  setTimeout(() => {
    const ls = $('loading-screen');
    if (ls) { ls.style.opacity = '0'; ls.style.transition = 'opacity 0.4s'; setTimeout(() => ls.classList.add('hidden'), 400); }
  }, 300);
}

// ══════════════════════════════════════════════════════════════════════════════
// FACE API — Model Loading
// ══════════════════════════════════════════════════════════════════════════════
async function loadFaceModels() {
  if (S.modelsOk) return;
  await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_MODEL_URL);
  S.modelsOk = true;
  console.log('face-api.js models loaded ✓');
}

// ══════════════════════════════════════════════════════════════════════════════
// CAMERA
// ══════════════════════════════════════════════════════════════════════════════
async function openCamera() {
  const video = $('camera-video');
  const overlay = $('camera-status-overlay');
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not support camera access.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'user' }, width: { ideal: 480 }, height: { ideal: 480 } },
    });
    S.camStream = stream;
    video.srcObject = stream;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Camera preview did not start.')), 8000);
      video.onloadedmetadata = () => { clearTimeout(timer); resolve(); };
    });
    await video.play();
    if (!video.videoWidth || !video.videoHeight) {
      throw new Error('Camera preview is unavailable.');
    }
    if (overlay) overlay.style.opacity = '0';
    $('camera-wrap').classList.add('detecting');
  } catch (err) {
    closeCamera();
    const detail = err?.name === 'NotAllowedError'
      ? 'Camera permission was blocked. Allow camera access in your browser settings and retry.'
      : (err?.message || 'Camera preview could not start.');
    throw new Error(detail);
  }
}

function closeCamera() {
  if (S.camStream) { S.camStream.getTracks().forEach(t => t.stop()); S.camStream = null; }
  if (S.detectLoop) { clearInterval(S.detectLoop); S.detectLoop = null; }
  $('camera-wrap').classList.remove('detecting');
  const overlay = $('camera-status-overlay');
  if (overlay) overlay.style.opacity = '1';
}

// ── Eye Aspect Ratio for blink detection ──────────────────────────────────────
function ptDist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}
function eyeAR(pts) {
  const A = ptDist(pts[1], pts[5]);
  const B = ptDist(pts[2], pts[4]);
  const C = ptDist(pts[0], pts[3]);
  return (A + B) / (2.0 * C);
}

/**
 * Detect a face on the video feed.
 * @param {boolean} requireBlink - If true, waits for a blink (anti-spoofing).
 * @returns {Promise<number[]>} 128-dim face descriptor array.
 */
function detectFace(requireBlink = false) {
  return new Promise((resolve, reject) => {
    const video = $('camera-video');
    let blinkDone = !requireBlink;
    let eyeOpen   = true;
    let frames    = 0;
    const MAX_FRAMES = 200; // ~40s at 5fps

    S.detectLoop = setInterval(async () => {
      frames++;
      if (frames > MAX_FRAMES) {
        clearInterval(S.detectLoop);
        return reject(new Error('Detection timed out. Please try again.'));
      }

      let det;
      try {
        det = await faceapi
          .detectSingleFace(video, FACE_DETECT_OPTS)
          .withFaceLandmarks()
          .withFaceDescriptor();
      } catch (err) {
        clearInterval(S.detectLoop);
        return reject(new Error('Face detection could not start. Refresh the page and try again.'));
      }

      if (!det) {
        setAuthStatus('No face detected — look at the camera', 'warning');
        return;
      }

      // Draw face box
      const faceBox = $('face-box');
      if (faceBox) {
        const box = det.detection.box;
        const wrap = $('camera-wrap');
        const scaleX = wrap.offsetWidth / video.videoWidth;
        const scaleY = wrap.offsetHeight / video.videoHeight;
        faceBox.style.display = 'block';
        faceBox.style.left   = (box.x * scaleX) + 'px';
        faceBox.style.top    = (box.y * scaleY) + 'px';
        faceBox.style.width  = (box.width * scaleX) + 'px';
        faceBox.style.height = (box.height * scaleY) + 'px';
      }

      setAuthStatus('Face detected! ✓', 'success');

      // Anti-spoofing: blink detection
      if (requireBlink && !blinkDone) {
        const lm = det.landmarks;
        const ear = (eyeAR(lm.getLeftEye()) + eyeAR(lm.getRightEye())) / 2;

        if (ear < BLINK_EAR_THRESH) {
          eyeOpen = false;
          setAuthStatus('👁 Blink detected — verifying…', 'info');
        } else if (!eyeOpen) {
          // Eye was closed, now open again → blink completed
          blinkDone = true;
        }

        if (!blinkDone) {
          setAuthStatus('Please blink once to prove you are real…', 'info');
          return;
        }
      }

      clearInterval(S.detectLoop);
      $('face-box').style.display = 'none';
      resolve(Array.from(det.descriptor));
    }, 200); // 5 fps
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// AUTH
// ══════════════════════════════════════════════════════════════════════════════
window.startRegister = async () => {
  disableAuthButtons(true);
  setAuthStatus('Loading…', 'info');
  try {
    await openCamera();
    setAuthStatus('Look at the camera and stay still…', 'info');
    const embedding = await detectFace(false); // no blink for registration
    closeCamera();
    S.pendingRegistrationEmbedding = embedding;
    setAuthStatus('Face scan complete. Add your details to continue.', 'success');
    openModal('registration-details-modal');
    $('registration-name').focus();
  } catch (err) {
    closeCamera();
    setAuthStatus(err.message || 'Registration failed. Try again.', 'error');
    disableAuthButtons(false);
  }
};

window.cancelRegistrationDetails = () => {
  S.pendingRegistrationEmbedding = null;
  closeModal('registration-details-modal');
  setAuthStatus('Registration cancelled. You can scan again when ready.', 'info');
  disableAuthButtons(false);
};

window.completeRegistration = async () => {
  const name = $('registration-name').value.trim();
  const age = Number($('registration-age').value);
  const termsAccepted = $('registration-terms').checked;

  if (name.length < 2) return notify('Please enter your full name.', 'error');
  if (!Number.isInteger(age) || age < 18 || age > 120) return notify('You must be 18 or older to register.', 'error');
  if (!termsAccepted) return notify('Please accept the Terms & Conditions to continue.', 'error');
  if (!S.pendingRegistrationEmbedding) return notify('Your face scan expired. Please scan again.', 'error');

  try {
    const res = await callApi('POST', '/auth/register', {
      embedding: S.pendingRegistrationEmbedding,
      name,
      age,
      terms_accepted: termsAccepted,
    });
    closeModal('registration-details-modal');
    S.pendingRegistrationEmbedding = null;
    saveToken(res.token, res.face_id);
    S.user = { token: res.token, face_id: res.face_id, display_name: res.display_name };
    notify(res.already_registered ? `Hello, ${res.display_name}!` : `Welcome, ${res.display_name}!`, 'success');
    enterApp();
  } catch (err) {
    notify(err.message || 'Registration failed. Please try again.', 'error');
  }
};

window.startLogin = async () => {
  disableAuthButtons(true);
  setAuthStatus('Loading…', 'info');
  try {
    await openCamera();
    setAuthStatus('Look at the camera and blink once…', 'info');
    const embedding = await detectFace(true); // require blink
    closeCamera();
    setAuthStatus('Verifying identity…', 'info');

    const res = await callApi('POST', '/auth/login', { embedding });
    saveToken(res.token, res.face_id);
    S.user = { token: res.token, face_id: res.face_id, display_name: res.display_name };
    notify(`Hello, ${res.display_name || 'there'}!`, 'success');
    enterApp();
  } catch (err) {
    closeCamera();
    setAuthStatus(err.message || 'Login failed. Face not recognised.', 'error');
    disableAuthButtons(false);
  }
};

window.logout = () => {
  clearToken();
  S.user = null;
  stopAudio();
  showScreen('auth-screen');
  hideScreen('app-screen');
  setAuthStatus('You have been logged out.', 'info');
  disableAuthButtons(false);
};

function disableAuthButtons(v) {
  const r = $('btn-register'), l = $('btn-login');
  if (r) r.disabled = v;
  if (l) l.disabled = v;
}

function setAuthStatus(msg, type = 'info') {
  const el = $('auth-status');
  if (!el) return;
  el.className = `auth-status ${type}`;
  el.querySelector('#auth-status-text').textContent = msg;
}

// ══════════════════════════════════════════════════════════════════════════════
// APP ENTRY
// ══════════════════════════════════════════════════════════════════════════════
function enterApp() {
  finishLoader();
  hideScreen('auth-screen');
  hideScreen('loading-screen');
  showScreen('app-screen');
  $('sidebar-face-id').textContent = S.user.face_id.slice(0, 12) + '…';
  loadSongs();
}

// ══════════════════════════════════════════════════════════════════════════════
// SIDEBAR & NAVIGATION
// ══════════════════════════════════════════════════════════════════════════════
window.toggleSidebar = () => {
  const sidebar  = $('sidebar');
  const backdrop = $('sidebar-backdrop');
  const open = sidebar.classList.toggle('open');
  backdrop.classList.toggle('visible', open);
};

window.showView = (name) => {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const el = $(`view-${name}`);
  if (el) el.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
};

// ══════════════════════════════════════════════════════════════════════════════
// SONG LIBRARY
// ══════════════════════════════════════════════════════════════════════════════
async function loadSongs(append = false) {
  if (!append) S.page = 1;
  try {
    const params = new URLSearchParams({
      search: S.searchQ,
      genre:  S.filterGenre,
      mood:   S.filterMood,
      page:   S.page,
      limit:  20,
    });
    const data = await callApi('GET', `/songs?${params}`, null, S.user.token);

    if (append) {
      S.songs = [...S.songs, ...data.songs];
    } else {
      S.songs = data.songs;
    }
    S.totalPages = data.pages;
    renderLibrary();
    $('load-more-wrap').classList.toggle('hidden', S.page >= S.totalPages);
  } catch (err) {
    console.error('Load songs error:', err);
    notify('Could not load songs: ' + err.message, 'error');
  }
}

window.loadMoreSongs = () => {
  S.page++;
  loadSongs(true);
};

function renderLibrary() {
  const grid = $('songs-grid');
  if (!S.songs.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🎵</div>
        <h3>No songs yet</h3>
        <p>Generate your first AI masterpiece!</p>
        <button class="btn-primary" onclick="showGenerator()">✨ Generate Song</button>
      </div>`;
    return;
  }

  grid.innerHTML = S.songs.map(s => songCardHTML(s)).join('');

  // Lazy-load images via IntersectionObserver
  const obs = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        img.onload = () => img.removeAttribute('data-src');
        observer.unobserve(img);
      }
    });
  }, { rootMargin: '100px' });

  grid.querySelectorAll('img[data-src]').forEach(img => obs.observe(img));
}

function songCardHTML(s) {
  const coverURL = s.cover_url || `https://picsum.photos/seed/${s.song_id}/400/400`;
  const date = new Date(s.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  const playing = S.curSong?.song_id === s.song_id ? 'playing' : '';

  return `
  <div class="song-card ${playing}" id="card-${s.song_id}">
    <div class="song-cover-wrap">
      <img class="song-cover" data-src="${escHtml(coverURL)}"
           src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%230d0d18' width='200' height='200'/%3E%3C/svg%3E"
           alt="${escHtml(s.title)}" />
      <div class="song-overlay">
        <button class="play-btn" title="Play"
          onclick="event.stopPropagation(); togglePlay('${s.song_id}')">
          ${S.curSong?.song_id === s.song_id && S.isPlaying ? '⏸' : '▶'}
        </button>
      </div>
    </div>
    <div class="song-info">
      <h3 class="song-title">${escHtml(s.title)}</h3>
      <div class="song-meta">
        <span class="genre-badge">${escHtml(s.genre)}</span>
        <span class="mood-badge">${escHtml(s.mood)}</span>
      </div>
      <div class="song-footer">
        <span class="song-date">${date}</span>
        <div class="song-actions">
          <button class="icon-btn" title="Lyrics"
            onclick="event.stopPropagation(); openLyricsModal('${s.song_id}')">📝</button>
          <button class="icon-btn" title="Download"
            onclick="event.stopPropagation(); downloadSong('${s.song_id}')">⬇</button>
          <button class="icon-btn delete-btn" title="Delete"
            onclick="event.stopPropagation(); openDeleteModal('${s.song_id}', '${escHtml(s.title)}')">🗑</button>
        </div>
      </div>
    </div>
  </div>`;
}

// ── Search & Filter ───────────────────────────────────────────────────────────
let searchDebounce;
window.onSearchInput = (val) => {
  S.searchQ = val;
  $('search-clear').classList.toggle('hidden', !val);
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => loadSongs(), 350);
};

window.clearSearch = () => {
  $('search-input').value = '';
  S.searchQ = '';
  $('search-clear').classList.add('hidden');
  loadSongs();
};

window.setGenreFilter = (genre) => {
  S.filterGenre = genre;
  document.querySelectorAll('.pill').forEach(p =>
    p.classList.toggle('active', p.dataset.genre === genre));
  loadSongs();
};

// ══════════════════════════════════════════════════════════════════════════════
// PLAYER
// ══════════════════════════════════════════════════════════════════════════════
function getAudio() {
  if (!S.audio) {
    S.audio = $('audio-player');
    S.audio.addEventListener('timeupdate',    onTimeUpdate);
    S.audio.addEventListener('ended',         onAudioEnd);
    S.audio.addEventListener('loadedmetadata',onMetaLoaded);
    S.audio.addEventListener('error',         () => notify('Audio playback error', 'error'));
  }
  return S.audio;
}

window.togglePlay = (songId) => {
  // If called from a card with explicit songId
  if (songId && S.curSong?.song_id !== songId) {
    playSong(songId);
    return;
  }
  // Toggle current song
  const audio = getAudio();
  if (S.isPlaying) {
    audio.pause();
    S.isPlaying = false;
  } else {
    if (!S.curSong) return;
    audio.play().catch(() => {});
    S.isPlaying = true;
  }
  syncPlayButtons();
};

function playSong(songId) {
  const song = S.songs.find(s => s.song_id === songId);
  if (!song) return;

  S.curSong = song;
  const audio = getAudio();

  if (song.audio_url) {
    audio.src = song.audio_url;
    audio.load();
    audio.play().catch(() => {
      notify('Audio not available — displaying song info only', 'warning');
    });
    S.isPlaying = true;
    callApi('POST', `/songs/${songId}/play`, null, S.user.token).catch(() => {});
  } else {
    S.isPlaying = false;
    notify('Audio not yet available for this song', 'info');
  }

  updatePlayerBar(song);
  showPlayerBar();
  refreshCardsPlayState();
}

function updatePlayerBar(song) {
  const coverURL = song.cover_url || `https://picsum.photos/seed/${song.song_id}/80/80`;
  $('player-cover').src = coverURL;
  $('player-cover').alt = song.title;
  $('player-title').textContent = song.title;
  $('player-genre').textContent = `${song.genre} • ${song.mood}`;

  // Expanded player
  $('exp-cover').src = coverURL;
  $('exp-title').textContent = song.title;
  $('exp-genre-mood').textContent = `${song.genre} · ${song.mood} · ${song.language}`;
  $('exp-download-btn').onclick = () => downloadSong(song.song_id);
  $('exp-share-btn').onclick    = () => shareSong(song.song_id);
}

function showPlayerBar() {
  $('player-bar').classList.remove('hidden');
  syncPlayButtons();
}

function stopAudio() {
  if (S.audio) { S.audio.pause(); S.audio.src = ''; }
  S.isPlaying = false;
  S.curSong = null;
  $('player-bar').classList.add('hidden');
}

function syncPlayButtons() {
  const icon = S.isPlaying ? '⏸' : '▶';
  const ppBtn = $('play-pause-btn');
  const expBtn = $('exp-play-btn');
  if (ppBtn) ppBtn.textContent = icon;
  if (expBtn) expBtn.textContent = icon;
  refreshCardsPlayState();
}

function refreshCardsPlayState() {
  document.querySelectorAll('.song-card').forEach(el => {
    const id = el.id.replace('card-', '');
    const active = S.curSong?.song_id === id;
    el.classList.toggle('playing', active);
    const btn = el.querySelector('.play-btn');
    if (btn && active) btn.textContent = S.isPlaying ? '⏸' : '▶';
    else if (btn) btn.textContent = '▶';
  });
}

function onTimeUpdate() {
  if (!S.audio || isNaN(S.audio.duration)) return;
  const pct = (S.audio.currentTime / S.audio.duration) * 100;
  $('progress-fill').style.width = pct + '%';
  $('exp-progress-fill').style.width = pct + '%';
  $('current-time').textContent = fmtTime(S.audio.currentTime);
  $('exp-current').textContent  = fmtTime(S.audio.currentTime);
}

function onMetaLoaded() {
  const dur = fmtTime(S.audio.duration);
  $('total-time').textContent = dur;
  $('exp-total').textContent  = dur;
}

function onAudioEnd() {
  S.isPlaying = false;
  syncPlayButtons();
  nextSong();
}

window.seekTo = (e) => {
  if (!S.audio || isNaN(S.audio.duration)) return;
  const bar = e.currentTarget;
  const rect = bar.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  S.audio.currentTime = ratio * S.audio.duration;
};

window.setVolume = (val) => {
  if (S.audio) S.audio.volume = parseFloat(val);
};

window.prevSong = () => {
  const idx = S.songs.findIndex(s => s.song_id === S.curSong?.song_id);
  if (idx > 0) playSong(S.songs[idx - 1].song_id);
};

window.nextSong = () => {
  const idx = S.songs.findIndex(s => s.song_id === S.curSong?.song_id);
  if (idx < S.songs.length - 1) playSong(S.songs[idx + 1].song_id);
};

// ── Expanded player ────────────────────────────────────────────────────────────
window.showExpandedPlayer = () => {
  if (!S.curSong) return;
  openModal('expanded-player');
};
window.hideExpandedPlayer = () => closeModal('expanded-player');

// ══════════════════════════════════════════════════════════════════════════════
// SONG GENERATOR
// ══════════════════════════════════════════════════════════════════════════════
window.showGenerator = () => openModal('generator-modal');
window.hideGenerator = () => closeModal('generator-modal');

window.generateSong = async () => {
  const prompt   = $('gen-prompt').value.trim();
  const genre    = $('gen-genre').value;
  const mood     = $('gen-mood').value;
  const language = $('gen-language').value;
  const duration = $('gen-duration').value;

  if (!prompt) { notify('Please describe your song!', 'warning'); return; }

  closeModal('generator-modal');
  showProgressOverlay();

  // Animate steps in sequence (simulated since it's one API call)
  setStep(1, 'active', 'Writing your lyrics…');
  animateProgressBar(0, 25, 3000);

  const step2Timer = setTimeout(() => {
    setStep(1, 'done', '✅ Lyrics written!');
    setStep(2, 'active', 'Generating album artwork…');
    animateProgressBar(25, 60, 6000);
  }, 3000);

  const step3Timer = setTimeout(() => {
    setStep(2, 'done', '✅ Cover art ready!');
    setStep(3, 'active', 'Composing music…');
    animateProgressBar(60, 90, 10000);
  }, 9000);

  try {
    const data = await callApi('POST', '/songs/generate',
      { prompt, genre, mood, language, duration },
      S.user.token
    );

    clearTimeout(step2Timer);
    clearTimeout(step3Timer);

    setStep(1, 'done', '✅ Lyrics written!');
    setStep(2, 'done', '✅ Cover art ready!');
    setStep(3, 'done', '✅ Music composed!');
    setStep(4, 'active', 'Saving to your library…');
    animateProgressBar(90, 100, 800);

    setTimeout(() => {
      setStep(4, 'done', '✅ Saved!');
      setTimeout(() => {
        hideProgressOverlay();
        S.songs.unshift(data.song);
        renderLibrary();
        playSong(data.song.song_id);
        notify(`"${data.song.title}" is ready! 🎵`, 'success');
      }, 700);
    }, 800);

  } catch (err) {
    clearTimeout(step2Timer);
    clearTimeout(step3Timer);
    hideProgressOverlay();
    notify('Generation failed: ' + err.message, 'error');
    openModal('generator-modal');
  }
};

function showProgressOverlay() {
  // Reset all steps
  [1,2,3,4].forEach(i => setStep(i, 'waiting', 'Waiting…'));
  $('gen-progress-fill').style.width = '0%';
  $('progress-overlay').classList.remove('hidden');
}

function hideProgressOverlay() {
  $('progress-overlay').classList.add('hidden');
}

function setStep(num, state, statusText) {
  const step = $(`step-${num}`);
  if (!step) return;
  step.className = `progress-step ${state}`;
  $(`step-${num}-status`).textContent = statusText;
}

function animateProgressBar(from, to, durationMs) {
  const bar = $('gen-progress-fill');
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / durationMs);
    bar.style.width = (from + (to - from) * t) + '%';
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ══════════════════════════════════════════════════════════════════════════════
// LYRICS MODAL
// ══════════════════════════════════════════════════════════════════════════════
window.openLyricsModal = (songId) => {
  const song = S.songs.find(s => s.song_id === songId);
  if (!song) return;

  $('lyrics-modal-title').textContent = song.title;
  $('lyrics-cover').src    = song.cover_url || `https://picsum.photos/seed/${songId}/80/80`;
  $('lyrics-genre-badge').textContent = song.genre;
  $('lyrics-mood-badge').textContent  = song.mood;
  $('lyrics-prompt').textContent = song.prompt;
  $('lyrics-text').textContent   = song.lyrics || '(No lyrics available)';

  $('lyrics-share-btn').onclick = () => shareSong(songId);

  openModal('lyrics-modal');
};

window.showLyricsPanel = () => {
  if (S.curSong) openLyricsModal(S.curSong.song_id);
};

window.hideLyrics = () => closeModal('lyrics-modal');

// ══════════════════════════════════════════════════════════════════════════════
// DELETE
// ══════════════════════════════════════════════════════════════════════════════
window.openDeleteModal = (songId, title) => {
  S.songToDelete = songId;
  $('delete-song-name').textContent = title;
  openModal('delete-modal');
};

window.hideDeleteModal = () => { S.songToDelete = null; closeModal('delete-modal'); };

window.confirmDelete = async () => {
  if (!S.songToDelete) return;
  try {
    await callApi('DELETE', `/songs/${S.songToDelete}`, null, S.user.token);
    S.songs = S.songs.filter(s => s.song_id !== S.songToDelete);
    if (S.curSong?.song_id === S.songToDelete) stopAudio();
    S.songToDelete = null;
    closeModal('delete-modal');
    renderLibrary();
    notify('Song deleted.', 'success');
  } catch (err) {
    notify('Delete failed: ' + err.message, 'error');
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// DOWNLOAD & SHARE
// ══════════════════════════════════════════════════════════════════════════════
window.downloadSong = (songId) => {
  const song = S.songs.find(s => s.song_id === songId);
  if (!song) return;

  if (song.audio_url) {
    const a = document.createElement('a');
    a.href = song.audio_url;
    a.download = `${song.title}.mp3`;
    a.target = '_blank';
    a.click();
  } else {
    notify('Audio download not available for this song yet.', 'info');
  }
};

window.shareSong = async (songId) => {
  const song = S.songs.find(s => s.song_id === songId);
  if (!song) return;
  const shareData = {
    title: `🎵 ${song.title} — DHUN AI`,
    text: `Listen to my AI-generated ${song.genre} song: "${song.title}"`,
    url: window.location.href,
  };
  if (navigator.share) {
    try { await navigator.share(shareData); }
    catch { /* user cancelled */ }
  } else {
    navigator.clipboard.writeText(`${shareData.title}\n${shareData.url}`)
      .then(() => notify('Link copied to clipboard!', 'success'));
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// ADMIN
// ══════════════════════════════════════════════════════════════════════════════
window.showAdminPin = () => {
  $('admin-pin-input').value = '';
  $('pin-error').classList.add('hidden');
  openModal('admin-pin-modal');
  setTimeout(() => $('admin-pin-input').focus(), 100);
};

window.hideAdminPin = () => closeModal('admin-pin-modal');

window.submitAdminPin = async () => {
  const pin = $('admin-pin-input').value.trim();
  if (!pin) return;
  try {
    const res = await callApi('POST', '/admin/login', { pin });
    S.adminToken = res.token;
    closeModal('admin-pin-modal');
    openAdminPanel();
  } catch {
    $('pin-error').classList.remove('hidden');
  }
};

function openAdminPanel() {
  hideScreen('app-screen');
  hideScreen('auth-screen');
  showScreen('admin-screen');
  switchAdminTab('overview');
}

window.exitAdmin = () => {
  hideScreen('admin-screen');
  if (S.user) showScreen('app-screen');
  else showScreen('auth-screen');
};

window.switchAdminTab = (tab) => {
  document.querySelectorAll('.admin-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.admin-nav-item').forEach(n => n.classList.toggle('active', n.dataset.tab === tab));
  $(`admin-tab-${tab}`).classList.add('active');

  switch (tab) {
    case 'overview':  loadAdminOverview();  break;
    case 'users':     loadAdminUsers();     break;
    case 'songs':     loadAdminSongs(1);    break;
    case 'settings':  loadAdminSettings();  break;
    case 'logs':      loadAdminLogs();      break;
  }
};

async function loadAdminOverview() {
  try {
    const data = await callApi('GET', '/admin/analytics', null, S.adminToken);
    $('stat-users').textContent = data.total_users;
    $('stat-songs').textContent = data.total_songs;
    $('stat-plays').textContent = data.total_plays;

    // Destroy old charts
    Object.values(S.genCharts).forEach(c => c.destroy());
    S.genCharts = {};

    // Genre bar chart
    if (data.genre_stats?.length) {
      S.genCharts.genre = new Chart($('chart-genre'), {
        type: 'bar',
        data: {
          labels: data.genre_stats.map(g => g._id || 'Unknown'),
          datasets: [{ label: 'Songs', data: data.genre_stats.map(g => g.count), backgroundColor: '#167a45' }],
        },
        options: chartOpts('Songs by Genre'),
      });
    }

    // Daily line chart
    if (data.daily_songs?.length) {
      S.genCharts.daily = new Chart($('chart-daily'), {
        type: 'line',
        data: {
          labels: data.daily_songs.map(d => d._id),
          datasets: [{ label: 'Songs', data: data.daily_songs.map(d => d.count), borderColor: '#167a45', backgroundColor: 'rgba(22,122,69,.12)', fill: true, tension: 0.4 }],
        },
        options: chartOpts('Daily Songs'),
      });
    }

    // Mood doughnut chart
    if (data.mood_stats?.length) {
      S.genCharts.mood = new Chart($('chart-mood'), {
        type: 'doughnut',
        data: {
          labels: data.mood_stats.map(m => m._id || 'Unknown'),
          datasets: [{ data: data.mood_stats.map(m => m.count), backgroundColor: ['#167a45','#39a96b','#92c95a','#f0b429','#d95d39','#be6a9b','#5171a5','#73a6ad','#9aa56a'], borderWidth: 0 }],
        },
        options: { ...chartOpts('Moods'), plugins: { legend: { position: 'bottom', labels: { color: '#66756b', font: { size: 11 } } } } },
      });
    }

    // Logs
    const logsList = $('logs-list');
    if (logsList && data.recent_events) {
      logsList.innerHTML = data.recent_events.map(e => `
        <div class="log-item">
          <span class="log-event">${escHtml(e.event || 'event')}</span>
          <span class="log-meta">${escHtml(e.face_id || e.song_id || '')}</span>
          <span class="log-time">${escHtml(e.timestamp || '')}</span>
        </div>`).join('');
    }
  } catch (err) {
    notify('Analytics load failed: ' + err.message, 'error');
  }
}

function chartOpts(label) {
  return {
    responsive: true,
    plugins: { legend: { display: false }, title: { display: false } },
    scales: {
      x: { ticks: { color: '#66756b', font: { size: 10 } }, grid: { color: 'rgba(22,122,69,.08)' } },
      y: { ticks: { color: '#66756b', font: { size: 10 } }, grid: { color: 'rgba(22,122,69,.08)' } },
    },
  };
}

async function loadAdminUsers() {
  const tbody = $('users-tbody');
  try {
    const data = await callApi('GET', '/admin/users', null, S.adminToken);
    tbody.innerHTML = data.users.map(u => `
      <tr>
        <td style="font-family:monospace;font-size:.75rem">${escHtml(u.face_id || '—')}</td>
        <td>${fmtDate(u.created_at)}</td>
        <td>${fmtDate(u.last_login)}</td>
        <td>${u.song_count ?? 0}</td>
        <td>
          <button class="btn-table-del"
            onclick="adminDeleteUser('${u.face_id}')">Delete</button>
        </td>
      </tr>`).join('') || '<tr><td colspan="5" class="loading-row">No users</td></tr>';
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" class="loading-row">Error: ${err.message}</td></tr>`;
  }
}

window.adminDeleteUser = async (faceId) => {
  if (!confirm(`Delete user ${faceId.slice(0,8)}… and all their songs?`)) return;
  try {
    await callApi('DELETE', `/admin/users/${faceId}`, null, S.adminToken);
    notify('User deleted.', 'success');
    loadAdminUsers();
  } catch (err) {
    notify('Delete failed: ' + err.message, 'error');
  }
};

async function loadAdminSongs(page = 1) {
  const tbody = $('admin-songs-tbody');
  try {
    const data = await callApi('GET', `/admin/songs?page=${page}&limit=15`, null, S.adminToken);
    tbody.innerHTML = data.songs.map(s => `
      <tr>
        <td><img class="thumb" src="${escHtml(s.cover_url || '')}" alt="" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\'/%3E'" /></td>
        <td>${escHtml(s.title)}</td>
        <td>${escHtml(s.genre)}</td>
        <td>${escHtml(s.mood)}</td>
        <td style="font-family:monospace;font-size:.72rem">${escHtml((s.user_face_id||'').slice(0,10))}…</td>
        <td>${s.play_count ?? 0}</td>
        <td>${fmtDate(s.created_at)}</td>
        <td>
          <button class="btn-table-del"
            onclick="adminDeleteSong('${s.song_id}')">Delete</button>
        </td>
      </tr>`).join('') || '<tr><td colspan="8" class="loading-row">No songs</td></tr>';

    // Pagination
    const pages = Math.ceil((data.total || 0) / 15);
    $('admin-songs-pagination').innerHTML = Array.from({length: pages}, (_, i) =>
      `<button class="${i+1===page?'active':''}" onclick="loadAdminSongs(${i+1})">${i+1}</button>`).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="loading-row">Error: ${err.message}</td></tr>`;
  }
}

window.adminDeleteSong = async (songId) => {
  if (!confirm('Permanently delete this song?')) return;
  try {
    await callApi('DELETE', `/admin/songs/${songId}`, null, S.adminToken);
    notify('Song deleted.', 'success');
    loadAdminSongs();
  } catch (err) {
    notify('Delete failed: ' + err.message, 'error');
  }
};

async function loadAdminSettings() {
  try {
    const s = await callApi('GET', '/admin/settings', null, S.adminToken);
    $('setting-lyrics-model').value   = s.lyrics_model   || '';
    $('setting-cover-model').value    = s.cover_model    || '';
    $('setting-audio-model').value    = s.audio_model    || '';
    $('setting-face-threshold').value = s.face_threshold || 0.55;
    $('setting-max-songs').value      = s.max_songs_per_user || 100;
  } catch (err) {
    notify('Settings load failed: ' + err.message, 'error');
  }
}

window.saveSettings = async () => {
  const payload = {
    lyrics_model:      $('setting-lyrics-model').value,
    cover_model:       $('setting-cover-model').value,
    audio_model:       $('setting-audio-model').value,
    face_threshold:    parseFloat($('setting-face-threshold').value),
    max_songs_per_user:parseInt($('setting-max-songs').value),
  };
  try {
    await callApi('POST', '/admin/settings', payload, S.adminToken);
    notify('Settings saved! ✅', 'success');
  } catch (err) {
    notify('Save failed: ' + err.message, 'error');
  }
};

async function loadAdminLogs() {
  // Logs are loaded as part of overview analytics — reuse
  loadAdminOverview();
}

// ══════════════════════════════════════════════════════════════════════════════
// NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════════════════
const NOTIF_ICONS = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };

function notify(msg, type = 'info', duration = 4000) {
  const container = $('notifications');
  const id = 'n' + Date.now();
  const el = document.createElement('div');
  el.className = `notif ${type}`;
  el.id = id;
  el.innerHTML = `<span class="notif-icon">${NOTIF_ICONS[type] || 'ℹ️'}</span>
                  <span class="notif-text">${escHtml(msg)}</span>`;
  el.onclick = () => dismissNotif(el);
  container.appendChild(el);
  setTimeout(() => dismissNotif(el), duration);
}

function dismissNotif(el) {
  el.classList.add('out');
  setTimeout(() => el.remove(), 260);
}

// ══════════════════════════════════════════════════════════════════════════════
// MODAL HELPERS
// ══════════════════════════════════════════════════════════════════════════════
window.onModalOverlayClick = (e, modalId) => {
  if (e.target.id === modalId) closeModal(modalId);
};

function openModal(id) {
  $(`${id}`).classList.remove('hidden');
}

function closeModal(id) {
  $(`${id}`).classList.add('hidden');
}

// ══════════════════════════════════════════════════════════════════════════════
// SCREEN HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function showScreen(id) { $(id).classList.remove('hidden'); }
function hideScreen(id) { $(id).classList.add('hidden'); }

// ══════════════════════════════════════════════════════════════════════════════
// API HELPER
// ══════════════════════════════════════════════════════════════════════════════
async function callApi(method, endpoint, body = null, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const opts = { method, headers };
  if (body && method !== 'GET') opts.body = JSON.stringify(body);

  const res = await fetch(`${API}${endpoint}`, opts);

  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`;
    try { const j = await res.json(); errMsg = j.error || errMsg; } catch {}
    throw new Error(errMsg);
  }
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════════════
// TOKEN STORAGE
// ══════════════════════════════════════════════════════════════════════════════
function saveToken(token, faceId) {
  localStorage.setItem('dhun_token', token);
  localStorage.setItem('dhun_face_id', faceId);
}
function loadToken() {
  return {
    token:  localStorage.getItem('dhun_token'),
    faceId: localStorage.getItem('dhun_face_id'),
  };
}
function clearToken() {
  localStorage.removeItem('dhun_token');
  localStorage.removeItem('dhun_face_id');
}

// ══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════════════════
function $(id) { return document.getElementById(id); }

function fmtTime(sec) {
  if (!sec || isNaN(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') { e.preventDefault(); if (S.curSong) togglePlay(); }
  if (e.code === 'ArrowRight') nextSong();
  if (e.code === 'ArrowLeft')  prevSong();
  if (e.code === 'Escape') {
    ['generator-modal','lyrics-modal','expanded-player','delete-modal','admin-pin-modal'].forEach(closeModal);
    if (!$('registration-details-modal').classList.contains('hidden')) cancelRegistrationDetails();
  }
});
