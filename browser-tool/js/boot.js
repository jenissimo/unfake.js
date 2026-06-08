// Sync boot shell — upload/preview works before ES modules load.
(function () {
  const MAX_BYTES = 10 * 1024 * 1024;

  function setStatus(text, className) {
    const el = document.getElementById('boot-status');
    if (!el) return;
    el.textContent = text;
    el.classList.remove('boot-error', 'boot-warn', 'boot-ok');
    if (className) el.classList.add(className);
  }

  function $(id) {
    return document.getElementById(id);
  }

  const uploadArea = $('upload-area');
  const fileInput = $('file-input');
  const chooseBtn = $('choose-file-btn');
  const imageDisplay = $('image-display');
  const originalImage = $('original-image');
  const processBtn = $('process-btn');

  if (!uploadArea || !fileInput) return;

  window.__unfakeBoot = {
    file: null,
    dataUrl: null,
    ready: false,
    resetUpload,
  };

  setStatus('Shell OK — choose an image', 'boot-ok');

  function openPicker() {
    fileInput.click();
  }

  if (chooseBtn) {
    chooseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openPicker();
    });
  }

  uploadArea.addEventListener('click', (e) => {
    if (e.target === chooseBtn || e.target.closest('#choose-file-btn')) return;
    openPicker();
  });

  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer?.files;
    if (files?.length) loadFile(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files?.length) loadFile(files[0]);
    e.target.value = '';
  });

  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        if (file) loadFile(file);
        break;
      }
    }
  });

  window.addEventListener('unfake:reset-upload', resetUpload);

  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file (PNG, JPG, JPEG, WebP)');
      return;
    }
    if (file.size > MAX_BYTES) {
      alert('File size must be less than 10MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;
      const img = new Image();
      img.onload = () => {
        originalImage.src = dataUrl;
        uploadArea.style.display = 'none';
        imageDisplay.style.display = 'block';
        if (processBtn) processBtn.style.display = 'block';

        window.__unfakeBoot.file = file;
        window.__unfakeBoot.dataUrl = dataUrl;
        window.__unfakeBoot.ready = true;

        window.dispatchEvent(new CustomEvent('unfake:file', {
          detail: { file, dataUrl, width: img.naturalWidth, height: img.naturalHeight, image: img },
        }));
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    window.__unfakeBoot.file = null;
    window.__unfakeBoot.dataUrl = null;
    window.__unfakeBoot.ready = false;
    fileInput.value = '';
    uploadArea.style.display = 'flex';
    imageDisplay.style.display = 'none';
    if (processBtn) processBtn.style.display = 'none';
    if (originalImage) originalImage.removeAttribute('src');
  }
})();
