const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture');
const imageDataInput = document.getElementById('image-data');
const form = document.getElementById('form');
const result = document.getElementById('result');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream);

// Capture image
captureBtn.addEventListener('click', () => {
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 224, 224);
    const dataURL = canvas.toDataURL('image/jpeg');
    imageDataInput.value = dataURL;
});

// Submit form via AJAX
form.addEventListener('submit', function (e) {
    e.preventDefault();

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(new FormData(form))
    })
    .then(res => res.text())
    .then(text => {
        result.innerText = text;
    });
});
