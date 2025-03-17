const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const resultElement = document.getElementById('result');
const uploadedImage = document.getElementById('uploadedImage');
const predictionElement = document.getElementById('prediction');
const suggestionText = document.getElementById('suggestion-text');
const suggestionLink = document.getElementById('suggestion-link');

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        predictionElement.textContent = "Please select an image first.";
        return;
    }

    // Display the uploaded image
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            predictionElement.textContent = `Prediction: ${result.class}, Confidence: ${result.confidence.toFixed(2)}%`;

            if (result.class === "Healthy") {
                suggestionText.textContent = "The image is healthy. Keep up the good work!";
                suggestionLink.href = "https://cropaia.com/blog/guide-to-potato-cultivation/"; 
                suggestionLink.style.display = "block";
            } else if (result.class === "Early Blight") {
                suggestionText.textContent = "The image shows signs of Early Blight.";
                suggestionLink.href = "https://www.cropscience.bayer.us/articles/cp/early-blight-potatoes"; 
                suggestionLink.style.display = "block";
            } else if (result.class === "Late Blight") {
                suggestionText.textContent = "The image shows signs of Late Blight.";
                suggestionLink.href = "https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato"; 
                suggestionLink.style.display = "block";
            } else {
                suggestionText.textContent = "Unable to classify the image.";
                suggestionLink.style.display = "none";
            }
        } else {
            predictionElement.textContent = "Error: Unable to get prediction.";
        }
    } catch (error) {
        predictionElement.textContent = `Error: ${error.message}`;
    }
});