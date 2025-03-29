document.addEventListener("DOMContentLoaded", () => {
    const uploadBox = document.getElementById("uploadBox");
    const fileInput = document.getElementById("fileInput");
    const uploadText = document.getElementById("uploadText");
    const uploadedImage = document.getElementById("uploadedImage");
    const submitBtn = document.getElementById("submitBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const arrow = document.getElementById("arrow");

    // Create Dark Overlay for Cursor Effect
    const darkOverlay = document.createElement("div");
    darkOverlay.id = "dark-overlay";
    document.body.appendChild(darkOverlay);

    // Click to Upload
    uploadBox.addEventListener("click", () => fileInput.click());

    // Drag & Drop Feature
    uploadBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadBox.style.border = "3px solid white";
        uploadBox.style.background = "rgba(255, 255, 255, 0.3)";
    });

    uploadBox.addEventListener("dragleave", () => {
        uploadBox.style.border = "3px dashed #bbb";
        uploadBox.style.background = "white";
    });
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = "block";
        uploadText.style.display = "none";
    };
    reader.readAsDataURL(file);
}
    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadBox.style.border = "3px dashed #bbb";
        uploadBox.style.background = "white";

        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });

    // File Input Change Event
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleFileUpload(file);
    });

    function handleFileUpload(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = "block";
            uploadedImage.classList.remove("hidden");
            uploadText.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

// Submit Button
submitBtn.addEventListener("click", async () => {
    submitBtn.classList.add("hidden");
    arrow.style.display = "block"; // Show large arrow

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0]; // Get selected file

    if (!file) {
        alert("Please select an image first!");
        submitBtn.classList.remove("hidden"); // Show button again
        arrow.style.display = "none"; // Hide arrow
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8080/upload", { // Fixed endpoint
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        alert(`File uploaded successfully!\nFilename: ${data.filename}`);
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        submitBtn.classList.remove("hidden"); // Show button again
        arrow.style.display = "none"; // Hide arrow
    }
});

    // Reset Button
    refreshBtn.addEventListener("click", () => {
        uploadedImage.classList.add("hidden");
        uploadedImage.style.display = "none";
        uploadText.style.display = "block";
        submitBtn.classList.remove("hidden");
        arrow.style.display = "none";
        fileInput.value = ""; // Clear file input
    });

    // Cursor Proximity Effect - Makes Elements Shrink Near Cursor
    document.addEventListener("mousemove", (e) => {
        const x = e.clientX;
        const y = e.clientY;

        // Check distance from the cursor to elements
        const elements = [uploadBox, submitBtn, refreshBtn];
        elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const elX = rect.left + rect.width / 2;
            const elY = rect.top + rect.height / 2;
            const distance = Math.sqrt((x - elX) ** 2 + (y - elY) ** 2);

            if (distance < 50) { // Very small range for shrink effect
                el.classList.add("shrink");
            } else {
                el.classList.remove("shrink");
            }
        });

        // Darkening Effect on Background - Only 1.5x Cursor Size
        const cursorSize = 10; // Default cursor size
        darkOverlay.style.opacity = "1";
        darkOverlay.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(0, 0, 0, 0) ${cursorSize}px, rgba(0, 0, 0, 0.5) ${cursorSize * 1.5}px)`;
    });

    document.addEventListener("mouseleave", () => {
        darkOverlay.style.opacity = "0"; // Fade effect when cursor leaves screen
    });
});
