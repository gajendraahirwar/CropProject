const form = document.getElementById("crop-form");
const button = document.getElementById("predict-btn");
const placeholder = document.getElementById("placeholder");
const output = document.getElementById("output");
const errorBox = document.getElementById("error");
const cropName = document.getElementById("crop-name");
const topList = document.getElementById("top-list");

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stop the page from reloading

  // 1. Read the 7 values from the form and turn them into numbers
  const payload = {};
  new FormData(form).forEach((value, key) => {
    payload[key] = Number(value);
  });

  button.disabled = true;
  button.textContent = "Predicting...";
  errorBox.hidden = true;

  try {
    // 2. Send the values to the Flask backend as JSON
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    // 3. Show the answer that came back from the model
    showResult(data);
  } catch (err) {
    placeholder.hidden = true;
    output.hidden = true;
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Predict Best Crop";
  }
});

function showResult(data) {
  placeholder.hidden = true;
  cropName.textContent = data.crop;

  topList.replaceChildren();
  data.top.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.crop} - ${item.probability}%`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${item.probability}%`;
    bar.appendChild(fill);
    li.appendChild(bar);
    topList.appendChild(li);
  });

  output.hidden = false;
}