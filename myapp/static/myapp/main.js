document.getElementById('upload-btn').addEventListener('click', function() {
  const fileInput = document.getElementById('csvFile');
  fileInput.click();  // Opens the prompt window to browse files
});

document.getElementById('csvFile').addEventListener('change', function() {
  const file = this.files[0];
  
  if (!file) {
    alert("Please select a CSV file.");
    return;
  }

  const reader = new FileReader();

  reader.onload = function(event) {
    const csvData = event.target.result;
    const processedData = preprocessCSV(csvData);
    
    displayProcessedData(processedData);
    enableDownload(processedData);
  };

  reader.readAsText(file);
});

function preprocessCSV(csvData) {
  // You can add your own preprocessing logic here
  const rows = csvData.split('\n').map(row => row.trim());
  return rows.join('\n'); // For simplicity, we're just returning the same data
}

function displayProcessedData(data) {
  const outputSection = document.getElementById('output-section');
  outputSection.style.display = 'block';
  document.getElementById('processed-data').textContent = data;
}

function enableDownload(data) {
  const downloadButton = document.getElementById('download-btn');
  const blob = new Blob([data], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);

  downloadButton.href = url;
  downloadButton.download = 'preprocessed_data.csv';
}
