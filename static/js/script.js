// Document upload and processing functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Show file display and hide upload label
        document.getElementById('uploadLabel').style.display = 'none';
        document.getElementById('fileDisplay').style.display = 'block';
        document.getElementById('fileName').textContent = file.name;
        
        // Show process button
        document.getElementById('processButton').style.display = 'block';
    }
}

function removeFile() {
    // Clear file input
    const fileInput = document.getElementById('fileInput');
    fileInput.value = '';
    
    // Hide file display and show upload label
    document.getElementById('fileDisplay').style.display = 'none';
    document.getElementById('uploadLabel').style.display = 'block';
    
    // Hide process button
    document.getElementById('processButton').style.display = 'none';
    
    // Clear error messages
    document.getElementById('error').innerHTML = '';
}

function processDocument() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        document.getElementById('error').innerHTML = 'Please select a file to upload.';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    document.getElementById('loading').style.display = 'block';
    document.getElementById('fileDisplay').style.opacity = '0.5';
    document.getElementById('error').innerHTML = '';

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('fileDisplay').style.opacity = '1';
        if (data.error) {
            document.getElementById('error').innerHTML = data.error;
        } else {
            displayResults(data);
            sessionStorage.setItem('currentDocument', file.name);
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('fileDisplay').style.opacity = '1';
        document.getElementById('error').innerHTML = 'Error processing document. Please try again.';
        console.error('Error:', error);
    });
}

function displayResults(data) {
    try {
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Update content
        if (data.summary) {
            document.getElementById('summary').innerHTML = `<p>${data.summary}</p>`;
        }
        
        if (data.highlights) {
            document.getElementById('keyHighlights').innerHTML = 
                data.highlights.map(h => `<p>${h}</p>`).join('');
        }
        
        if (data.insights) {
            document.getElementById('actionableInsights').innerHTML = 
                data.insights.map(i => `<p>${i}</p>`).join('');
        }
        
        // Display evaluation metrics
        if (data.evaluation && data.evaluation.metrics) {
            const metricsHtml = Object.entries(data.evaluation.metrics)
                .map(([label, value]) => `
                    <div class="metric-item">
                        <div class="metric-label">${label}</div>
                        <div class="metric-value">${value}</div>
                    </div>
                `).join('');
            document.getElementById('evaluationMetrics').innerHTML = metricsHtml;
        }
        
        // Display next steps
        if (data.evaluation && data.evaluation.next_steps) {
            const nextStepsHtml = `
                <ul>
                    ${data.evaluation.next_steps.map(step => `
                        <li>${step}</li>
                    `).join('')}
                </ul>
            `;
            document.getElementById('nextSteps').innerHTML = nextStepsHtml;
        }
        
    } catch (error) {
        console.error('Error displaying results:', error);
        document.getElementById('error').innerHTML = 'Error displaying results. Please try again.';
    }
}

function removeDocument() {
    try {
        // Clear the file input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Hide all sections
        const sections = [
            'summarySection',
            'highlightsSection',
            'insightsSection',
            'askSection'
        ];
        
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.display = 'none';
            }
        });
        
        // Clear all content
        const elements = [
            'summary',
            'keyHighlights',
            'legalReferences',
            'actionableInsights',
            'chatHistory'
        ];
        
        elements.forEach(elementId => {
            const element = document.getElementById(elementId);
            if (element) {
                element.innerHTML = '';
            }
        });
        
        // Clear error messages
        const errorElement = document.getElementById('error');
        if (errorElement) {
            errorElement.innerHTML = '';
        }
        
        // Reset chat input
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.value = '';
        }
        
        // Hide remove button
        const removeButton = document.getElementById('removeButton');
        if (removeButton) {
            removeButton.style.display = 'none';
        }
        
        // Show upload section
        const uploadSection = document.getElementById('uploadSection');
        if (uploadSection) {
            uploadSection.style.display = 'block';
        }
        
        // Clear session storage
        sessionStorage.removeItem('currentDocument');
        
        console.log('Document and all outputs cleared');
        
    } catch (error) {
        console.error('Error removing document:', error);
        const errorElement = document.getElementById('error');
        if (errorElement) {
            errorElement.innerHTML = 'Error removing document. Please try again.';
        }
    }
}

// Chat functionality
function askQuestion() {
    const chatInput = document.getElementById('chatInput');
    const question = chatInput.value;
    if (!question.trim()) return;

    const chatHistory = document.getElementById('chatHistory');
    
    // Add user message
    chatHistory.innerHTML += `
        <p class="user-message">
            <strong>You:</strong> ${question}
        </p>
    `;
    
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            chatHistory.innerHTML += `
                <p class="error-message">
                    ${data.error}
                </p>
            `;
        } else {
            chatHistory.innerHTML += `
                <p class="assistant-message">
                    <strong>Assistant:</strong> ${data.response}
                </p>
            `;
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        chatHistory.innerHTML += `
            <p class="error-message">
                Error getting response. Please try again.
            </p>
        `;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    }

    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    const removeButton = document.getElementById('removeFile');
    if (removeButton) {
        removeButton.addEventListener('click', removeFile);
    }

    const processButton = document.getElementById('processButton');
    if (processButton) {
        processButton.addEventListener('click', processDocument);
    }
});