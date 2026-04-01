// scripts.js

// Function to generate a random color
function getRandomColor(opacity) {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

// Function to handle author identification and profile visualization
function identifyAuthor(doc) {
    const selectedModel = document.querySelector('input[name="model_choice"]:checked').value;

    fetch('/identify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `unknown_doc=${doc}&selected_model=${selectedModel}`
    })
    .then(response => response.json())
    .then(data => {
        // Display the identified author
        document.getElementById('author-result').innerText = 'The author of ' + doc + ' is: ' + data.author;

        const profileSection = document.getElementById('profileSection');
        profileSection.innerHTML = ''; // Clear previous profiles

        if (selectedModel === 'model1' || selectedModel === 'both') {
            // Display Model 1 profile (Single-sentence classification)
            displayProfile('Model 1 Illocutionary Force Profile', data.unknown_profile_model1, data.author_profiles, data.labels);
        }

        if (selectedModel === 'model2' || selectedModel === 'both') {
            // Display Model 2 profile (Sentence-pair classification)
            displayProfile('Model 2 Illocutionary Force Profile', data.unknown_profile_model2, data.author_profiles, data.labels_pairs);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to display the profiles as bar charts
function displayProfile(profileTitle, unknownProfile, authorProfiles, labels) {
    const profileSection = document.getElementById('profileSection');

    // Check if the profiles exist
    if (!unknownProfile || !authorProfiles) {
        console.error('Unknown profile or author profiles are missing:', unknownProfile, authorProfiles);
        return; // Exit the function if data is missing
    }

    // Log the labels to check their structure
    console.log('Labels:', labels); // Debugging: Check the labels being passed

    // Convert the labels object to an array of values
    const labelsArray = Object.values(labels);
    
    // Add a title for the profile
    const title = document.createElement('h3');
    title.innerText = profileTitle;
    profileSection.appendChild(title);

    // Create a canvas for the chart
    const canvas = document.createElement('canvas');
    canvas.id = profileTitle.replace(/\s+/g, '') + 'Chart'; // Generate unique ID
    profileSection.appendChild(canvas);

    const datasets = [];
    let colorIndex = 0;

    // Loop through known author profiles and display them
    Object.keys(authorProfiles).forEach(author => {
        const modelLabels = authorProfiles[author][profileTitle.includes('Model 1') ? 'model1_labels' : 'model2_labels'];
        const frequencies = new Array(labelsArray.length).fill(0);

        // Count how often each illocutionary force appears
        if (modelLabels) {
            modelLabels.forEach(label => {
                if (label < frequencies.length) {
                    frequencies[label]++;
                } else {
                    console.error('Label index out of range:', label);
                }
            });
        }

        datasets.push({
            label: author,
            data: frequencies,
            backgroundColor: getRandomColor(0.2), // Use random colors with opacity 0.2
            borderColor: getRandomColor(1),       // Use random colors with full opacity for border
            borderWidth: 1
        });
    });

    // Process the unknown document profile
    const unknownFrequencies = new Array(labelsArray.length).fill(0);
    if (unknownProfile) {
        unknownProfile.forEach(label => {
            if (label < unknownFrequencies.length) {
                unknownFrequencies[label]++;
            } else {
                console.error('Label index out of range for unknown profile:', label);
            }
        });
    }

    datasets.push({
        label: 'Unknown Document',
        data: unknownFrequencies,
        backgroundColor: 'rgba(255, 159, 64, 0.2)', // Keep the orange color for unknown doc
        borderColor: 'rgba(255, 159, 64, 1)',       // Orange border
        borderWidth: 1
    });

    // Create the chart using Chart.js
    const ctx = document.getElementById(canvas.id).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labelsArray, // Use the converted array of labels
            datasets: datasets
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
